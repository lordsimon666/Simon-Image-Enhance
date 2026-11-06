""""KI-Korrektur": automatic white balance + colour curves.

Runs completely offline. The analysis searches for achromatic surfaces whose colour lies on the
locus of physically plausible light sources (black-body radiators 2000-14000 K) and
blends that with classic colour-constancy estimators (grey world, shades of grey,
grey edge, white patch), weighted by how confident the grey-pixel search is. Tone curves are derived from the luminance histogram
(black/white point, midtone gamma, adaptive S-contrast) and per-channel histograms
(shadow / highlight colour casts).

The result ONLY touches ``temp``, ``tint`` and ``curves`` - exactly the corrections
the user asked the AI feature to be limited to. All other sliders stay untouched, and
everything it sets can be fine-tuned afterwards.
"""

from __future__ import annotations

import math

import numpy as np

from .processing import (
    LUM_WEIGHTS, NEUTRAL_TEMP, Params, Renderer, blackbody_rgb, eval_curve, identity_curve,
    solve_white_balance,
)


def _downsample(img: np.ndarray, max_side: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_side)))
    return img[::step, ::step, :3].astype(np.float32)


def _downsample_mask(mask, max_side: int):
    if mask is None:
        return None
    h, w = mask.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_side)))
    return mask[::step, ::step]


def _chroma(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=np.float64), 1e-6)
    return np.log(v / v[..., 1:2])  # log(r/g), 0, log(b/g)


_LOCUS = None
_DEBUG = False


def _locus():
    """Chromaticities (log r/g, log b/g) of grey surfaces under Planckian light
    between 2000 K and 14000 K, relative to the neutral reference."""
    global _LOCUS
    if _LOCUS is None:
        temps = 1e6 / np.linspace(1e6 / 2000.0, 1e6 / 14000.0, 160)
        ref = blackbody_rgb(NEUTRAL_TEMP)
        pts = np.array([_chroma(blackbody_rgb(t) / ref)[[0, 2]] for t in temps])
        _LOCUS = (temps, pts.astype(np.float32))
    return _LOCUS


def _statistical_estimate(img, v, lv, grad_mask=None):
    """Classic colour-constancy estimators combined in log-chroma space."""
    estimates = [
        (v.mean(axis=0), 0.15),                                    # grey world
        (np.mean(v ** 6, axis=0) ** (1 / 6), 0.30),                # shades of grey
        (v[lv >= np.quantile(lv, 0.98)].mean(axis=0), 0.20),       # white patch
    ]
    gx = np.abs(np.diff(img, axis=1))[:-1]
    gy = np.abs(np.diff(img, axis=0))[:, :-1]
    grad = np.sqrt(gx ** 2 + gy ** 2).reshape(-1, 3).astype(np.float64)
    if grad.size:
        estimates.append((np.mean(grad ** 6, axis=0) ** (1 / 6) + 1e-9, 0.35))  # grey edge
    chroma = np.zeros(3)
    wsum = 0.0
    for e, w in estimates:
        if np.all(np.isfinite(e)) and np.all(e > 0):
            chroma += w * _chroma(e)
            wsum += w
    return chroma / max(wsum, 1e-9)


_GRID = None
_TINT_STEPS = np.linspace(-0.075, 0.075, 7)  # log green offset of the light source


def _grid():
    """2-D grid of candidate light sources: Planckian temperature x tint offset."""
    global _GRID
    if _GRID is None:
        temps = 1e6 / np.linspace(1e6 / 2000.0, 1e6 / 14000.0, 64)
        ref = blackbody_rgb(NEUTRAL_TEMP)
        base = np.array([_chroma(blackbody_rgb(t) / ref)[[0, 2]] for t in temps])
        # more green in the light lowers both log(r/g) and log(b/g)
        pts = base[:, None, :] - _TINT_STEPS[None, :, None]
        _GRID = (temps, pts.reshape(-1, 2).astype(np.float32))
    return _GRID


def _grey_pixel_estimate(pix_chroma, lv):
    """Find pixels whose colour matches a physically plausible light source
    (black body 2000-14000 K, with a moderate green/magenta offset) - these are most
    likely achromatic surfaces (walls, clouds, asphalt, paper ...) lit by the scene
    illuminant. Returns (chroma, confidence)."""
    if len(pix_chroma) > 24000:  # keep the N x grid distance matrix small
        step = int(math.ceil(len(pix_chroma) / 24000))
        pix_chroma, lv = pix_chroma[::step], lv[::step]
    temps, grid = _grid()
    nt, ns = len(temps), len(_TINT_STEPS)
    c = pix_chroma[:, [0, 2]].astype(np.float32)
    d = np.sqrt(((c[:, None, :] - grid[None, :, :]) ** 2).sum(-1))  # N x (nt*ns)
    k = d.argmin(axis=1)
    dist = d[np.arange(len(c)), k]
    sel = dist < 0.06
    if sel.sum() < 20:
        return None, 0.0
    it, is_ = k[sel] // ns, k[sel] % ns
    mired = 1e6 / temps[it]
    tint = _TINT_STEPS[is_]
    # brighter surfaces are more reliable; mild priors favour common light sources
    # (near daylight, near the Planckian locus). They only matter when candidates
    # disagree, e.g. a wooden desk that looks like "grey under tungsten".
    prior_t = np.exp(-0.5 * ((mired - 1e6 / NEUTRAL_TEMP) / 110.0) ** 2)
    prior_s = np.exp(-0.5 * (tint / 0.035) ** 2)
    w_raw = np.clip(lv[sel], 0.02, 0.9) * (1.0 - dist[sel] / 0.06)
    w = w_raw * (0.25 + 0.75 * prior_t) * (0.05 + 0.95 * prior_s)
    # robust: dominant mode of the smoothed 2-D distribution. (A median would land
    # between two light sources in mixed lighting.)
    hist = np.zeros((nt, ns))
    np.add.at(hist, (it, is_), w)
    kt = np.exp(-0.5 * (np.arange(-4, 5) / 2.0) ** 2)
    ks = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2)
    hist = np.apply_along_axis(lambda r: np.convolve(r, kt, mode="same"), 0, hist)
    hist = np.apply_along_axis(lambda r: np.convolve(r, ks, mode="same"), 1, hist)
    mt, ms = np.unravel_index(int(np.argmax(hist)), hist.shape)
    # ... then average the chroma of the candidates near that mode
    near = (np.abs(it - mt) <= 4) & (np.abs(is_ - ms) <= 1)
    if near.sum() < 10:
        near = np.ones_like(it, dtype=bool)
    chroma = np.average(pix_chroma[sel][near], axis=0, weights=w[near])
    # amount: ~1.5 % of the image being plausible grey surfaces is plenty
    n_good = int((lv[sel][near] > 0.04).sum())
    amount = float(np.clip(n_good / (len(lv) * 0.015), 0.0, 1.0))
    # unanimity: in mixed light the candidates split between light sources
    concentration = float(w[near].sum() / max(w.sum(), 1e-9))
    unanimity = float(np.clip((concentration - 0.25) / 0.4, 0.0, 1.0))
    confidence = amount * unanimity
    # a best match at the edge of the tint range is suspicious: typically large
    # coloured areas (foliage!) mimicking grey under strangely tinted light
    if abs(_TINT_STEPS[ms]) >= 0.07:
        confidence *= 0.3
    if _DEBUG:
        print(f"grey-pixel: mode {temps[mt]:.0f} K tint {_TINT_STEPS[ms]:+.2f}  amount {amount:.2f} "
              f"concentration {concentration:.2f} -> conf {confidence:.2f}")
    return chroma, confidence


def estimate_illuminant(linear: np.ndarray, mask=None) -> np.ndarray:
    """Return the estimated scene illuminant as linear RGB (G normalised to 1)."""
    return estimate_illuminant_conf(linear, mask)[0]


def estimate_illuminant_conf(linear: np.ndarray, mask=None) -> tuple[np.ndarray, float]:
    """Estimated illuminant plus a confidence 0..1 (how many reliable grey
    surfaces supported the estimate)."""
    img = _downsample(linear, 256)
    px = img.reshape(-1, 3).astype(np.float64)
    lum = px @ LUM_WEIGHTS
    valid = (px.max(axis=1) < 0.97) & (lum > 0.01)
    m = _downsample_mask(mask, 256)
    if m is not None and m.sum() > 200:  # only analyse the foreground (key removed)
        valid &= m.reshape(-1)
    if valid.sum() < 50:
        valid = lum > 0
    if valid.sum() < 10:
        return np.ones(3), 0.0
    v = px[valid]
    lv = lum[valid]

    pix_chroma = _chroma(v)
    # Strongly coloured pixels (screens, LEDs, neon, saturated objects) cannot be
    # lit surfaces of a neutral colour - keep them out of the statistical estimators.
    temps, locus = _locus()
    c2 = pix_chroma[:, [0, 2]].astype(np.float32)
    dmin = np.sqrt(((c2[:, None, :] - locus[None, :, :]) ** 2).sum(-1)).min(axis=1)
    calm = dmin < 0.35
    if calm.sum() >= max(50, 0.1 * len(v)):
        stat = _statistical_estimate(img, v[calm], lv[calm], grad_mask=None)
    else:
        stat = _statistical_estimate(img, v, lv)

    grey, conf = _grey_pixel_estimate(pix_chroma, lv)
    if grey is None:
        return np.exp(stat), 0.0
    wg = 0.25 + 0.7 * conf  # 0.25 .. 0.95 weight on the grey-pixel estimate
    # when both methods disagree strongly the statistics are usually fooled by a
    # dominant colour (lawn, sky, a monitor ...) -> trust the grey surfaces
    disagreement = float(np.hypot(*(grey - stat)[[0, 2]]))
    if disagreement > 0.2 and conf >= 0.6:
        wg = max(wg, 0.9)
    return np.exp(wg * grey + (1 - wg) * stat), conf


def auto_white_balance(linear: np.ndarray, mask=None) -> tuple[float, float]:
    illum, conf = estimate_illuminant_conf(linear, mask)
    temp, tint = solve_white_balance(illum)
    # The camera already applied its own white balance ("as shot"). With little
    # evidence (few reliable grey surfaces, mixed light) we only move part of the
    # way - a wrong strong correction looks much worse than a mild one.
    trust = 0.5 + 0.5 * conf
    mired_ref = 1e6 / NEUTRAL_TEMP
    temp = 1e6 / (mired_ref + (1e6 / temp - mired_ref) * trust)
    tint *= trust
    # keep a little of the original mood for very warm / very cool light (sunsets,
    # candle light, blue hour) - photographers rarely want those fully neutralised.
    mired = 1e6 / temp
    shift = mired - mired_ref
    keep = 0.12 if abs(shift) > 60 else 0.0
    temp = 1e6 / (mired_ref + shift * (1 - keep))
    # real light sources rarely deviate far from the Planckian locus -> limit tint;
    # without grey surfaces as evidence the statistics are least reliable for tint
    limit = 10.0 + 70.0 * conf
    return float(temp), float(np.clip(tint * 0.9, -limit, limit))


def _blend_points(points, strength: float):
    return [[x, x + (y - x) * strength] if i not in (0, len(points) - 1)
            else [x * strength if x < 0.5 else 1 - (1 - x) * strength,
                  y * strength if y < 0.5 else 1 - (1 - y) * strength]
            for i, (x, y) in enumerate(points)]


def auto_curves(linear: np.ndarray, params: Params, mask=None) -> dict:
    """Derive master + per-channel curves for an image rendered with ``params``."""
    base = params.copy()
    base.curves = {c: identity_curve() for c in base.curves}
    img = Renderer(base).apply(_downsample(linear, 512).copy())
    px = img.reshape(-1, 3)
    m = _downsample_mask(mask, 512)
    if m is not None and m.sum() > 500:
        px = px[m.reshape(-1)]
    lum = px @ LUM_WEIGHTS

    bp = float(np.clip(np.quantile(lum, 0.003), 0.0, 0.06))
    wp = float(np.clip(np.quantile(lum, 0.997), 0.88, 1.0))
    if wp - bp < 0.3:
        bp, wp = 0.0, 1.0
    ln = np.clip((lum - bp) / (wp - bp), 0, 1)
    med = float(np.clip(np.median(ln), 0.02, 0.98))
    target = 0.46
    # move only part of the way to the target so dark / bright moods survive
    gamma = math.log(target) / math.log(med)
    gamma = float(np.clip(1.0 + 0.6 * (gamma - 1.0), 0.75, 1.35))
    std = float(ln.std())
    s_amt = float(np.clip((0.21 - std) * 0.35, 0.0, 0.045))

    def x_at(f):
        return bp + f * (wp - bp)

    master = [
        [bp, 0.0],
        [x_at(0.25), float(np.clip(0.25 ** gamma - s_amt, 0.01, 0.99))],
        [x_at(0.5), float(0.5 ** gamma)],
        [x_at(0.75), float(np.clip(0.75 ** gamma + s_amt, 0.01, 0.99))],
        [wp, 1.0],
    ]
    for i in range(1, len(master)):  # keep y monotone
        master[i][1] = max(master[i][1], master[i - 1][1])

    # per-channel shadow / highlight cast removal on the master-corrected image
    # (measured only on near-neutral pixels - a blue sky or a red car is not a cast)
    corrected = eval_curve(master, px)
    mx = corrected.max(axis=1)
    neutral = (mx - corrected.min(axis=1)) < 0.12 * np.maximum(mx, 0.05)
    curves = {"rgb": master}
    if neutral.sum() < max(200, 0.03 * len(px)):
        for c in "rgb":
            curves[c] = identity_curve()
        curves["rgb"] = [[round(x, 4), round(y, 4)] for x, y in master]
        return curves
    nl = corrected[neutral]
    nlum = nl @ LUM_WEIGHTS
    dark = nl[nlum <= np.quantile(nlum, 0.1)].mean(axis=0)
    bright = nl[nlum >= np.quantile(nlum, 0.9)].mean(axis=0)
    for i, c in enumerate("rgb"):
        dlo = float(np.clip((dark[i] - dark.mean()) * 0.5, -0.03, 0.03))
        dhi = float(np.clip((bright[i] - bright.mean()) * 0.5, -0.03, 0.03))
        # channel too high in the shadows -> clip it down; too low -> lift it
        start = [dlo, 0.0] if dlo > 0 else [0.0, -dlo]
        # channel too high in the highlights -> pull it down; too low -> boost it
        end = [1.0, 1.0 - dhi] if dhi > 0 else [1.0 + dhi, 1.0]
        curves[c] = [[round(start[0], 4), round(start[1], 4)], [round(end[0], 4), round(end[1], 4)]]
        if abs(dlo) < 0.003 and abs(dhi) < 0.003:
            curves[c] = identity_curve()
    curves["rgb"] = [[round(x, 4), round(y, 4)] for x, y in master]
    return curves


def ai_correct(linear: np.ndarray, params: Params, strength: float = 1.0) -> Params:
    """Return a copy of ``params`` with AI white balance and curves applied."""
    strength = float(np.clip(strength, 0.0, 1.0))
    out = params.copy()
    mask = None
    from .greenscreen import KeyMatte, has_key
    if has_key(params):
        # with a removed backdrop only the subject counts (the green screen would
        # otherwise dominate the statistics)
        linear = _downsample(linear, 768)
        linear, alpha = KeyMatte(params, linear.shape[0]).apply(linear)
        mask = alpha > 0.5
    temp, tint = auto_white_balance(linear, mask)
    mired_ref = 1e6 / NEUTRAL_TEMP
    out.temp = 1e6 / (mired_ref + (1e6 / temp - mired_ref) * strength)
    out.tint = tint * strength
    curves = auto_curves(linear, out, mask)
    if strength < 1.0:
        curves = {c: _blend_points(p, strength) for c, p in curves.items()}
    out.curves = curves
    return out
