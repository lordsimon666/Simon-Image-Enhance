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


def _chroma(v: np.ndarray) -> np.ndarray:
    v = np.maximum(np.asarray(v, dtype=np.float64), 1e-6)
    return np.log(v / v[..., 1:2])  # log(r/g), 0, log(b/g)


_LOCUS = None


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


def _statistical_estimate(img, v, lv):
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


def _grey_pixel_estimate(pix_chroma, lv):
    """Find pixels whose colour lies on the locus of physically plausible light
    sources - these are most likely achromatic surfaces (walls, clouds, asphalt,
    paper ...) lit by the scene illuminant. Returns (chroma, confidence)."""
    temps, locus = _locus()
    c = pix_chroma[:, [0, 2]].astype(np.float32)
    d = np.sqrt(((c[:, None, :] - locus[None, :, :]) ** 2).sum(-1))  # N x L
    k = d.argmin(axis=1)
    dist = d[np.arange(len(c)), k]
    sel = dist < 0.07
    if sel.sum() < 20:
        return None, 0.0
    w = (np.clip(lv[sel], 0.02, 0.9) ** 0.5) * (1.0 - dist[sel] / 0.07)
    # robust: weighted median position along the locus (in mired) ...
    mired = 1e6 / temps[k[sel]]
    order = np.argsort(mired)
    cw = np.cumsum(w[order])
    med = mired[order][np.searchsorted(cw, cw[-1] / 2)]
    # ... then average the chroma of the candidates near that median
    near = np.abs(mired - med) < 25
    if near.sum() < 10:
        near = np.ones_like(mired, dtype=bool)
    chroma = np.average(pix_chroma[sel][near], axis=0, weights=w[near])
    # ~1.5 % of the image being plausible grey surfaces already gives full confidence
    confidence = float(np.clip(w[near].sum() / (len(lv) * 0.015), 0.0, 1.0))
    return chroma, confidence


def estimate_illuminant(linear: np.ndarray) -> np.ndarray:
    """Return the estimated scene illuminant as linear RGB (G normalised to 1)."""
    img = _downsample(linear, 256)
    px = img.reshape(-1, 3).astype(np.float64)
    lum = px @ LUM_WEIGHTS
    valid = (px.max(axis=1) < 0.97) & (lum > 0.01)
    if valid.sum() < 50:
        valid = lum > 0
    if valid.sum() < 10:
        return np.ones(3)
    v = px[valid]
    lv = lum[valid]

    stat = _statistical_estimate(img, v, lv)
    grey, conf = _grey_pixel_estimate(_chroma(v), lv)
    if grey is None:
        return np.exp(stat)
    wg = 0.4 + 0.55 * conf  # 0.4 .. 0.95 weight on the grey-pixel estimate
    return np.exp(wg * grey + (1 - wg) * stat)


def auto_white_balance(linear: np.ndarray) -> tuple[float, float]:
    illum = estimate_illuminant(linear)
    temp, tint = solve_white_balance(illum)
    # keep a little of the original mood for very warm / very cool light (sunsets,
    # candle light, blue hour) - photographers rarely want those fully neutralised.
    mired = 1e6 / temp
    mired_ref = 1e6 / NEUTRAL_TEMP
    shift = mired - mired_ref
    keep = 0.12 if abs(shift) > 60 else 0.0
    temp = 1e6 / (mired_ref + shift * (1 - keep))
    # real light sources rarely deviate far from the Planckian locus -> limit tint
    return float(temp), float(np.clip(tint * 0.9, -80.0, 80.0))


def _blend_points(points, strength: float):
    return [[x, x + (y - x) * strength] if i not in (0, len(points) - 1)
            else [x * strength if x < 0.5 else 1 - (1 - x) * strength,
                  y * strength if y < 0.5 else 1 - (1 - y) * strength]
            for i, (x, y) in enumerate(points)]


def auto_curves(linear: np.ndarray, params: Params) -> dict:
    """Derive master + per-channel curves for an image rendered with ``params``."""
    base = params.copy()
    base.curves = {c: identity_curve() for c in base.curves}
    img = Renderer(base).apply(_downsample(linear, 512).copy())
    px = img.reshape(-1, 3)
    lum = px @ LUM_WEIGHTS

    bp = float(np.clip(np.quantile(lum, 0.003), 0.0, 0.06))
    wp = float(np.clip(np.quantile(lum, 0.997), 0.88, 1.0))
    if wp - bp < 0.3:
        bp, wp = 0.0, 1.0
    ln = np.clip((lum - bp) / (wp - bp), 0, 1)
    med = float(np.clip(np.median(ln), 0.02, 0.98))
    target = 0.46
    gamma = float(np.clip(math.log(target) / math.log(med), 0.65, 1.5))
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
    corrected = eval_curve(master, px)
    lo = np.quantile(corrected, 0.01, axis=0)
    hi = np.quantile(corrected, 0.99, axis=0)
    lo_m, hi_m = lo.mean(), hi.mean()
    curves = {"rgb": master}
    for i, c in enumerate("rgb"):
        dlo = float(np.clip((lo[i] - lo_m) * 0.6, -0.04, 0.04))
        dhi = float(np.clip((hi[i] - hi_m) * 0.6, -0.04, 0.04))
        start = [dlo, 0.0] if dlo > 0 else [0.0, -dlo]
        end = [1.0 - dhi, 1.0] if dhi > 0 else [1.0, 1.0 + dhi]
        curves[c] = [[round(start[0], 4), round(start[1], 4)], [round(end[0], 4), round(end[1], 4)]]
        if abs(dlo) < 0.003 and abs(dhi) < 0.003:
            curves[c] = identity_curve()
    curves["rgb"] = [[round(x, 4), round(y, 4)] for x, y in master]
    return curves


def ai_correct(linear: np.ndarray, params: Params, strength: float = 1.0) -> Params:
    """Return a copy of ``params`` with AI white balance and curves applied."""
    strength = float(np.clip(strength, 0.0, 1.0))
    out = params.copy()
    temp, tint = auto_white_balance(linear)
    mired_ref = 1e6 / NEUTRAL_TEMP
    out.temp = 1e6 / (mired_ref + (1e6 / temp - mired_ref) * strength)
    out.tint = tint * strength
    curves = auto_curves(linear, out)
    if strength < 1.0:
        curves = {c: _blend_points(p, strength) for c, p in curves.items()}
    out.curves = curves
    return out
