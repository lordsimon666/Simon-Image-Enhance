"""KI-Hintergrundentfernung für Green- und Bluescreens.

Steps
1. ``detect_key`` analyses the image border, decides whether the backdrop is a green
   or blue screen, measures its colour and how strongly coloured it is (also in
   folds and shadows) and derives the keying thresholds from those statistics.
2. ``KeyMatte.apply`` computes a soft alpha matte from the *key excess*
   ``(K - max(other channels)) / K`` - brightness independent, so creases, shadows and
   uneven lighting of the backdrop do not matter.
3. The matte is refined with an edge-aware guided filter (hair, glasses, fabric
   edges follow the real image edges, JPEG block noise is smoothed out).
4. Edge colours are un-mixed from the backdrop colour and remaining green/blue
   spill on the subject is removed (despill).

Everything runs offline with NumPy. All settings are stored in ``Params`` so preview,
batch processing and export give identical results.
"""

from __future__ import annotations

import math

import numpy as np

from .processing import LUM_WEIGHTS, linear_to_srgb, srgb_to_linear

CHANNEL_NAMES = {1: "Grün", 2: "Blau"}


def _key_excess(srgb: np.ndarray, k: int, floor: float = 0.05) -> np.ndarray:
    """Relative excess of the key channel. ``floor`` keeps very dark pixels (where a
    tiny green reflection would look like a huge relative excess) from keying out."""
    others = [c for c in range(3) if c != k]
    K = srgb[..., k]
    o = np.maximum(srgb[..., others[0]], srgb[..., others[1]])
    return (K - o) / np.maximum(K, floor)


def _border(img: np.ndarray, frac: float = 0.06) -> np.ndarray:
    """Top, left and right border pixels (the bottom often shows the subject)."""
    h, w = img.shape[:2]
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    return np.concatenate([
        img[:bh].reshape(-1, 3),
        img[:, :bw].reshape(-1, 3),
        img[:, -bw:].reshape(-1, 3),
    ])


def detect_key(linear: np.ndarray) -> dict | None:
    """Detect a green/blue backdrop. Returns the key settings or None."""
    h, w = linear.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / 600)))
    srgb = linear_to_srgb(linear[::step, ::step, :3])
    border = _border(srgb)
    best = None
    for k in (1, 2):
        s = _key_excess(border, k)
        bg = s > 0.12
        frac = float(bg.mean())
        if frac >= 0.35 and (best is None or frac > best[1]):
            best = (k, frac, s[bg], border[bg])
    if best is None:
        return None
    k, frac, s_bg, px_bg = best
    # level of the weakest (brightest / most washed-out) parts of the backdrop
    level = float(np.percentile(s_bg, 3))
    key = np.median(px_bg, axis=0)
    # how much of the whole picture is backdrop - plausibility / UI info
    coverage = float((_key_excess(srgb, k) > level * 0.6).mean())
    return {
        "channel": k,
        "level": level,
        "key_rgb": [float(v) for v in key],
        "border_fraction": frac,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# guided filter (He et al.) - box filters via integral images
# ---------------------------------------------------------------------------

def _box_axis(x: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Windowed mean along one axis (window 2r+1, shortened and renormalised at the
    borders). Sums run in float64 - float32 loses precision on large photos."""
    n = x.shape[axis]
    c = np.cumsum(x, axis=axis, dtype=np.float64)
    shape = list(c.shape)
    shape[axis] = 1
    c = np.concatenate([np.zeros(shape), c], axis=axis)
    hi = np.minimum(np.arange(n) + r + 1, n)
    lo = np.maximum(np.arange(n) - r, 0)
    s = np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)
    cnt = (hi - lo).astype(np.float64)
    cnt = cnt[:, None] if axis == 0 else cnt[None, :]
    return s / cnt


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edges handled by normalising with the count.
    Separable: rows, then columns. Runs on the GPU (OpenCL) when available."""
    from . import gpu
    res = gpu.box(x, r)
    if res is not None:
        return res
    return _box_axis(_box_axis(x, r, 0), r, 1).astype(np.float32)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    mean_i = _box(guide, r)
    mean_p = _box(src, r)
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    var_i = _box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


# ---------------------------------------------------------------------------
# matte
# ---------------------------------------------------------------------------

class KeyMatte:
    """Applies a configured key to linear images (works on strips, see ``pad``)."""

    def __init__(self, params, full_height: int):
        self.k = int(params.key_channel) if params.key_channel in (1.0, 2.0) else 1
        level = max(float(params.key_level), 0.05)
        tol = params.key_tolerance / 100.0      # 0..1, default 0.5
        soft = params.key_softness / 100.0      # 0..1, default 0.5
        # s >= t_hi -> fully transparent, s <= t_lo -> fully opaque
        self.t_hi = level * (0.98 - 0.5 * tol)
        width = level * (0.05 + 0.4 * soft)
        self.t_lo = max(self.t_hi - width, 0.015)
        self.despill = params.key_despill / 100.0
        self.edge = params.key_edge / 100.0     # >0 shrink, <0 grow
        key_srgb = np.array(params.key_rgb_list(), np.float32)
        self.key_lin = srgb_to_linear(key_srgb)
        # darkest plausible backdrop (deep folds/shadows) ~ 45 % of its typical brightness
        self.floor = max(0.05, 0.45 * float(key_srgb[self.k]))
        # refinement radius scales with the resolution (preview vs. full export)
        # (capped: very large photos would otherwise need huge filter windows)
        self.radius = max(1, min(4, int(round(full_height / 900))))
        # neighbourhood used to measure the local backdrop colour next to edges
        self.bg_radius = self.radius * 6
        self.speck_radius = self.radius * 2 + 1
        self.cluster_radius = self.radius * 8
        self.pad = (self.radius * 2 + 1 + self.bg_radius + self.speck_radius
                    + self.cluster_radius + 3)

    def matte(self, srgb: np.ndarray) -> np.ndarray:
        s = _key_excess(srgb, self.k, self.floor)
        alpha = np.clip((self.t_hi - s) / max(self.t_hi - self.t_lo, 1e-4), 0.0, 1.0).astype(np.float32)
        guide = (srgb @ LUM_WEIGHTS).astype(np.float32)
        refined = guided_filter(guide, alpha, self.radius, 2e-3)
        # keep crisp interior / exterior, only let the filter shape the transition
        alpha = np.clip(refined, 0.0, 1.0)
        # despeckle: isolated holes inside the subject / isolated dots in the backdrop
        local = _box(alpha, self.speck_radius)
        alpha = np.where(local > 0.85, np.maximum(alpha, np.clip((local - 0.85) / 0.08, 0, 1)), alpha)
        alpha = np.where(local < 0.1, np.minimum(alpha, np.clip(local * 10.0, 0, 1)), alpha)
        # small clusters of "transparent" pixels well inside the subject are green
        # reflections on fabric / skin, not backdrop -> fill them
        wide = _box(alpha, self.cluster_radius)
        alpha = np.maximum(alpha, np.clip((wide - 0.8) / 0.12, 0.0, 1.0))
        # edge: pull the transition inwards / outwards (choke / spread)
        if self.edge > 0:
            c = 0.6 * self.edge
            alpha = np.clip((alpha - c) / (1.0 - c), 0.0, 1.0)
        elif self.edge < 0:
            c = 0.6 * -self.edge
            alpha = np.clip(alpha / (1.0 - c), 0.0, 1.0)
        alpha[alpha < 0.02] = 0.0
        alpha[alpha > 0.98] = 1.0
        return alpha

    def apply(self, linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cleaned linear RGB, alpha)."""
        srgb = linear_to_srgb(linear)
        alpha = self.matte(srgb)
        out = linear.astype(np.float32, copy=True)
        others = [c for c in range(3) if c != self.k]
        # 1) despill: remove green/blue reflections from the subject
        if self.despill > 0:
            limit = np.maximum(out[..., others[0]], out[..., others[1]])
            excess = np.maximum(out[..., self.k] - limit, 0.0)
            out[..., self.k] -= self.despill * excess
        # 2) colour decontamination of the edge. Semi-transparent pixels are a mix of
        #    subject and backdrop; mathematically "un-mixing" them amplifies every small
        #    error (dark / red fringes on thin edges). Instead they take the colour of
        #    the clean subject right behind the edge - the alpha alone makes the soft
        #    transition. Opaque pixels touching the edge are blended half-way, because
        #    they usually still carry some backdrop colour.
        transp = (alpha < 1.0).astype(np.float32)
        touching = _box(transp, 2) > 0
        if touching.any():
            clean_w = ((alpha >= 1.0) & ~touching).astype(np.float32)
            wsum = _box(clean_w, self.bg_radius)
            have = wsum > 1e-3
            semi = (alpha > 0.0) & (alpha < 1.0) & have
            ring = (alpha >= 1.0) & touching & have
            sel = semi | ring
            if sel.any():
                inner = np.empty((int(sel.sum()), 3), np.float32)
                for c in range(3):
                    inner[:, c] = (_box(out[..., c] * clean_w, self.bg_radius)[sel]
                                   / wsum[sel])
                own = out[sel]
                # how much of the pixel's own colour is kept (detail of hair strands)
                keep = np.where(ring[sel], 0.5, 0.25 * alpha[sel] ** 2)[:, None]
                out[sel] = inner * (1.0 - keep) + own * keep
        # 3) final safety: no key-colour excess may remain anywhere near the edge
        near = touching | (alpha < 1.0)
        limit = np.maximum(out[..., others[0]], out[..., others[1]])
        out[..., self.k] = np.where(near, np.minimum(out[..., self.k], limit), out[..., self.k])
        np.clip(out, 0.0, 1.0, out=out)
        return out, alpha


def has_key(params) -> bool:
    return bool(params.key_enabled) and params.key_level > 0


def checkerboard(h: int, w: int, size: int = 12) -> np.ndarray:
    """sRGB float checkerboard shown behind transparent areas."""
    y = (np.arange(h) // size)[:, None]
    x = (np.arange(w) // size)[None, :]
    board = np.where((x + y) % 2 == 0, 0.78, 0.6).astype(np.float32)
    return np.repeat(board[..., None], 3, axis=-1)


def composite(rgb: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return rgb * a + background * (1.0 - a)


def ai_remove_background(linear: np.ndarray, params):
    """Return (new params with the key configured, info dict) or (None, None)."""
    info = detect_key(linear)
    if info is None:
        return None, None
    p = params.copy()
    p.key_enabled = 1.0
    p.key_channel = float(info["channel"])
    p.key_level = info["level"]
    p.key_r, p.key_g, p.key_b = info["key_rgb"]
    return p, info
