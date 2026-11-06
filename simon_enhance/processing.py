"""Image processing pipeline (non-destructive, Lightroom-style).

Pipeline order:
    linear RGB  ->  white balance + exposure (linear multipliers)
                ->  sRGB encode  ->  basic tone (blacks/whites/contrast/highlights/shadows)
                ->  master curve ->  per-channel curves
                ->  vibrance / saturation

Everything after the linear multipliers is a pure per-channel 1D function, so it is
baked into one 65536-entry lookup table per channel. That makes slider updates fast.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict

import numpy as np

LUT_SIZE = 65536
LUM_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

NEUTRAL_TEMP = 6500.0
TEMP_MIN, TEMP_MAX = 2000.0, 15000.0
TINT_MIN, TINT_MAX = -150.0, 150.0

CHANNELS = ("rgb", "r", "g", "b")


def identity_curve() -> list[list[float]]:
    return [[0.0, 0.0], [1.0, 1.0]]


@dataclass
class Params:
    temp: float = NEUTRAL_TEMP   # Kelvin; 6500 = "as shot"
    tint: float = 0.0            # -150 (green) .. +150 (magenta)
    exposure: float = 0.0        # EV, -5 .. +5
    contrast: float = 0.0        # -100 .. 100
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    # background removal (green / blue screen) - see greenscreen.py
    key_enabled: float = 0.0     # 0 / 1
    key_channel: float = 1.0     # 1 = green, 2 = blue
    key_level: float = 0.0       # measured key excess of the backdrop (0 = not analysed)
    key_r: float = 0.0           # backdrop colour (sRGB 0..1)
    key_g: float = 0.0
    key_b: float = 0.0
    key_tolerance: float = 50.0  # 0..100
    key_softness: float = 50.0   # 0..100
    key_despill: float = 80.0    # 0..100
    key_edge: float = 25.0       # -100 (grow) .. 100 (shrink the matte)
    curves: dict = field(default_factory=lambda: {c: identity_curve() for c in CHANNELS})

    def copy(self) -> "Params":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        p = cls()
        for k, v in d.items():
            if k == "curves" and isinstance(v, dict):
                for c in CHANNELS:
                    if c in v and len(v[c]) >= 2:
                        p.curves[c] = [[float(a), float(b)] for a, b in v[c]]
            elif hasattr(p, k) and k != "curves":
                setattr(p, k, float(v))
        return p

    def is_default(self) -> bool:
        return self == Params()

    def key_rgb_list(self) -> list[float]:
        return [self.key_r, self.key_g, self.key_b]

    def without_key(self) -> "Params":
        p = self.copy()
        for k, v in Params().to_dict().items():
            if k.startswith("key_"):
                setattr(p, k, v)
        return p


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


_XYZ_TO_SRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def blackbody_rgb(temp: float) -> np.ndarray:
    """Linear sRGB colour of a Planckian radiator (Kim et al. approximation), Y=1."""
    t = float(np.clip(temp, 1667.0, 25000.0))
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    xyz = np.array([x / y, 1.0, (1 - x - y) / y])
    return _XYZ_TO_SRGB @ xyz


_REF_WHITE = blackbody_rgb(NEUTRAL_TEMP)


def _tint_factor(tint: float) -> float:
    return 2.0 ** (-tint / 300.0)


def wb_multipliers(temp: float, tint: float, normalize: bool = True) -> np.ndarray:
    """Per-channel linear gains. Lower temp -> cooler (bluer) image, like Lightroom."""
    m = _REF_WHITE / blackbody_rgb(temp)
    m = m.astype(np.float64)
    m[1] *= _tint_factor(tint)
    if normalize:
        m /= float(LUM_WEIGHTS @ m)
    return m.astype(np.float32)


def solve_white_balance(rgb_linear) -> tuple[float, float]:
    """Find (temp, tint) that make the given linear RGB colour neutral grey."""
    r, g, b = (max(float(v), 1e-6) for v in rgb_linear)
    target = math.log(b / r)  # need log(m_r / m_b) == log(b / r)
    mireds = np.linspace(1e6 / TEMP_MIN, 1e6 / TEMP_MAX, 600)
    temps = 1e6 / mireds
    vals = np.array([math.log(m[0] / m[2]) for m in (wb_multipliers(t, 0, False) for t in temps)])
    # vals increases with temperature
    order = np.argsort(vals)
    temp = float(np.interp(target, vals[order], temps[order]))
    m = wb_multipliers(temp, 0, False)
    gf = (m[0] * r) / (m[1] * g)
    tint = -300.0 * math.log2(max(gf, 1e-6))
    return float(np.clip(temp, TEMP_MIN, TEMP_MAX)), float(np.clip(tint, TINT_MIN, TINT_MAX))


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n)
    if n == 2:
        d[:] = delta[0]
        return d
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    def edge(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > abs(3 * d0):
            return 3 * d0
        return s

    d[0] = edge(h[0], h[1], delta[0], delta[1])
    d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    return d


def eval_curve(points, t: np.ndarray) -> np.ndarray:
    """Monotone cubic (PCHIP) interpolation through the control points."""
    pts = sorted((float(a), float(b)) for a, b in points)
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    # guard against duplicate x positions
    keep = np.concatenate([[True], np.diff(x) > 1e-6])
    x, y = x[keep], y[keep]
    t = np.asarray(t, dtype=np.float64)
    if len(x) < 2:
        return np.clip(t, 0, 1)
    d = _pchip_slopes(x, y)
    idx = np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    hh = x1 - x0
    s = np.clip((t - x0) / hh, 0.0, 1.0)
    s2, s3 = s * s, s * s * s
    out = ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * hh * d[idx]
           + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * hh * d[idx + 1])
    out = np.where(t <= x[0], y[0], out)
    out = np.where(t >= x[-1], y[-1], out)
    return np.clip(out, 0.0, 1.0)


def is_identity_curve(points) -> bool:
    return len(points) == 2 and points[0] == [0.0, 0.0] and points[1] == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Basic tone
# ---------------------------------------------------------------------------

def tone_curve(t: np.ndarray, p: Params) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    # blacks / whites move the end points
    bp = -p.blacks / 100.0 * 0.15
    wp = 1.0 - p.whites / 100.0 * 0.15
    t = np.clip((t - bp) / max(wp - bp, 1e-3), 0.0, 1.0)

    # shadows / highlights: smooth bumps that keep end points fixed
    norm = 4.0 / 27.0  # max of t^2 (1-t)
    if p.highlights:
        t = t + p.highlights / 100.0 * 0.14 * (t * t * (1 - t)) / norm
    if p.shadows:
        t = t + p.shadows / 100.0 * 0.14 * (t * (1 - t) * (1 - t)) / norm
    t = np.clip(t, 0.0, 1.0)

    # contrast: blend with smoothstep (S) or a flattened line
    c = p.contrast / 100.0
    if c > 0:
        t = t + c * ((3 * t * t - 2 * t * t * t) - t)
    elif c < 0:
        t = t + (-c) * ((0.5 + (t - 0.5) * 0.55) - t)
    t = np.maximum.accumulate(np.clip(t, 0.0, 1.0))
    return t


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

_KEY_CACHE = None  # (signature, (clean linear, matte)) of the last keyed preview

_ENC_GRID = linear_to_srgb(np.linspace(0, 1, LUT_SIZE, dtype=np.float32)).astype(np.float64)


class Renderer:
    """Pre-computes LUTs for a parameter set and applies them to linear images."""

    def __init__(self, params: Params):
        self.params = params.copy()
        p = self.params
        self.mult = wb_multipliers(p.temp, p.tint) * np.float32(2.0 ** p.exposure)
        base = tone_curve(_ENC_GRID, p)
        master = p.curves.get("rgb", identity_curve())
        if not is_identity_curve(master):
            base = eval_curve(master, base)
        luts = []
        for c in ("r", "g", "b"):
            pts = p.curves.get(c, identity_curve())
            luts.append(base if is_identity_curve(pts) else eval_curve(pts, base))
        self.luts = np.stack(luts).astype(np.float32)
        self.sat = 1.0 + p.saturation / 100.0
        self.vib = p.vibrance / 100.0

    def apply(self, linear: np.ndarray) -> np.ndarray:
        """linear: HxWx3 float32 in [0, 1+]. Returns sRGB float32 HxWx3 in [0, 1]."""
        x = linear * self.mult
        np.clip(x, 0.0, 1.0, out=x)
        idx = (x * (LUT_SIZE - 1) + 0.5).astype(np.uint16)
        out = np.empty_like(x)
        for c in range(3):
            out[..., c] = self.luts[c][idx[..., c]]
        if self.sat != 1.0 or self.vib != 0.0:
            lum = out @ LUM_WEIGHTS
            factor = np.float32(self.sat)
            if self.vib != 0.0:
                chroma = out.max(axis=-1) - out.min(axis=-1)
                factor = factor * (1.0 + self.vib * (1.0 - chroma) ** 2)
                factor = factor[..., None]
            out = lum[..., None] + (out - lum[..., None]) * factor
            np.clip(out, 0.0, 1.0, out=out)
        return out

    def apply_tiled(self, linear: np.ndarray, rows: int = 512) -> np.ndarray:
        out = np.empty(linear.shape, dtype=np.float32)
        for y in range(0, linear.shape[0], rows):
            out[y:y + rows] = self.apply(linear[y:y + rows].astype(np.float32, copy=True))
        return out

    # -- with background removal -------------------------------------------
    def _keyer(self, height: int):
        from .greenscreen import KeyMatte, has_key
        return KeyMatte(self.params, height) if has_key(self.params) else None

    def apply_rgba(self, linear: np.ndarray, alpha: np.ndarray | None = None):
        """Returns (sRGB, alpha or None). Alpha combines the source alpha and the key."""
        keyer = self._keyer(linear.shape[0])
        if keyer is None:
            return self.apply(linear.astype(np.float32, copy=True)), alpha
        # the key only depends on the source and the key settings -> cache it, so
        # moving any other slider stays instant
        sig = (id(linear), linear.shape, float(linear[::97, ::89].sum()),
               tuple(v for k, v in sorted(self.params.to_dict().items()) if k.startswith("key_")))
        global _KEY_CACHE
        if _KEY_CACHE is not None and _KEY_CACHE[0] == sig:
            clean, matte = _KEY_CACHE[1]
        else:
            clean, matte = keyer.apply(linear)
            _KEY_CACHE = (sig, (clean, matte))
        if alpha is not None:
            matte = matte * alpha
        return self.apply(clean), matte

    def iter_strips(self, linear, alpha=None, rows: int = 256):
        """Render in horizontal strips. ``linear`` / ``alpha`` may be arrays or lazy
        sources (anything with ``.shape`` and row slicing that returns float32 arrays),
        so huge photos never need a full float copy in memory.
        Yields (y, sRGB strip, alpha strip or None)."""
        h = linear.shape[0]
        keyer = self._keyer(h)
        pad = keyer.pad if keyer is not None else 0
        for y in range(0, h, rows):
            y0, y1 = max(0, y - pad), min(h, y + rows + pad)
            strip = np.asarray(linear[y0:y1], dtype=np.float32).copy()
            a, b = y - y0, y - y0 + min(rows, h - y)
            src_a = None if alpha is None else np.asarray(alpha[y:y + rows], dtype=np.float32)
            if keyer is None:
                yield y, self.apply(strip[a:b]), src_a
                continue
            clean, m = keyer.apply(strip)
            m = m[a:b]
            if src_a is not None:
                m = m * src_a
            yield y, self.apply(clean[a:b]), m

    def apply_tiled_rgba(self, linear, alpha=None, rows: int = 512):
        h, w = linear.shape[:2]
        out = np.empty((h, w, 3), dtype=np.float32)
        keyed = self._keyer(h) is not None
        matte = np.empty((h, w), dtype=np.float32) if (keyed or alpha is not None) else None
        for y, rgb, m in self.iter_strips(linear, alpha, rows):
            out[y:y + len(rgb)] = rgb
            if matte is not None:
                matte[y:y + len(rgb)] = m
        return out, matte


def render(linear: np.ndarray, params: Params) -> np.ndarray:
    return Renderer(params).apply(linear.astype(np.float32, copy=True))


def to_uint8(srgb: np.ndarray) -> np.ndarray:
    return (np.clip(srgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(srgb: np.ndarray) -> np.ndarray:
    return (np.clip(srgb, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
