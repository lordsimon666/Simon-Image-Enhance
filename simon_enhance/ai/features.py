"""AI image operations used by the app.

* Background removal without green screen (BiRefNet). The model runs once per image
  at 1024x1024; the low-resolution mask is cached and refined at full resolution with
  the edge-aware guided filter, so preview and export use the same mask.
* "KI-Verbessern" (like Lightroom's Enhance): denoising (SCUNet) and/or upscaling
  (Real-ESRGAN) of the full-resolution original into a NEW 16-bit TIFF file.
"""

from __future__ import annotations

import hashlib
import os
import threading

import numpy as np
from PIL import Image

from ..processing import linear_to_srgb
from ..store import data_dir
from . import engine as E

MAX_OUTPUT_PIXELS = 400_000_000


# ---------------------------------------------------------------------------
# background mask
# ---------------------------------------------------------------------------

_mask_mem: dict = {}
_mask_lock = threading.Lock()


def _masks_dir() -> str:
    d = os.path.join(data_dir(), "masks")
    os.makedirs(d, exist_ok=True)
    return d


def mask_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|birefnet"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_mask(key: str) -> np.ndarray | None:
    """Cached mask as float32 HxW in [0, 1] (preview resolution)."""
    if not key:
        return None
    with _mask_lock:
        m = _mask_mem.get(key)
        if m is not None:
            return m
        f = os.path.join(_masks_dir(), key + ".png")
        if not os.path.exists(f):
            return None
        m = np.asarray(Image.open(f), dtype=np.float32) / 255.0
        _mask_mem[key] = m
        return m


def compute_mask(path: str, linear_preview: np.ndarray) -> str:
    """Run BiRefNet on the (preview) image, cache the mask, return its key."""
    key = mask_key(path)
    if load_mask(key) is not None:
        return key
    srgb = linear_to_srgb(linear_preview)
    h, w = srgb.shape[:2]
    im = Image.fromarray((srgb * 255 + 0.5).astype(np.uint8)).resize((1024, 1024), Image.Resampling.BILINEAR)
    x = np.asarray(im, np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    logits = E.run("birefnet", np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
    y = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))  # the exported graph ends with logits
    m = Image.fromarray(y[0, 0].astype(np.float32), "F").resize((w, h), Image.Resampling.BILINEAR)
    m8 = (np.clip(np.asarray(m), 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(m8, "L").save(os.path.join(_masks_dir(), key + ".png"))
    with _mask_lock:
        _mask_mem[key] = m8.astype(np.float32) / 255.0
    return key


def sample_mask(mask: np.ndarray, y0: int, rows: int, full_h: int, full_w: int) -> np.ndarray:
    """Bilinear sample of the low-res mask for image rows [y0, y0+rows) at full width."""
    mh, mw = mask.shape
    ys = (np.arange(y0, y0 + rows) + 0.5) * mh / full_h - 0.5
    xs = (np.arange(full_w) + 0.5) * mw / full_w - 0.5
    ys = np.clip(ys, 0, mh - 1)
    xs = np.clip(xs, 0, mw - 1)
    y_lo = np.floor(ys).astype(int)
    y_hi = np.minimum(y_lo + 1, mh - 1)
    fy = (ys - y_lo)[:, None]
    band = mask[y_lo] * (1 - fy) + mask[y_hi] * fy          # rows x mw
    x_lo = np.floor(xs).astype(int)
    x_hi = np.minimum(x_lo + 1, mw - 1)
    fx = (xs - x_lo)[None, :]
    return (band[:, x_lo] * (1 - fx) + band[:, x_hi] * fx).astype(np.float32)


# ---------------------------------------------------------------------------
# enhance: denoise + upscale into a new file
# ---------------------------------------------------------------------------

def enhance_output_path(src: str, denoise: bool, scale: int) -> str:
    base, _ = os.path.splitext(src)
    tags = []
    if denoise:
        tags.append("entrauscht")
    if scale > 1:
        tags.append(f"x{scale}")
    out = f"{base}-{'-'.join(tags)}.tif"
    n = 2
    while os.path.exists(out):
        out = f"{base}-{'-'.join(tags)}-{n}.tif"
        n += 1
    return out


def enhance(src_path: str, dst_path: str, denoise_strength: float = 0.0, scale: int = 1,
            progress=None, cancel=None) -> str:
    """Denoise (strength 0..1) and/or upscale (1, 2, 4) the original at full resolution
    and write a 16-bit TIFF. Works in horizontal bands to bound memory."""
    from ..fileio import load_source
    import tifffile  # type: ignore

    if denoise_strength <= 0 and scale == 1:
        raise ValueError("Nichts zu tun")
    src = load_source(src_path)
    lin = src.linear
    h, w = lin.shape[:2]
    if h * w * scale * scale > MAX_OUTPUT_PIXELS:
        raise ValueError(f"Ergebnis wäre {w * scale}×{h * scale} Pixel – zu groß "
                         f"(maximal {MAX_OUTPUT_PIXELS // 1_000_000} Megapixel).")
    out = np.empty((h * scale, w * scale, 3), np.uint16)
    alpha = None
    if src.alpha is not None:
        a = src.alpha[:]
        alpha = np.asarray(Image.fromarray(a.astype(np.float32), "F").resize(
            (w * scale, h * scale), Image.Resampling.BICUBIC)) if scale > 1 else a

    steps = (1 if denoise_strength > 0 else 0) + (1 if scale > 1 else 0)
    band, margin = 768, 48
    bands = list(range(0, h, band))
    done = [0]

    def sub_progress(frac):
        if progress is not None:
            progress(min(0.999, (done[0] + frac) / (len(bands) * steps)))

    for y in bands:
        y0, y1 = max(0, y - margin), min(h, y + band + margin)
        img = linear_to_srgb(np.asarray(lin[y0:y1], np.float32))
        if denoise_strength > 0:
            dn = E.run_tiled("scunet", img, progress=sub_progress, cancel=cancel)
            img = img + (dn - img) * float(np.clip(denoise_strength, 0, 1))
            done[0] += 1
        if scale > 1:
            up = E.run_tiled("realesr", img, progress=sub_progress, cancel=cancel)
            if scale == 2:  # the model is 4x; 2x = 4x result downscaled (sharper than 2x resize)
                up = np.stack([np.asarray(Image.fromarray(up[..., c], "F").resize(
                    (up.shape[1] // 2, up.shape[0] // 2), Image.Resampling.LANCZOS)) for c in range(3)], -1)
            img = up
            done[0] += 1
        a, b = (y - y0) * scale, (y - y0 + min(band, h - y)) * scale
        out[y * scale:y * scale + (b - a)] = (np.clip(img[a:b], 0, 1) * 65535 + 0.5).astype(np.uint16)

    data = out if alpha is None else np.concatenate(
        [out, (np.clip(alpha, 0, 1) * 65535 + 0.5).astype(np.uint16)[..., None]], axis=-1)
    tifffile.imwrite(dst_path, data, photometric="rgb", compression="zlib",
                     extrasamples=["unassalpha"] if alpha is not None else None)
    if progress is not None:
        progress(1.0)
    return dst_path


def used_device(model_id: str) -> str:
    return E.describe(E.last_device(model_id))
