"""Loading (JPEG / PNG / RAW) and exporting (JPEG / PNG)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageCms, ImageOps

from .processing import Params, Renderer, srgb_to_linear, to_uint16, to_uint8

RASTER_EXTS = {".jpg", ".jpeg", ".jpe", ".png", ".tif", ".tiff", ".webp"}
RAW_EXTS = {
    ".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
    ".orf", ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".fff", ".iiq", ".mos", ".mrw",
    ".kdc", ".dcr", ".erf", ".mef", ".raw", ".rwl",
}
SUPPORTED_EXTS = RASTER_EXTS | RAW_EXTS

PREVIEW_MAX = 2560

# Pillow refuses images above ~179 MP as a "decompression bomb" guard. Current phone
# cameras shoot 200 MP, and this app only opens files the user picked, so allow up to
# 1 gigapixel (still refusing absurd, maliciously crafted dimensions).
Image.MAX_IMAGE_PIXELS = 1_000_000_000

try:  # optional at runtime, but bundled in all builds
    import rawpy  # type: ignore
except Exception:  # pragma: no cover
    rawpy = None


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTS


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


def raw_available() -> bool:
    return rawpy is not None


@dataclass
class ImageData:
    path: str
    linear: np.ndarray          # HxWx3 float32 linear sRGB
    alpha: np.ndarray | None    # HxW float32 0..1
    exif: bytes | None
    is_raw: bool
    full_size: tuple[int, int]  # (w, h) of the original

    @property
    def size(self) -> tuple[int, int]:
        return self.linear.shape[1], self.linear.shape[0]


class LazyLinear:
    """Full-resolution source kept in its compact encoded form (8/16 bit). Row slices
    are converted to linear float32 on demand - a 200 MP photo needs ~0.6 GB instead of
    2.4 GB per float copy."""

    def __init__(self, data: np.ndarray, kind: str):
        self.data = data          # HxWx3 (colour) or HxW (alpha)
        self.kind = kind          # "srgb8", "srgb16", "linear16", "alpha8", "alpha16", "float"
        self.shape = data.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        part = self.data[key]
        if self.kind == "srgb8":
            return _LUT8[part]
        if self.kind == "srgb16":
            return _lut16()[part]
        if self.kind in ("linear16", "alpha16"):
            return part.astype(np.float32) / 65535.0
        if self.kind == "alpha8":
            return part.astype(np.float32) / 255.0
        return np.asarray(part, dtype=np.float32)


_LUT8 = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
_LUT16 = None


def _lut16() -> np.ndarray:
    global _LUT16
    if _LUT16 is None:
        _LUT16 = srgb_to_linear(np.arange(65536, dtype=np.float32) / 65535.0)
    return _LUT16


def _resize_float(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return arr
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    chans = []
    for c in range(arr.shape[2]):
        im = Image.fromarray(np.ascontiguousarray(arr[..., c], dtype=np.float32), mode="F")
        chans.append(np.asarray(im.resize((nw, nh), Image.Resampling.LANCZOS, reducing_gap=3.0)))
    return np.stack(chans, axis=-1).astype(np.float32)


def _to_srgb(im: Image.Image) -> Image.Image:
    """Convert embedded ICC profiles (e.g. AdobeRGB) to sRGB for 8-bit images."""
    icc = im.info.get("icc_profile")
    if not icc or im.mode not in ("RGB", "RGBA"):
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = ImageCms.getProfileDescription(src).lower()
        if "srgb" in desc:
            return im
        dst = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(im, src, dst, outputMode=im.mode)
    except Exception:
        return im


def _orientation(im: Image.Image) -> int:
    try:
        return int(im.getexif().get(0x0112, 1))
    except Exception:
        return 1


def _load_raster(path: str, max_side: int | None, lazy: bool = False):
    im = Image.open(path)
    exif = im.info.get("exif")
    w0, h0 = im.size
    full = (h0, w0) if _orientation(im) in (5, 6, 7, 8) else (w0, h0)
    if max_side and im.format == "JPEG":
        # decode directly at 1/2, 1/4 or 1/8 size: fast and memory friendly for huge photos
        im.draft("RGB", (max_side, max_side))
    im = ImageOps.exif_transpose(im)
    alpha8 = None

    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.uint32)
        maxv = 65535 if arr.max() > 255 or im.mode.startswith("I;16") else 255
        g = np.clip(arr * (65535.0 / maxv), 0, 65535).astype(np.uint16)
        enc, kind = np.repeat(g[..., None], 3, axis=-1), "srgb16"
    else:
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            alpha8 = np.asarray(im.getchannel("A"), dtype=np.uint8)
        if im.mode == "CMYK":
            im = im.convert("RGB")
        im = _to_srgb(im)
        rgb = im.convert("RGB")
        if max_side and max(rgb.size) > max_side:
            # previews: shrink while still 8 bit (never builds a full-size float copy)
            rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
            if alpha8 is not None:
                alpha8 = np.asarray(Image.fromarray(alpha8, "L").resize(rgb.size, Image.Resampling.LANCZOS))
        enc, kind = np.asarray(rgb, dtype=np.uint8), "srgb8"

    if lazy:
        alpha = None if alpha8 is None else LazyLinear(alpha8, "alpha8")
        return LazyLinear(enc, kind), alpha, exif, full
    lin = LazyLinear(enc, kind)[:]
    alpha = None if alpha8 is None else alpha8.astype(np.float32) / 255.0
    if max_side:
        lin = _resize_float(lin, max_side)
        if alpha is not None:
            alpha = _resize_float(alpha[..., None], max_side)[..., 0]
    return lin.astype(np.float32), alpha, exif, full


def _png_is_16bit_color(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        return head[12:16] == b"IHDR" and head[24] == 16 and head[25] in (2, 6)
    except Exception:
        return False


def _load_png16(path: str, max_side: int | None, lazy: bool = False):
    """Pillow cannot read 16-bit RGB PNGs at full depth; use imageio if present."""
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        return None
    try:
        arr = iio.imread(path)
    except Exception:
        return None
    if arr.dtype != np.uint16 or arr.ndim != 3:
        return None
    full = (arr.shape[1], arr.shape[0])
    if lazy:
        alpha = LazyLinear(arr[..., 3], "alpha16") if arr.shape[2] == 4 else None
        return LazyLinear(arr[..., :3], "srgb16"), alpha, None, full
    alpha = arr[..., 3].astype(np.float32) / 65535.0 if arr.shape[2] == 4 else None
    lin = _lut16()[arr[..., :3]]
    if max_side:
        lin = _resize_float(lin, max_side)
        if alpha is not None:
            alpha = _resize_float(alpha[..., None], max_side)[..., 0]
    return lin.astype(np.float32), alpha, None, full


def _load_raw(path: str, max_side: int | None, lazy: bool = False):
    if rawpy is None:
        raise RuntimeError("RAW-Unterstützung nicht verfügbar (rawpy fehlt).")
    with rawpy.imread(path) as raw:
        full = (raw.sizes.width, raw.sizes.height)
        if raw.sizes.flip in (5, 6):
            full = (full[1], full[0])
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=False,
            output_color=rawpy.ColorSpace.sRGB,
            half_size=bool(max_side),
            user_flip=None,
        )
    if lazy:
        return LazyLinear(rgb, "linear16"), None, None, full
    lin = rgb.astype(np.float32) / 65535.0
    if max_side:
        lin = _resize_float(lin, max_side)
    return lin, None, None, full


def load_image(path: str, preview: bool = True) -> ImageData:
    max_side = PREVIEW_MAX if preview else None
    if is_raw(path):
        lin, alpha, exif, full = _load_raw(path, max_side)
        raw_flag = True
    else:
        res = None
        if path.lower().endswith(".png") and _png_is_16bit_color(path):
            res = _load_png16(path, max_side)
        if res is None:
            res = _load_raster(path, max_side)
        lin, alpha, exif, full = res
        raw_flag = False
    return ImageData(path=path, linear=lin, alpha=alpha, exif=exif, is_raw=raw_flag, full_size=full)


def load_source(path: str) -> ImageData:
    """Full resolution for export: ``linear`` / ``alpha`` are LazyLinear sources."""
    if is_raw(path):
        lin, alpha, exif, full = _load_raw(path, None, lazy=True)
        raw_flag = True
    else:
        res = None
        if path.lower().endswith(".png") and _png_is_16bit_color(path):
            res = _load_png16(path, None, lazy=True)
        if res is None:
            res = _load_raster(path, None, lazy=True)
        lin, alpha, exif, full = res
        raw_flag = False
    return ImageData(path=path, linear=lin, alpha=alpha, exif=exif, is_raw=raw_flag, full_size=full)


def _resize_source(src: LazyLinear, long_edge: int) -> LazyLinear:
    """Downscale an encoded source (still 8/16 bit) for exports with a size limit."""
    h, w = src.shape[:2]
    scale = long_edge / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    d = src.data
    if d.dtype == np.uint8:
        mode = "RGB" if d.ndim == 3 else "L"
        out = np.asarray(Image.fromarray(d, mode).resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0))
        return LazyLinear(out, src.kind)
    chans = [d] if d.ndim == 2 else [d[..., c] for c in range(d.shape[2])]
    res = []
    for ch in chans:
        im = Image.fromarray(ch.astype(np.int32), "I").resize(size, Image.Resampling.LANCZOS)
        res.append(np.clip(np.asarray(im), 0, 65535).astype(np.uint16))
    out = res[0] if d.ndim == 2 else np.stack(res, axis=-1)
    return LazyLinear(out, src.kind)


def load_thumbnail(path: str, size: int = 200) -> np.ndarray:
    """Fast 8-bit sRGB thumbnail (HxWx3 uint8)."""
    if is_raw(path) and rawpy is not None:
        try:
            with rawpy.imread(path) as raw:
                th = raw.extract_thumb()
            if th.format == rawpy.ThumbFormat.JPEG:
                im = Image.open(io.BytesIO(th.data))
                im = ImageOps.exif_transpose(im)
            else:
                im = Image.fromarray(th.data)
            im = im.convert("RGB")
            im.thumbnail((size, size))
            return np.asarray(im)
        except Exception:
            data = load_image(path, preview=True)
            return to_uint8(Renderer(Params()).apply(_resize_float(data.linear, size)))
    with Image.open(path) as im:
        im.draft("RGB", (size * 2, size * 2))
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((size, size))
        return np.asarray(im)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@dataclass
class ExportOptions:
    fmt: str = "jpeg"          # "jpeg" | "png"
    quality: int = 92
    png_bits: int = 8          # 8 | 16
    long_edge: int | None = None
    keep_exif: bool = True
    jpeg_background: tuple = (1.0, 1.0, 1.0)  # used for transparent areas in JPEGs


def export_image(src_path: str, params: Params, dst_path: str, opts: ExportOptions,
                 progress=None) -> str:
    """Render ``src_path`` at full resolution and save it. ``progress(fraction)`` is
    called while rendering (large photos with background removal take a while)."""
    data = load_source(src_path)
    lin, alpha = data.linear, data.alpha
    if opts.long_edge and max(lin.shape[:2]) > opts.long_edge:
        lin = _resize_source(lin, opts.long_edge)
        if alpha is not None:
            alpha = _resize_source(alpha, opts.long_edge)

    renderer = Renderer(params)
    h, w = lin.shape[:2]
    from .greenscreen import has_key
    has_alpha = alpha is not None or has_key(params)
    sixteen = opts.fmt == "png" and opts.png_bits == 16
    keep_alpha = has_alpha and opts.fmt == "png"
    out = np.empty((h, w, 4 if keep_alpha else 3), np.uint16 if sixteen else np.uint8)
    quant = to_uint16 if sixteen else to_uint8
    bg = np.array(opts.jpeg_background, np.float32)
    # render strip by strip straight into the output format (low memory); keyed
    # images use taller strips because the matte filters need overlapping borders
    rows = 768 if has_key(params) else 256
    for y, rgb, a in renderer.iter_strips(lin, alpha, rows=rows):
        n = len(rgb)
        if a is not None and not keep_alpha:  # JPEG: no transparency -> background colour
            rgb = rgb * a[..., None] + bg * (1.0 - a[..., None])
        out[y:y + n, :, :3] = quant(rgb)
        if keep_alpha:
            out[y:y + n, :, 3] = quant(a) if a is not None else (65535 if sixteen else 255)
        if progress is not None:
            progress(0.9 * (y + n) / h)
    del lin, alpha
    data.linear = data.alpha = None  # free the source before encoding the output

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    srgb_icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    if opts.fmt == "png":
        if sixteen:
            _write_png16(dst_path, out)
        else:
            im = Image.fromarray(out, "RGBA" if keep_alpha else "RGB")
            im.save(dst_path, "PNG", optimize=False, compress_level=6, icc_profile=srgb_icc)
    else:
        im = Image.fromarray(out, "RGB")
        kw = dict(quality=int(opts.quality), subsampling=0 if opts.quality >= 90 else 2,
                  optimize=True, icc_profile=srgb_icc)
        if opts.keep_exif and data.exif:
            try:
                ex = Image.Exif()
                ex.load(data.exif)
                ex[0x0112] = 1  # orientation already applied
                kw["exif"] = ex.tobytes()
            except Exception:
                pass
        im.save(dst_path, "JPEG", **kw)
    if progress is not None:
        progress(1.0)
    return dst_path


def _write_png16(path: str, arr: np.ndarray) -> None:
    try:
        import imageio.v3 as iio  # type: ignore
        iio.imwrite(path, arr, extension=".png")
        return
    except Exception:
        pass
    # fallback: minimal 16-bit PNG writer
    import struct
    import zlib
    h, w, c = arr.shape
    color_type = 6 if c == 4 else 2
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
