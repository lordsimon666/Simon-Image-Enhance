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


def _load_raster(path: str, max_side: int | None):
    im = Image.open(path)
    exif = im.info.get("exif")
    im = ImageOps.exif_transpose(im)
    full = im.size
    alpha = None

    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.uint32)
        maxv = 65535 if arr.max() > 255 or im.mode.startswith("I;16") else 255
        g = (arr.astype(np.float32) / maxv).clip(0, 1)
        arr = np.repeat(srgb_to_linear(g)[..., None], 3, axis=-1)
    else:
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            alpha = np.asarray(im.getchannel("A"), dtype=np.float32) / 255.0
        if im.mode == "CMYK":
            im = im.convert("RGB")
        im = _to_srgb(im)
        rgb = im.convert("RGB")
        arr8 = np.asarray(rgb, dtype=np.uint8)
        arr = _LUT8[arr8]

    if max_side:
        arr = _resize_float(arr, max_side)
        if alpha is not None:
            alpha = _resize_float(alpha[..., None], max_side)[..., 0]
    return arr.astype(np.float32), alpha, exif, full


def _png_is_16bit_color(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        return head[12:16] == b"IHDR" and head[24] == 16 and head[25] in (2, 6)
    except Exception:
        return False


def _load_png16(path: str, max_side: int | None):
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
    alpha = arr[..., 3].astype(np.float32) / 65535.0 if arr.shape[2] == 4 else None
    lin = _lut16()[arr[..., :3]]
    full = (arr.shape[1], arr.shape[0])
    if max_side:
        lin = _resize_float(lin, max_side)
        if alpha is not None:
            alpha = _resize_float(alpha[..., None], max_side)[..., 0]
    return lin.astype(np.float32), alpha, None, full


def _load_raw(path: str, max_side: int | None):
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


def export_image(src_path: str, params: Params, dst_path: str, opts: ExportOptions) -> str:
    data = load_image(src_path, preview=False)
    lin = data.linear
    alpha = data.alpha
    if opts.long_edge and max(lin.shape[:2]) > opts.long_edge:
        lin = _resize_float(lin, opts.long_edge)
        if alpha is not None:
            alpha = _resize_float(alpha[..., None], opts.long_edge)[..., 0]
    out, alpha = Renderer(params).apply_tiled_rgba(lin, alpha)
    del lin

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)
    srgb_icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    if opts.fmt == "png":
        if opts.png_bits == 16:
            arr = to_uint16(out)
            if alpha is not None:
                arr = np.concatenate([arr, to_uint16(alpha)[..., None]], axis=-1)
            _write_png16(dst_path, arr)
        else:
            arr = to_uint8(out)
            im = Image.fromarray(arr, "RGB")
            if alpha is not None:
                im.putalpha(Image.fromarray(to_uint8(alpha), "L"))
            im.save(dst_path, "PNG", optimize=False, compress_level=6, icc_profile=srgb_icc)
    else:
        if alpha is not None:  # JPEG has no transparency -> place on the chosen background
            bg = np.array(opts.jpeg_background, np.float32)
            out = out * alpha[..., None] + bg * (1.0 - alpha[..., None])
        im = Image.fromarray(to_uint8(out), "RGB")
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
