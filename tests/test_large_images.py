"""Large photos (200 MP phone cameras) and the low-memory export path."""
import os

import numpy as np
import pytest
from PIL import Image

from simon_enhance.fileio import (
    ExportOptions, LazyLinear, export_image, load_image, load_source,
)
from simon_enhance.greenscreen import ai_remove_background
from simon_enhance.processing import Params, Renderer, to_uint8


def _photo(w, h, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.stack([0.3 + 0.5 * x, 0.6 - 0.3 * y, 0.2 + 0.4 * x * y], -1)
    img += rng.normal(0, 0.02, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def test_pixel_limit_allows_200_megapixel_photos():
    # Pillow's default (~179 MP) rejected 200 MP phone photos as "decompression bomb"
    assert Image.MAX_IMAGE_PIXELS >= 250_000_000


def test_jpeg_preview_is_decoded_small_but_reports_full_size(tmp_path):
    src = tmp_path / "big.jpg"
    Image.fromarray(_photo(6000, 4000)).save(src, quality=90)
    d = load_image(str(src))
    assert max(d.linear.shape[:2]) <= 2560
    assert d.full_size == (6000, 4000)


def test_lazy_source_matches_eager_decode(tmp_path):
    src = tmp_path / "a.png"
    Image.fromarray(_photo(300, 200)).save(src)
    lazy = load_source(str(src)).linear
    eager = load_image(str(src), preview=False).linear
    assert isinstance(lazy, LazyLinear) and lazy.shape == eager.shape
    assert np.array_equal(lazy[:], eager)
    assert np.array_equal(lazy[50:90], eager[50:90])


@pytest.mark.parametrize("keyed", [False, True])
def test_streamed_export_matches_in_memory_render(tmp_path, keyed):
    img = _photo(400, 900)
    if keyed:  # add a green screen frame around a grey subject
        img[:, :, :] = (75, 160, 95)
        img[300:800, 120:280] = (60, 60, 62)
    src = tmp_path / "s.png"
    Image.fromarray(img).save(src)
    d = load_image(str(src), preview=False)
    p = Params(contrast=20, exposure=0.3)
    if keyed:
        p, _ = ai_remove_background(d.linear, p)
    out = export_image(str(src), p, str(tmp_path / "o.png"), ExportOptions(fmt="png"))
    got = np.asarray(Image.open(out)).astype(int)
    rgb, a = Renderer(p).apply_tiled_rgba(d.linear, d.alpha, rows=10_000)  # one piece
    exp = to_uint8(rgb).astype(int)
    assert np.abs(got[..., :3] - exp).max() <= 1
    if keyed:
        assert np.abs(got[..., 3] - to_uint8(a).astype(int)).max() <= 1


def test_export_progress_is_reported(tmp_path):
    src = tmp_path / "p.jpg"
    Image.fromarray(_photo(500, 1200)).save(src)
    seen = []
    export_image(str(src), Params(), str(tmp_path / "o.jpg"), ExportOptions(), progress=seen.append)
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)


def test_resize_of_16bit_source_keeps_depth():
    """16-bit sources (RAW files) are downscaled without dropping to 8 bit."""
    from simon_enhance.fileio import _resize_source
    arr = (np.random.default_rng(1).random((300, 500, 3)) * 65535).astype(np.uint16)
    src = LazyLinear(arr, "linear16")
    small = _resize_source(src, 250)
    assert small.shape == (150, 250, 3) and small.data.dtype == np.uint16
    assert abs(float(small[:].mean()) - float(src[:].mean())) < 0.01


@pytest.mark.skipif(not os.environ.get("SIE_200MP_SAMPLE"), reason="no 200 MP sample provided")
def test_real_200mp_photo(tmp_path):
    path = os.environ["SIE_200MP_SAMPLE"]
    d = load_image(path)
    assert d.full_size[0] * d.full_size[1] > 180_000_000
    out = export_image(path, Params(exposure=0.2), str(tmp_path / "o.jpg"), ExportOptions(long_edge=4000))
    assert max(Image.open(out).size) == 4000
