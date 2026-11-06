import numpy as np
import pytest
from PIL import Image

from simon_enhance.auto import ai_correct
from simon_enhance.fileio import ExportOptions, export_image, load_image
from simon_enhance.greenscreen import ai_remove_background, detect_key
from simon_enhance.processing import Params, Renderer, srgb_to_linear


def studio(screen=(0.30, 0.63, 0.38), size=(400, 300), seed=0):
    """sRGB studio shot: unevenly lit screen with folds, a person-like subject with
    dark shirt (incl. green reflections), skin and blue jeans."""
    h, w = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    light = 0.75 + 0.35 * x + 0.12 * np.sin(y * 25 + x * 7)  # hot spot + folds
    img = np.clip(np.array(screen)[None, None] * light[..., None], 0, 1)
    body = (y > 0.35) & (np.abs(x - 0.5) < 0.22)
    head = ((y - 0.22) / 0.12) ** 2 + ((x - 0.5) / 0.1) ** 2 < 1
    img[body] = (0.20, 0.20, 0.21)
    img[head] = (0.80, 0.56, 0.50)
    img[(y > 0.85) & body] = (0.38, 0.45, 0.52)
    # green reflections on the shirt
    spill = body & (rng.random((h, w)) < 0.01)
    img[spill] = (0.23, 0.31, 0.24)
    img += rng.normal(0, 0.006, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), body | head


def test_detect_green_and_blue_screen():
    g, _ = studio()
    info = detect_key(srgb_to_linear(g))
    assert info and info["channel"] == 1 and info["level"] > 0.2
    b, _ = studio(screen=(0.15, 0.30, 0.75))
    info = detect_key(srgb_to_linear(b))
    assert info and info["channel"] == 2


def test_no_screen_detected_on_normal_photo():
    rng = np.random.default_rng(3)
    img = rng.random((200, 300, 3)).astype(np.float32) * 0.5 + 0.25
    assert detect_key(srgb_to_linear(img)) is None
    p, info = ai_remove_background(srgb_to_linear(img), Params())
    assert p is None and info is None


def test_matte_separates_subject_and_backdrop():
    img, subject = studio()
    lin = srgb_to_linear(img)
    p, _ = ai_remove_background(lin, Params())
    rgb, alpha = Renderer(p).apply_rgba(lin)
    inner = subject.copy()
    # ignore a 3 px band around the outline (soft edge)
    from simon_enhance.greenscreen import _box
    band = _box(subject.astype(np.float32), 3)
    inner = band > 0.999
    outer = band < 0.001
    assert (alpha[outer] == 0).mean() > 0.995
    assert (alpha[inner] == 1).mean() > 0.995  # incl. the green reflections on the shirt
    # despill: no green excess left on the subject
    g_excess = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    assert np.percentile(g_excess[inner], 99.5) < 0.02


def test_tiled_export_matches_preview():
    img, _ = studio(size=(700, 300))
    lin = srgb_to_linear(img)
    p, _ = ai_remove_background(lin, Params())
    r = Renderer(p)
    rgb1, a1 = r.apply_rgba(lin)
    rgb2, a2 = r.apply_tiled_rgba(lin, rows=128)
    assert np.abs(a1 - a2).max() < 1e-4
    assert np.abs(rgb1 - rgb2).max() < 1e-4


def test_png_export_has_transparency_jpeg_gets_background(tmp_path):
    img, _ = studio()
    src = tmp_path / "studio.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(src)
    data = load_image(str(src))
    p, _ = ai_remove_background(data.linear, Params())
    out = export_image(str(src), p, str(tmp_path / "cut.png"), ExportOptions(fmt="png"))
    im = Image.open(out)
    assert im.mode == "RGBA"
    a = np.asarray(im)[..., 3]
    assert a[5, 5] == 0 and a[300, 150] == 255
    out16 = export_image(str(src), p, str(tmp_path / "cut16.png"), ExportOptions(fmt="png", png_bits=16))
    assert load_image(out16).alpha is not None
    jpg = export_image(str(src), p, str(tmp_path / "cut.jpg"),
                       ExportOptions(fmt="jpeg", jpeg_background=(1.0, 1.0, 1.0)))
    assert np.asarray(Image.open(jpg))[5, 5].min() > 245


def test_key_settings_roundtrip_and_toggle():
    img, _ = studio()
    lin = srgb_to_linear(img)
    p, _ = ai_remove_background(lin, Params())
    assert Params.from_dict(p.to_dict()) == p
    p.key_enabled = 0.0
    _, alpha = Renderer(p).apply_rgba(lin)
    assert alpha is None


@pytest.mark.parametrize("tol", [0, 50, 100])
def test_tolerance_is_monotone(tol):
    img, _ = studio()
    lin = srgb_to_linear(img)
    p, _ = ai_remove_background(lin, Params())
    p.key_tolerance = tol
    _, a = Renderer(p).apply_rgba(lin)
    p.key_tolerance = min(100, tol + 30)
    _, b = Renderer(p).apply_rgba(lin)
    assert b.sum() <= a.sum() + 1e-3


def test_ai_correction_ignores_removed_backdrop():
    img, _ = studio()
    lin = srgb_to_linear(img)
    pk, _ = ai_remove_background(lin, Params())
    p = ai_correct(lin, pk)
    # the green screen must not trick the AI into a strong magenta correction
    assert abs(p.tint) < 15
    assert p.key_enabled == 1.0
