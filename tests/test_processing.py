import os

import numpy as np
import pytest
from PIL import Image

from simon_enhance.auto import ai_correct, estimate_illuminant
from simon_enhance.fileio import ExportOptions, export_image, load_image, load_thumbnail
from simon_enhance.processing import (
    NEUTRAL_TEMP, Params, Renderer, eval_curve, linear_to_srgb, render, solve_white_balance,
    srgb_to_linear, wb_multipliers,
)
from simon_enhance.store import EditStore


def synthetic_scene(cast=(1.25, 1.0, 0.7), size=(240, 320)):
    """Linear image of grey ramps + coloured patches lit by a coloured light."""
    h, w = size
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.float32)
    ramp = np.linspace(0.03, 0.85, w, dtype=np.float32)
    img[: h // 2] = ramp[None, :, None]
    cols = [(0.6, 0.1, 0.1), (0.1, 0.5, 0.1), (0.1, 0.15, 0.6), (0.6, 0.55, 0.1), (0.3, 0.3, 0.3)]
    pw = w // len(cols)
    for i, c in enumerate(cols):
        img[h // 2:, i * pw:(i + 1) * pw] = c
    img += rng.normal(0, 0.005, img.shape).astype(np.float32)
    return np.clip(img * np.array(cast, np.float32), 0, 1)


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_identity_params_is_identity():
    lin = srgb_to_linear(np.random.default_rng(1).random((50, 60, 3)).astype(np.float32))
    out = render(lin, Params())
    assert np.abs(out - linear_to_srgb(lin)).max() < 2e-4


def test_neutral_wb_multipliers_are_one():
    assert np.allclose(wb_multipliers(NEUTRAL_TEMP, 0), 1.0, atol=1e-5)


def test_lower_temp_is_cooler():
    m = wb_multipliers(3000, 0)
    assert m[2] > m[0]
    m = wb_multipliers(10000, 0)
    assert m[0] > m[2]


@pytest.mark.parametrize("cast", [(1.3, 1.0, 0.6), (0.8, 1.0, 1.3), (1.0, 0.8, 1.0), (1.0, 1.15, 0.9)])
def test_solve_white_balance_neutralises(cast):
    temp, tint = solve_white_balance(np.array(cast))
    m = wb_multipliers(temp, tint)
    out = np.array(cast) * m
    assert out.max() / out.min() < 1.02


def test_curve_interpolation():
    pts = [[0, 0], [0.25, 0.15], [0.75, 0.85], [1, 1]]
    t = np.linspace(0, 1, 501)
    y = eval_curve(pts, t)
    assert np.all(np.diff(y) >= -1e-9)  # monotone
    assert abs(eval_curve(pts, np.array([0.25]))[0] - 0.15) < 1e-6
    # endpoints moved -> flat outside
    y = eval_curve([[0.1, 0.0], [0.9, 1.0]], np.array([0.0, 0.05, 0.95, 1.0]))
    assert np.allclose(y, [0, 0, 1, 1])


def test_all_sliders_stay_in_range():
    lin = synthetic_scene()
    p = Params(temp=3000, tint=40, exposure=1.5, contrast=80, highlights=-100, shadows=100,
               whites=50, blacks=-50, vibrance=60, saturation=40)
    out = render(lin, p)
    assert out.min() >= 0 and out.max() <= 1 and np.isfinite(out).all()
    p = Params(contrast=-100, highlights=100, shadows=-100, whites=-100, blacks=100,
               vibrance=-100, saturation=-100)
    out = render(lin, p)
    assert out.min() >= 0 and out.max() <= 1
    assert np.abs(out - out.mean(axis=-1, keepdims=True)).max() < 1e-4  # fully desaturated


def _cast_strength(srgb):
    grey = srgb[: srgb.shape[0] // 2, 20:-20]
    m = grey.reshape(-1, 3).mean(axis=0)
    return m.max() / m.min()


@pytest.mark.parametrize("cast,limit", [
    ((1.3, 1.0, 0.65), 1.10),   # warm light (strong casts keep a little mood on purpose)
    ((0.75, 1.0, 1.35), 1.10),  # cool light
    # pure magenta cast far off the Planckian locus: no physically plausible light
    # source explains it, so the AI deliberately corrects tint only cautiously
    ((1.1, 0.85, 1.05), 1.12),
])
def test_ai_removes_colour_cast(cast, limit):
    lin = synthetic_scene(cast)
    before = _cast_strength(render(lin, Params()))
    p = ai_correct(lin, Params())
    after = _cast_strength(render(lin, p))
    assert after < before
    assert after < limit, (before, after, p.temp, p.tint)
    # only WB + curves changed
    for k in ("exposure", "contrast", "highlights", "shadows", "whites", "blacks", "vibrance", "saturation"):
        assert getattr(p, k) == 0


def test_ai_on_neutral_image_is_gentle():
    lin = synthetic_scene((1, 1, 1))
    p = ai_correct(lin, Params())
    assert abs(p.temp - NEUTRAL_TEMP) < 400
    assert abs(p.tint) < 10


def test_ai_strength_zero_is_noop():
    lin = synthetic_scene()
    p = ai_correct(lin, Params(), strength=0.0)
    out = render(lin, p)
    assert np.abs(out - render(lin, Params())).max() < 2e-3


def test_illuminant_estimate_direction():
    e = estimate_illuminant(synthetic_scene((1.3, 1.0, 0.6)))
    assert e[0] > e[1] > e[2]


def test_params_roundtrip(tmp_path):
    p = Params(temp=4200, tint=12, contrast=30)
    p.curves["r"] = [[0, 0.02], [0.5, 0.55], [1, 1]]
    assert Params.from_dict(p.to_dict()) == p
    store = EditStore(str(tmp_path))
    img = str(tmp_path / "a.jpg")
    store.set(img, p)
    assert store.get(img) == p
    store.set(img, Params())
    assert store.get(img) is None


def test_load_and_export_jpeg_png(tmp_path):
    arr = (np.random.default_rng(2).random((120, 160, 3)) * 255).astype(np.uint8)
    src = tmp_path / "in.jpg"
    Image.fromarray(arr).save(src, quality=95)
    data = load_image(str(src))
    assert data.linear.shape == (120, 160, 3)
    p = Params(temp=5000, exposure=0.3)

    out_jpg = export_image(str(src), p, str(tmp_path / "out.jpg"), ExportOptions(fmt="jpeg", quality=90))
    assert Image.open(out_jpg).size == (160, 120)

    out_png = export_image(str(src), p, str(tmp_path / "out.png"), ExportOptions(fmt="png"))
    assert Image.open(out_png).mode == "RGB"

    out16 = export_image(str(src), p, str(tmp_path / "out16.png"), ExportOptions(fmt="png", png_bits=16))
    back = load_image(out16)
    assert back.linear.shape == (120, 160, 3)

    small = export_image(str(src), p, str(tmp_path / "small.jpg"), ExportOptions(long_edge=80))
    assert max(Image.open(small).size) == 80

    assert load_thumbnail(str(src), 64).shape[1] == 64


def test_png_alpha_preserved(tmp_path):
    arr = np.zeros((40, 50, 4), np.uint8)
    arr[..., :3] = 128
    arr[..., 3] = np.linspace(0, 255, 50, dtype=np.uint8)[None, :]
    src = tmp_path / "alpha.png"
    Image.fromarray(arr, "RGBA").save(src)
    data = load_image(str(src))
    assert data.alpha is not None
    out = export_image(str(src), Params(exposure=0.5), str(tmp_path / "o.png"), ExportOptions(fmt="png"))
    im = Image.open(out)
    assert im.mode == "RGBA"
    assert np.asarray(im)[0, -1, 3] == 255


@pytest.mark.skipif(not os.environ.get("SIE_RAW_SAMPLE"), reason="no RAW sample provided")
def test_raw_loading(tmp_path):
    path = os.environ["SIE_RAW_SAMPLE"]
    data = load_image(path)
    assert data.is_raw and data.linear.ndim == 3
    out = export_image(path, ai_correct(data.linear, Params()), str(tmp_path / "raw.jpg"), ExportOptions())
    assert os.path.getsize(out) > 0
    assert load_thumbnail(path).ndim == 3


def test_renderer_tiled_matches():
    lin = synthetic_scene(size=(300, 200))
    r = Renderer(Params(contrast=20, vibrance=30))
    assert np.allclose(r.apply(lin.copy()), r.apply_tiled(lin, rows=64))


# --- regressions found with real photos -------------------------------------

def _patch(img, y0, y1, x0, x1):
    m = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return m.max() / m.min()


def test_ai_keeps_neutral_ground_neutral_under_blue_sky():
    """Per-channel highlight curves must not push a neutral scene blue
    (bug: wrong sign in the highlight cast removal)."""
    h, w = 240, 360
    img = np.zeros((h, w, 3), np.float32)
    img[: h // 3] = (0.30, 0.42, 0.75)          # bright blue dusk sky
    img[h // 3:] = 0.16                         # neutral asphalt
    img[h // 3:, 200:260] = 0.62                # neutral white car
    p = ai_correct(img, Params())
    out = render(img, p)
    assert _patch(out, 200, 230, 20, 180) < 1.03
    assert _patch(out, 100, 200, 205, 255) < 1.03


def test_ai_ignores_saturated_screen_and_led():
    """A bright green monitor and a blue LED must not produce a strong tint."""
    h, w = 240, 360
    img = np.full((h, w, 3), 0.05, np.float32)  # dark desk mat
    img[:80, 200:] = (0.05, 0.85, 0.55)          # green/cyan screen
    img[200:, :60] = (0.02, 0.05, 0.9)           # blue LED glow
    img[100:180, 80:260] = 0.55                  # grey model car
    img[:80, :200] = (0.55, 0.32, 0.15)          # warm wooden desk
    p = ai_correct(img, Params())
    assert abs(p.tint) < 15, p.tint
    assert _patch(render(img, p), 110, 170, 90, 250) < 1.05


def test_ai_mixed_light_is_conservative():
    """When grey candidates split between two light sources the correction is
    milder than a full neutralisation of either."""
    from simon_enhance.auto import estimate_illuminant_conf
    from simon_enhance.processing import blackbody_rgb
    h, w = 200, 300
    warm = blackbody_rgb(4300) / blackbody_rgb(NEUTRAL_TEMP)
    cool = blackbody_rgb(11000) / blackbody_rgb(NEUTRAL_TEMP)
    img = np.zeros((h, w, 3), np.float32)
    img[:, :150] = 0.25 * warm / warm[1]
    img[:, 150:] = 0.25 * cool / cool[1]
    _, conf = estimate_illuminant_conf(img)
    assert conf < 0.8
