"""AI engine and features. The small upscaling model ships with the app and is always
tested. The large models (BiRefNet 224 MB, SCUNet 77 MB) are tested when they are
installed or SIE_AI_DOWNLOAD=1 allows downloading them."""
import os

import numpy as np
import pytest
from PIL import Image

from simon_enhance.ai import engine as E, models as M


@pytest.fixture(autouse=True)
def fresh_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "_probe_file", lambda: str(tmp_path / "ai_probe.json"))
    E.reset()
    E.set_preferred("auto")
    yield
    E.reset()


def _need(model_id):
    if not M.is_available(model_id):
        if os.environ.get("SIE_AI_DOWNLOAD") != "1":
            pytest.skip(f"Modell {model_id} nicht installiert (SIE_AI_DOWNLOAD=1 lädt es)")
        M.download(model_id)


def _photo(h=150, w=200):
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    img = np.stack([0.5 + 0.4 * np.sin(x * 9), 0.4 + 0.3 * y, 0.3 + 0.2 * np.cos(y * 7)], -1)
    img[40:90, 60:140] = (0.9, 0.85, 0.2)  # sharp-edged object
    return img.astype(np.float32)


def test_bundled_model_present_and_verified():
    import hashlib
    f = M.MODELS["realesr"].files[0]
    path = M.path_of(f)
    assert os.path.exists(path)
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == f.sha256
    assert M.ncnn_paths("realesr") is not None


def test_cpu_device_always_present():
    assert any(d.kind == "cpu" for d in E.discover(refresh=True))


def test_upscale_tiled_matches_shape_and_content():
    img = _photo()
    up = E.run_tiled("realesr", img)
    assert up.shape == (img.shape[0] * 4, img.shape[1] * 4, 3)
    small = np.asarray(Image.fromarray((up * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]),
                                                                          Image.Resampling.BOX), np.float32) / 255
    assert np.abs(small - img).mean() < 0.03      # same picture, just bigger
    assert E.last_device("realesr") is not None


def test_tiles_are_seamless():
    img = np.full((300, 300, 3), 0.5, np.float32)  # flat image: tile borders would show up
    up = E.run_tiled("realesr", img)
    assert up.std() < 0.01


def test_fallback_to_next_device(monkeypatch):
    bad = E.Device("ort:broken:0", "Kaputtes Gerät", "dgpu", "onnxruntime")
    devs = [bad] + [d for d in E.discover() if d.kind == "cpu"]
    monkeypatch.setattr(E, "discover", lambda refresh=False: devs)
    out = E.run("realesr", np.random.rand(1, 3, 32, 32).astype(np.float32))
    assert out.shape == (1, 3, 128, 128)
    assert E.last_device("realesr").kind == "cpu"
    assert ("realesr", bad.id) in E._failed


def test_vulkan_ncnn_matches_onnx(monkeypatch):
    monkeypatch.setenv("SIE_ALLOW_CPU_VULKAN", "1")
    vk = [d for d in E.discover(refresh=True) if d.runtime == "ncnn"]
    if not vk:
        pytest.skip("kein Vulkan-Gerät")
    x = np.random.default_rng(0).random((1, 3, 48, 48)).astype(np.float32)
    E.set_preferred("ort:cpu")
    ref = E.run("realesr", x)
    E.set_preferred(vk[0].id)
    out = E.run("realesr", x)
    assert E.last_device("realesr").runtime == "ncnn"
    assert np.abs(out - ref).mean() < 0.01          # fp16 on the GPU
    monkeypatch.delenv("SIE_ALLOW_CPU_VULKAN")
    E.discover(refresh=True)


def test_enhance_upscale_writes_16bit_tiff(tmp_path):
    from simon_enhance.ai import features as F
    from simon_enhance.fileio import load_image, load_source
    src = tmp_path / "s.png"
    Image.fromarray((_photo(120, 160) * 255).astype(np.uint8)).save(src)
    for scale, size in ((2, (320, 240)), (4, (640, 480))):
        dst = F.enhance_output_path(str(src), False, scale)
        seen = []
        F.enhance(str(src), dst, 0.0, scale, progress=seen.append)
        assert seen[-1] == 1.0
        d = load_source(dst)
        assert d.full_size == size and d.linear.kind == "srgb16"
        assert load_image(dst).linear.shape[:2] == (size[1], size[0])


def test_enhance_refuses_huge_output(tmp_path):
    from simon_enhance.ai import features as F
    src = tmp_path / "s.png"
    Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(src)
    old = F.MAX_OUTPUT_PIXELS
    F.MAX_OUTPUT_PIXELS = 100 * 100 * 4
    try:
        with pytest.raises(ValueError):
            F.enhance(str(src), str(tmp_path / "o.tif"), 0.0, 4)
    finally:
        F.MAX_OUTPUT_PIXELS = old


def test_sample_mask_bilinear():
    from simon_enhance.ai.features import sample_mask
    m = np.zeros((10, 10), np.float32)
    m[:, 5:] = 1
    full = sample_mask(m, 0, 100, 100, 100)
    assert full.shape == (100, 100)
    assert full[:, :40].max() == 0 and full[:, 60:].min() == 1
    strip = sample_mask(m, 30, 20, 100, 100)
    assert np.allclose(strip, full[30:50])


def test_download_verifies_checksum(tmp_path, monkeypatch):
    import io
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(M, "models_dir", lambda: str(tmp_path))
    f = M.ModelFile("fake.onnx", "https://example.invalid/fake.onnx", "0" * 64, 5)
    info = M.ModelInfo("fake", "Fake", "MIT", "-", (f,))
    monkeypatch.setitem(M.MODELS, "fake", info)

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    with pytest.raises(IOError):
        M.download("fake", opener=lambda url, timeout: R(b"hello"))
    assert not os.path.exists(tmp_path / "fake.onnx")  # never keeps an unverified file


def test_ai_background_mask_on_studio_image(tmp_path):
    _need("birefnet")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_greenscreen import studio
    from simon_enhance.ai import features as F
    from simon_enhance.processing import Params, Renderer, srgb_to_linear
    img, subject = studio(size=(400, 300))
    src = tmp_path / "studio.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(src)
    key = F.compute_mask(str(src), srgb_to_linear(img))
    p = Params(key_enabled=1.0, key_mode=1.0, key_mask=key)
    _, a = Renderer(p).apply_rgba(srgb_to_linear(img))
    assert a[subject].mean() > 0.9 and a[~subject].mean() < 0.1


def test_denoise_reduces_noise(tmp_path):
    _need("scunet")
    rng = np.random.default_rng(0)
    clean = _photo(128, 128)
    noisy = np.clip(clean + rng.normal(0, 0.06, clean.shape), 0, 1).astype(np.float32)
    out = E.run_tiled("scunet", noisy)
    assert np.abs(out - clean).mean() < 0.6 * np.abs(noisy - clean).mean()


def test_cidnet_bundled_and_neural_color_only_sets_wb_and_curves():
    from simon_enhance.auto import neural_color
    from simon_enhance.processing import Params, render, srgb_to_linear
    assert M.is_available("cidnet")
    import hashlib
    f = M.MODELS["cidnet"].files[0]
    assert hashlib.sha256(open(M.path_of(f), "rb").read()).hexdigest() == f.sha256
    lin = srgb_to_linear(_photo(300, 400) * 0.6)  # a little dark
    base = Params(exposure=0.1, saturation=5)
    p = neural_color(lin, base)
    for k in ("exposure", "contrast", "highlights", "shadows", "whites", "blacks", "vibrance", "saturation"):
        assert getattr(p, k) == getattr(base, k)
    assert p.curves["rgb"] != [[0.0, 0.0], [1.0, 1.0]]
    out = render(lin, p)
    assert np.isfinite(out).all() and out.mean() > render(lin, base).mean()  # brightens a dark photo
    assert neural_color(lin, base, strength=0.0).temp == pytest.approx(base.temp, abs=1)


def test_cidnet_network_output_is_sane():
    x = np.full((1, 3, 512, 512), 0.4, np.float32)
    x[:, :, 100:300, 100:300] = [[[0.8]], [[0.3]], [[0.2]]]
    y = E.run("cidnet", x)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert 0.0 <= y.min() and y.max() <= 1.0 + 1e-4
    # keeps the picture: the red square stays red, the background stays grey
    assert y[0, 0, 200, 200] > y[0, 1, 200, 200] + 0.2
    assert abs(float(y[0, 0, 20, 20]) - float(y[0, 2, 20, 20])) < 0.05


def test_openvino_runner_matches_cpu(monkeypatch):
    pytest.importorskip("openvino")
    monkeypatch.setenv("SIE_OPENVINO_ALLOW_CPU", "1")
    devs = [d for d in E.discover(refresh=True) if d.runtime == "openvino"]
    if not devs:
        pytest.skip("OpenVINO ohne Gerät")
    x = np.random.default_rng(1).random((1, 3, 64, 64)).astype(np.float32)
    E.set_preferred("ort:cpu")
    ref = E.run("realesr", x)
    E.set_preferred(devs[0].id)
    out = E.run("realesr", x)
    assert E.last_device("realesr").runtime == "openvino"
    assert np.abs(out - ref).mean() < 0.01
    monkeypatch.delenv("SIE_OPENVINO_ALLOW_CPU")
    E.discover(refresh=True)


def test_webgpu_runner_matches_cpu(monkeypatch):
    import onnxruntime as ort
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ONNX Runtime ohne WebGPU")
    monkeypatch.setenv("SIE_ALLOW_CPU_VULKAN", "1")
    devs = [d for d in E.discover(refresh=True) if d.id.startswith("ort:webgpu")]
    if not devs:
        pytest.skip("kein Vulkan-Gerät")
    x = np.random.default_rng(2).random((1, 3, 64, 64)).astype(np.float32)
    E.set_preferred("ort:cpu")
    ref = E.run("realesr", x)
    E.set_preferred(devs[0].id)
    out = E.run("realesr", x)
    assert E.last_device("realesr").id == devs[0].id
    assert np.abs(out - ref).mean() < 0.01
    monkeypatch.delenv("SIE_ALLOW_CPU_VULKAN")
    E.discover(refresh=True)


def _vulkan_device(monkeypatch):
    monkeypatch.setenv("SIE_ALLOW_CPU_VULKAN", "1")
    devs = [d for d in E.discover(refresh=True) if d.kind != "cpu" and d.runtime == "onnxruntime"]
    if not devs:
        pytest.skip("kein GPU-Gerät (auch kein Software-Vulkan)")
    return devs[0]


def test_probe_accepts_working_device_and_caches(monkeypatch):
    dev = _vulkan_device(monkeypatch)
    r = E._run_probe("realesr", dev)
    assert r["ok"] and r["diff"] < 0.05, r
    assert E._probe_ok("realesr", dev)
    import json
    cache = json.load(open(E._probe_file()))
    assert list(cache.values())[0]["ok"]


def test_driver_crash_is_contained_and_explained(monkeypatch):
    """A segfaulting driver must not take the app down: the device is skipped with a reason."""
    dev = _vulkan_device(monkeypatch)
    monkeypatch.setenv("SIE_PROBE_TEST_CRASH", dev.id)
    E.set_preferred(dev.id)
    out = E.run("realesr", np.random.rand(1, 3, 32, 32).astype(np.float32))
    assert out.shape == (1, 3, 128, 128)
    assert E.last_device("realesr").kind == "cpu"
    reasons = dict(E.fallback_reasons("realesr"))
    assert "abgestürzt" in reasons[dev.label]


def test_wrong_results_are_detected():
    ref = E.probe_input((1, 3, 64, 64))
    assert E.compare(ref * 1.001, ref) < 0.01
    assert E.compare(np.zeros_like(ref), ref) > 0.5           # black output (broken GPU kernel)
    assert E.compare(ref * np.nan, ref) == float("inf")


def test_crash_while_computing_only_ends_the_worker(monkeypatch):
    """The probe passed, but the driver crashes during the real work: the app survives,
    the tile is computed on the next device and the reason is kept."""
    dev = _vulkan_device(monkeypatch)
    E.set_preferred(dev.id)
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    ref = E.run("realesr", x)
    assert E.last_device("realesr").id == dev.id and dev.id in E._workers   # ran in a worker
    E._stop_worker(dev.id)
    monkeypatch.setenv("SIE_WORKER_TEST_CRASH", dev.id)
    out = E.run("realesr", x)
    assert E.last_device("realesr").kind == "cpu" and np.abs(out - ref).mean() < 0.02
    assert "abgestürzt" in dict(E.fallback_reasons("realesr"))[dev.label]
