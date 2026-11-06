"""OpenCL acceleration. On machines without a GPU the tests use a CPU OpenCL
implementation (e.g. PoCL) if installed, otherwise they are skipped."""
import numpy as np
import pytest

from simon_enhance import gpu
from simon_enhance.greenscreen import KeyMatte, _box_axis, ai_remove_background
from simon_enhance.processing import Params, srgb_to_linear


@pytest.fixture
def opencl(monkeypatch):
    monkeypatch.setenv("SIE_OPENCL_ALLOW_CPU", "1")
    monkeypatch.delenv("SIE_GPU", raising=False)
    monkeypatch.setattr(gpu, "_state", dict(gpu._state, tried=False, ctx=None, enabled=True))
    if not gpu.available():
        pytest.skip("no OpenCL device available: " + str(gpu._state["error"]))
    yield
    gpu._state["tried"] = False


@pytest.mark.parametrize("r", [1, 4, 24, 32])
def test_gpu_box_matches_cpu(opencl, r):
    x = np.random.default_rng(r).random((600, 700)).astype(np.float32)
    g = gpu.box(x, r)
    assert g is not None
    assert np.abs(g - _box_axis(_box_axis(x, r, 0), r, 1)).max() < 1e-5


def test_matte_identical_on_gpu_and_cpu(opencl):
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from test_greenscreen import studio
    img, _ = studio(size=(900, 700))
    lin = srgb_to_linear(img)
    p, _ = ai_remove_background(lin, Params())
    k = KeyMatte(p, lin.shape[0])
    gpu.set_enabled(True)
    c1, a1 = k.apply(lin)
    gpu.set_enabled(False)
    c2, a2 = k.apply(lin)
    gpu.set_enabled(True)
    assert np.abs(a1 - a2).max() < 1e-4 and np.abs(c1 - c2).max() < 1e-4


def test_small_inputs_and_disabled_use_cpu(opencl):
    assert gpu.box(np.zeros((10, 10), np.float32), 2) is None  # transfer not worth it
    gpu.set_enabled(False)
    assert gpu.box(np.zeros((1000, 1000), np.float32), 2) is None
    assert "ausgeschaltet" in gpu.status()
    gpu.set_enabled(True)


def test_env_switch_disables_gpu(monkeypatch):
    monkeypatch.setenv("SIE_GPU", "0")
    monkeypatch.setattr(gpu, "_state", dict(gpu._state, tried=False, ctx=None, enabled=True))
    assert not gpu.available()
    assert gpu.box(np.zeros((1000, 1000), np.float32), 2) is None
    assert "SIE_GPU=0" in gpu.status()
    monkeypatch.setattr(gpu, "_state", dict(gpu._state, tried=False))
