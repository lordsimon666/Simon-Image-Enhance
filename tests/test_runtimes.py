"""Optional CUDA / ROCm runtimes: detection, package resolution, verified install and the
import override (tested with a fake 'onnxruntime' package - no GPU needed)."""
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

import pytest

from simon_enhance import runtimes as R

PY = f"cp{sys.version_info[0]}{sys.version_info[1]}"


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(R, "data_dir", lambda: str(tmp_path))
    return tmp_path


def _wheel(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return buf.getvalue()


class FakePyPI:
    """Serves PyPI JSON and wheel downloads from memory."""

    def __init__(self, packages):  # {name: {version: (wheel_bytes, requires)}}
        self.packages = packages

    def _meta(self, name):
        rel = {}
        for v, (data, _req) in self.packages[name].items():
            fn = f"{name.replace('-', '_')}-{v}-{PY}-{PY}-manylinux_2_28_x86_64.whl"
            rel[v] = [{"filename": fn, "url": f"https://files.example/{fn}", "size": len(data),
                       "digests": {"sha256": hashlib.sha256(data).hexdigest()}}]
        return rel

    def __call__(self, url, timeout=None):
        if url.startswith(R.PYPI):
            parts = url[len(R.PYPI) + 1:].split("/")
            name = parts[0]
            if len(parts) == 2:  # /name/json
                body = {"releases": self._meta(name), "info": {}}
            else:                # /name/version/json
                body = {"info": {"requires_dist": self.packages[name][parts[1]][1]}}
            return _Resp(json.dumps(body).encode())
        fn = url.rsplit("/", 1)[1]
        for name, vers in self.packages.items():
            for v, (data, _r) in vers.items():
                if fn.startswith(f"{name.replace('-', '_')}-{v}-"):
                    return _Resp(data)
        raise FileNotFoundError(url)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return super().read(n) if n != -1 else super().read()


def test_version_matching_and_wheel_tags():
    assert R._matches("13.2.1", "~=13.0") and not R._matches("14.0", "~=13.0")
    assert R._matches("1.26.0", "==1.26.0") and not R._matches("1.27.0", "==1.26.0")
    assert R._wheel_ok(f"x-1.0-{PY}-{PY}-manylinux_2_28_x86_64.whl")
    assert R._wheel_ok("nvidia_cublas-13.0-py3-none-manylinux_2_27_x86_64.whl")
    assert not R._wheel_ok(f"x-1.0-{PY}-{PY}-win_amd64.whl")
    assert not R._wheel_ok("x-1.0-cp27-cp27m-manylinux_2_28_x86_64.whl")


def test_detect_nvidia_and_amd(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "linux")
    monkeypatch.setattr(R.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(R, "cuda_driver", lambda: {"version": 13020, "gpus": ["RTX Test"]})
    monkeypatch.setattr(R, "rocm_driver", lambda: {"hip": "libamdhip64.so.7", "rocm7": True})
    have = {"libamdhip64.so.7", "libcudart.so.13"}
    monkeypatch.setattr(R, "_find_lib", lambda n: n if n in have else None)
    offers = {o.profile.id: o for o in R.detect()}
    assert offers["cuda13"].possible and offers["cuda13"].need_lib_wheels  # cuDNN etc. missing
    assert "libcudnn.so.9" in offers["cuda13"].missing
    amd = [o for o in offers.values() if o.profile.vendor == "amd"][0]
    assert not amd.possible and "libmigraphx_c.so.3" in amd.reason
    # old driver -> CUDA 12 build
    monkeypatch.setattr(R, "cuda_driver", lambda: {"version": 12040, "gpus": []})
    assert [o.profile.id for o in R.detect() if o.profile.vendor == "nvidia"] == ["cuda12"]
    # too old
    monkeypatch.setattr(R, "cuda_driver", lambda: {"version": 11080, "gpus": []})
    nv = [o for o in R.detect() if o.profile.vendor == "nvidia"][0]
    assert not nv.possible


def test_plan_resolves_nvidia_dependencies(monkeypatch):
    pypi = FakePyPI({
        "onnxruntime-gpu": {"1.31.1": (b"x", []), "1.26.0": (b"y", [])},
        "nvidia-cuda-runtime": {"13.0.1": (b"a", []), "12.9.0": (b"b", [])},
        "nvidia-cublas": {"13.1.0": (b"c", [])},
        "nvidia-curand": {"10.4.0": (b"d", [])},
        "nvidia-cufft": {"12.0.0": (b"e", ["nvidia-nvjitlink"])},
        "nvidia-cudnn-cu13": {"9.10.0": (b"f", ["nvidia-cublas"])},
        "nvidia-cuda-nvrtc": {"13.0.0": (b"g", [])},
        "nvidia-nvjitlink": {"13.0.0": (b"h", [])},
    })
    offer = R.Offer(R.PROFILES["cuda13"], True, need_lib_wheels=True)
    items = R.plan(offer, opener=pypi)
    names = [i["name"] for i in items]
    assert names[0] == "onnxruntime-gpu" and items[0]["version"] == "1.31.1"
    assert "nvidia-nvjitlink" in names and names.count("nvidia-cublas") == 1
    assert [i for i in items if i["name"] == "nvidia-cuda-runtime"][0]["version"] == "13.0.1"
    # pinned version for CUDA 12
    items12 = R.plan(R.Offer(R.PROFILES["cuda12"], True), opener=pypi)
    assert items12[0]["version"] == "1.26.0"


def test_install_verifies_checksum_and_activates(data):
    good = _wheel({"onnxruntime/__init__.py": "__version__ = 'fake-gpu'\n"})
    pypi = FakePyPI({"onnxruntime-gpu": {"1.31.1": (good, [])}})
    offer = R.Offer(R.PROFILES["cuda13"], True)
    items = R.plan(offer, opener=pypi)
    bad = [dict(items[0], sha256="0" * 64)]
    with pytest.raises(IOError):
        R.install(offer, opener=pypi, items=bad)
    assert "cuda13" not in R.installed() and R.active() is None
    seen = []
    R.install(offer, progress=seen.append, opener=pypi, items=items)
    assert seen[-1] == pytest.approx(1.0)
    assert R.active() == "cuda13" and "cuda13" in R.installed()
    R.remove("cuda13")
    assert R.active() is None and not R.installed()


def test_activation_overrides_bundled_onnxruntime(data):
    """In a fresh process the downloaded package must win over the installed one."""
    wheel = _wheel({"onnxruntime/__init__.py": "__version__ = 'fake-gpu'\n"
                                               "def get_available_providers(): return ['CUDAExecutionProvider', 'CPUExecutionProvider']\n"})
    pypi = FakePyPI({"onnxruntime-gpu": {"1.31.1": (wheel, [])}})
    R.install(R.Offer(R.PROFILES["cuda13"], True), opener=pypi)
    code = ("from simon_enhance import runtimes as R; R.data_dir = lambda: %r; print(R.activate());"
            "import onnxruntime as o; print(o.__version__)") % str(data)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env=dict(os.environ, PYTHONPATH=root)).stdout.split()
    assert out == ["cuda13", "fake-gpu"]
    # SIE_GPU_RUNTIME=0 switches the override off
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env=dict(os.environ, PYTHONPATH=root, SIE_GPU_RUNTIME="0")).stdout.split()
    assert out[0] == "None" and out[1] != "fake-gpu"


def test_broken_runtime_falls_back_to_bundled(data):
    wheel = _wheel({"onnxruntime/__init__.py": "raise ImportError('libcublas.so.13 missing')\n"})
    pypi = FakePyPI({"onnxruntime-gpu": {"1.31.1": (wheel, [])}})
    R.install(R.Offer(R.PROFILES["cuda13"], True), opener=pypi)
    code = ("from simon_enhance import runtimes as R; R.data_dir = lambda: %r; R.activate();"
            "from simon_enhance.ai import engine as E; ort = E._ort();"
            "print(ort is not None and 'fake' not in getattr(ort, '__version__', 'fake'), R.status())") % str(data)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                         env=dict(os.environ, PYTHONPATH=root)).stdout
    assert out.startswith("True") and "nicht nutzbar" in out
