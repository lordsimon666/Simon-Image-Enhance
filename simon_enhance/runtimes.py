"""Optional GPU runtimes that are downloaded on demand: NVIDIA CUDA and AMD ROCm.

The app ships ONNX Runtime with WebGPU/Vulkan (all GPU vendors) and OpenVINO (Intel NPU).
If an NVIDIA (CUDA) or AMD (ROCm) driver stack is found, the user is offered the
vendor-optimised ONNX Runtime build (cuDNN / MIGraphX - usually considerably faster,
uses Tensor / Matrix cores). It is installed into the user's data folder and loaded
*instead of* the bundled ONNX Runtime at the next start; removing it restores the
bundled one. The GPU driver itself always comes from the system.

* CUDA: needs the NVIDIA driver. cuBLAS / cuDNN / ... are used from the system when
  installed, otherwise the official NVIDIA library wheels are downloaded as well.
* ROCm: needs ROCm 7 incl. MIGraphX (or MIOpen/hipBLAS for the older ROCm build) on the
  system - those libraries are far too large and too version-specific to ship.

Linux only (on Windows DirectML already covers NVIDIA, AMD and Intel GPUs).
"""

from __future__ import annotations

import ctypes
import glob
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import platform
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field

from .store import data_dir

PYPI = "https://pypi.org/pypi"

SYSTEM_LIB_DIRS = ["/opt/rocm/lib", "/opt/rocm/lib64", "/usr/local/cuda/lib64", "/usr/lib64",
                   "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"]


@dataclass(frozen=True)
class Profile:
    id: str
    vendor: str                     # "nvidia" | "amd"
    title: str
    package: str                    # ONNX Runtime build on PyPI
    version: str                    # "" = latest
    system_libs: tuple              # libraries needed from the system / extra wheels
    lib_wheels: tuple = ()          # PyPI packages that provide system_libs (NVIDIA only)
    min_driver: int = 0             # CUDA driver API version (e.g. 13000)
    provider: str = ""
    install_hint: str = ""


PROFILES = {
    "cuda13": Profile(
        "cuda13", "nvidia", "NVIDIA CUDA 13 (cuDNN, Tensor-Kerne)", "onnxruntime-gpu", "",
        ("libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10",
         "libcufft.so.12", "libcudnn.so.9"),
        ("nvidia-cuda-runtime~=13.0", "nvidia-cublas~=13.0", "nvidia-curand~=10.0",
         "nvidia-cufft~=12.0", "nvidia-cudnn-cu13~=9.0", "nvidia-cuda-nvrtc~=13.0"),
        min_driver=13000, provider="CUDAExecutionProvider"),
    "cuda12": Profile(
        "cuda12", "nvidia", "NVIDIA CUDA 12 (cuDNN, Tensor-Kerne)", "onnxruntime-gpu", "1.26.0",
        ("libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10",
         "libcufft.so.11", "libcudnn.so.9"),
        ("nvidia-cuda-runtime-cu12~=12.0", "nvidia-cublas-cu12~=12.0", "nvidia-curand-cu12~=10.0",
         "nvidia-cufft-cu12~=11.0", "nvidia-cudnn-cu12~=9.0", "nvidia-cuda-nvrtc-cu12~=12.0"),
        min_driver=12000, provider="CUDAExecutionProvider"),
    "migraphx": Profile(
        "migraphx", "amd", "AMD ROCm 7 – MIGraphX", "onnxruntime-migraphx", "",
        ("libamdhip64.so.7", "libmigraphx_c.so.3"),
        provider="MIGraphXExecutionProvider",
        install_hint="Dafür muss ROCm 7 mit MIGraphX installiert sein (über die Paketquellen der "
                     "Distribution oder AMDs ROCm-Paketquelle)."),
    "rocm": Profile(
        "rocm", "amd", "AMD ROCm 7 – MIOpen (ältere Laufzeit)", "onnxruntime-rocm", "1.22.2.post3",
        ("libamdhip64.so.7", "libhipblas.so.3", "libMIOpen.so.1", "libhipfft.so.0",
         "librocm_smi64.so.1", "libroctracer64.so.4"),
        provider="ROCMExecutionProvider",
        install_hint="ROCm 7 mit MIOpen, hipBLAS und hipFFT installieren."),
}


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def _find_lib(name: str) -> str | None:
    """Path (or loadable name) of a shared library, or None."""
    try:
        ctypes.CDLL(name)
        return name
    except OSError:
        pass
    for d in SYSTEM_LIB_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def cuda_driver() -> dict | None:
    """NVIDIA driver info via libcuda (no CUDA toolkit needed)."""
    lib = _find_lib("libcuda.so.1")
    if not lib:
        return None
    try:
        cu = ctypes.CDLL(lib)
        ver = ctypes.c_int(0)
        if cu.cuDriverGetVersion(ctypes.byref(ver)) != 0:
            return None
        names = []
        if cu.cuInit(0) == 0:
            n = ctypes.c_int(0)
            cu.cuDeviceGetCount(ctypes.byref(n))
            for i in range(n.value):
                dev = ctypes.c_int(0)
                buf = ctypes.create_string_buffer(256)
                if cu.cuDeviceGet(ctypes.byref(dev), i) == 0 and cu.cuDeviceGetName(buf, 256, dev) == 0:
                    names.append(buf.value.decode(errors="replace"))
        return {"version": ver.value, "gpus": names}
    except Exception:
        return None


def rocm_driver() -> dict | None:
    """AMD ROCm: kernel driver (/dev/kfd) plus the HIP runtime."""
    if not os.path.exists("/dev/kfd"):
        return None
    hip = _find_lib("libamdhip64.so.7") or _find_lib("libamdhip64.so.6") or _find_lib("libamdhip64.so")
    return {"hip": hip, "rocm7": bool(_find_lib("libamdhip64.so.7"))}


@dataclass
class Offer:
    profile: Profile
    possible: bool
    reason: str = ""
    gpus: list = field(default_factory=list)
    missing: list = field(default_factory=list)   # system libs not found
    need_lib_wheels: bool = False                 # NVIDIA libs will be downloaded too


def detect() -> list[Offer]:
    """Which optional runtimes make sense on this machine (best first per vendor)."""
    if not sys.platform.startswith("linux") or platform.machine() not in ("x86_64", "AMD64"):
        return []
    offers: list[Offer] = []
    cu = cuda_driver()
    if cu:
        drv = cu["version"]
        for pid in ("cuda13", "cuda12"):
            p = PROFILES[pid]
            if drv >= p.min_driver:
                missing = [l for l in p.system_libs if not _find_lib(l)]
                offers.append(Offer(p, True, f"NVIDIA-Treiber mit CUDA {drv // 1000}.{(drv % 1000) // 10}",
                                    cu["gpus"], missing, need_lib_wheels=bool(missing)))
                break
        else:
            offers.append(Offer(PROFILES["cuda12"], False,
                                f"NVIDIA-Treiber zu alt (CUDA {drv // 1000}.{(drv % 1000) // 10}, "
                                "mindestens CUDA 12 nötig)", cu["gpus"]))
    rocm = rocm_driver()
    if rocm:
        best = None
        for pid in ("migraphx", "rocm"):
            p = PROFILES[pid]
            missing = [l for l in p.system_libs if not _find_lib(l)]
            if not missing:
                best = Offer(p, True, "ROCm gefunden", ["AMD-GPU"])
                break
            if best is None:
                best = Offer(p, False, "ROCm gefunden, aber benötigte Bibliotheken fehlen: " + ", ".join(missing),
                             ["AMD-GPU"], missing)
        offers.append(best)
    return offers


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------

def runtimes_dir() -> str:
    d = os.path.join(data_dir(), "runtimes")
    os.makedirs(d, exist_ok=True)
    return d


def site_dir(profile_id: str) -> str:
    return os.path.join(runtimes_dir(), profile_id, "site")


def installed() -> dict:
    """{profile_id: info} of installed runtimes."""
    out = {}
    for pid in PROFILES:
        f = os.path.join(runtimes_dir(), pid, "installed.json")
        if os.path.exists(f):
            try:
                out[pid] = json.load(open(f, encoding="utf-8"))
            except Exception:
                pass
    return out


def active() -> str | None:
    try:
        pid = json.load(open(os.path.join(runtimes_dir(), "active.json"), encoding="utf-8")).get("profile")
    except Exception:
        return None
    return pid if pid in installed() else None


def set_active(profile_id: str | None) -> None:
    f = os.path.join(runtimes_dir(), "active.json")
    if profile_id is None:
        if os.path.exists(f):
            os.remove(f)
        return
    json.dump({"profile": profile_id}, open(f, "w", encoding="utf-8"))


def remove(profile_id: str) -> None:
    if active() == profile_id:
        set_active(None)
    shutil.rmtree(os.path.join(runtimes_dir(), profile_id), ignore_errors=True)


def _glibc() -> tuple:
    try:
        v = os.confstr("CS_GNU_LIBC_VERSION").split()[1]
        return tuple(int(x) for x in v.split(".")[:2])
    except Exception:
        return (2, 28)


def _wheel_ok(filename: str) -> bool:
    py = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    if not filename.endswith(".whl") or "x86_64" not in filename:
        return False
    tags = filename[:-4].split("-")
    pytag, abi, plat = tags[-3], tags[-2], tags[-1]
    if not (pytag in (py, "py3") or pytag.startswith("py3")) or abi not in (py, "none", "abi3"):
        return False
    for m in re.finditer(r"manylinux_(\d+)_(\d+)", plat):
        if (int(m.group(1)), int(m.group(2))) <= _glibc():
            return True
    return "manylinux2014" in plat or "manylinux1" in plat


def _version_key(v: str):
    return [int(x) if x.isdigit() else 0 for x in re.split(r"[.+]", v)]


def _matches(version: str, spec: str) -> bool:
    """Minimal '~=X.Y' / '==X' / '>=X' matcher for the specifiers used here."""
    if not spec:
        return True
    op, val = re.match(r"(~=|==|>=)\s*(.+)", spec).groups()
    v, w = _version_key(version), _version_key(val)
    if op == "==":
        return version == val
    if op == ">=":
        return v >= w
    return v >= w and v[:len(w) - 1] == w[:len(w) - 1]   # ~=


def _fetch_json(url: str, opener=urllib.request.urlopen):
    with opener(url, timeout=60) as r:
        return json.load(r)


def _pick_release(name: str, spec: str, opener=urllib.request.urlopen) -> dict:
    meta = _fetch_json(f"{PYPI}/{name}/json", opener)
    versions = sorted((v for v in meta["releases"] if re.fullmatch(r"[\d.]+(post\d+)?", v) and _matches(v, spec)),
                      key=_version_key, reverse=True)
    for v in versions:
        files = [f for f in meta["releases"][v] if _wheel_ok(f["filename"]) and not f.get("yanked")]
        if files:
            files.sort(key=lambda f: f["filename"])
            rel = _fetch_json(f"{PYPI}/{name}/{v}/json", opener)
            f = files[-1]
            return {"name": name, "version": v, "url": f["url"], "sha256": f["digests"]["sha256"],
                    "size": f["size"], "filename": f["filename"],
                    "requires": rel["info"].get("requires_dist") or []}
    raise RuntimeError(f"Kein passendes Paket {name} {spec} für Python "
                       f"{sys.version_info[0]}.{sys.version_info[1]} gefunden")


def plan(offer: Offer, opener=urllib.request.urlopen) -> list[dict]:
    """Wheels to download for an offer (ONNX Runtime build + NVIDIA libs if needed)."""
    p = offer.profile
    spec = f"=={p.version}" if p.version else ""
    items = [_pick_release(p.package, spec, opener)]
    if offer.need_lib_wheels:
        todo = list(p.lib_wheels)
        seen = set()
        while todo:
            req = todo.pop(0)
            m = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", req)
            name, spec = m.group(1).lower(), m.group(2).split(";")[0].strip()
            if name in seen:
                continue
            seen.add(name)
            rel = _pick_release(name, spec, opener)
            items.append(rel)
            for r in rel["requires"]:  # NVIDIA wheels depend on each other (cudnn -> cublas)
                if r.lower().startswith("nvidia-") and "extra ==" not in r:
                    todo.append(r)
    return items


class Cancelled(Exception):
    pass


def install(offer: Offer, progress=None, opener=urllib.request.urlopen, items=None) -> None:
    """Download (SHA-256 verified) and unpack into the runtime folder, then activate."""
    p = offer.profile
    items = items or plan(offer, opener)
    total = sum(i["size"] for i in items) or 1
    base = os.path.join(runtimes_dir(), p.id)
    tmp = base + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "site"))
    done = 0
    for it in items:
        whl = os.path.join(tmp, it["filename"])
        h = hashlib.sha256()
        with opener(it["url"], timeout=120) as r, open(whl, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress is not None and progress(done / total) is False:
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise Cancelled()
        if h.hexdigest() != it["sha256"]:
            shutil.rmtree(tmp, ignore_errors=True)
            raise IOError(f"Prüfsumme von {it['filename']} stimmt nicht – Download abgebrochen.")
        with zipfile.ZipFile(whl) as z:
            z.extractall(os.path.join(tmp, "site"))
        os.remove(whl)
    json.dump({"profile": p.id, "packages": [{k: i[k] for k in ("name", "version", "sha256")} for i in items]},
              open(os.path.join(tmp, "installed.json"), "w", encoding="utf-8"), indent=1)
    shutil.rmtree(base, ignore_errors=True)
    os.replace(tmp, base)
    set_active(p.id)


# ---------------------------------------------------------------------------
# activation (must run before onnxruntime is imported)
# ---------------------------------------------------------------------------

class _OverrideFinder(importlib.abc.MetaPathFinder):
    """Loads the top-level 'onnxruntime' package from the runtime folder, ahead of the
    copy bundled with the app (also inside PyInstaller builds)."""

    def __init__(self, path: str):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "onnxruntime":
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, [self.path])


_state = {"active": None, "error": None, "finder": None}


def _preload(site: str, profile: Profile) -> None:
    """Load the needed CUDA / ROCm libraries globally so ONNX Runtime finds them:
    first from the downloaded NVIDIA wheels, then from the usual system folders."""
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    order = ["cudart", "nvrtc", "cublasLt", "cublas", "curand", "cufft", "cudnn"]
    libs = glob.glob(os.path.join(site, "nvidia", "*", "lib", "*.so*"))
    libs.sort(key=lambda f: next((i for i, k in enumerate(order) if os.path.basename(f).startswith("lib" + k)), 99))
    for f in libs:
        try:
            ctypes.CDLL(f, mode=mode)
        except OSError:
            pass
    for name in profile.system_libs:
        lib = _find_lib(name)
        if lib and os.path.isabs(lib):
            try:
                ctypes.CDLL(lib, mode=mode)
            except OSError:
                pass


def activate() -> str | None:
    """Use the installed runtime for this process. Returns the active profile id."""
    if "onnxruntime" in sys.modules or os.environ.get("SIE_GPU_RUNTIME") == "0":
        return None
    pid = active()
    if not pid:
        return None
    site = site_dir(pid)
    if not os.path.isdir(os.path.join(site, "onnxruntime")):
        _state["error"] = "Laufzeit unvollständig – bitte neu installieren"
        return None
    _preload(site, PROFILES[pid])
    finder = _OverrideFinder(site)
    sys.meta_path.insert(0, finder)
    sys.path.append(site)  # for the 'nvidia' namespace packages
    _state.update(active=pid, finder=finder)
    return pid


def deactivate_after_failure(exc: Exception) -> None:
    """Called when importing the downloaded runtime failed: fall back to the bundled one."""
    f = _state.get("finder")
    if f in sys.meta_path:
        sys.meta_path.remove(f)
    for k in [k for k in sys.modules if k == "onnxruntime" or k.startswith("onnxruntime.")]:
        del sys.modules[k]
    _state.update(error=f"{_state.get('active')}: {exc}", active=None)


def status() -> str:
    if _state["active"]:
        return f"Zusätzliche GPU-Laufzeit aktiv: {PROFILES[_state['active']].title}"
    if _state["error"]:
        return f"Zusätzliche GPU-Laufzeit nicht nutzbar: {_state['error']}"
    return "Keine zusätzliche GPU-Laufzeit aktiv"


def download_size(offer: Offer, opener=urllib.request.urlopen) -> int:
    return sum(i["size"] for i in plan(offer, opener))
