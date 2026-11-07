"""Optional GPU runtimes that are downloaded on demand: NVIDIA CUDA and AMD ROCm.

The app ships ONNX Runtime with WebGPU/Vulkan (all GPU vendors) and OpenVINO (Intel NPU).
If an NVIDIA (CUDA) or AMD (ROCm) driver stack is found, the user is offered the
vendor-optimised ONNX Runtime build (cuDNN / MIGraphX - usually considerably faster,
uses Tensor / Matrix cores). It is installed into the user's data folder and loaded
*instead of* the bundled ONNX Runtime at the next start; removing it restores the
bundled one. The GPU driver itself always comes from the system.

* CUDA: needs the NVIDIA driver. cuBLAS / cuDNN / ... are used from the system when
  installed, otherwise the official NVIDIA library wheels are downloaded as well.
* ROCm 10: AMD ships the whole ROCm user space as Python wheels (TheRock). The app
  downloads them for the detected GPU (gfx target) together with the MIGraphX
  execution-provider plugin for ONNX Runtime 1.29 - nothing has to be installed on the
  system except the amdgpu kernel driver (part of every current kernel) and access to
  /dev/kfd. The AMD wheels are pinned to exact files and SHA-256 sums.

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
import zipfile
from dataclasses import dataclass, field

from . import net
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
    pinned: tuple = ()              # exact wheels (dicts with name/version/url/sha256/size/filename)
    devices: dict = field(default_factory=dict)   # gfx target -> extra pinned wheel
    plugin: str = ""                # ONNX Runtime EP plugin module to register after import
    python: tuple = ()              # required Python version (the plugin is a CPython extension)
    disk_gb: float = 0.0            # space needed after unpacking (for the texts)


AMD_REPO = "https://stable.repo.amd.com/rocm"
ROCM_VERSION = "10.1.0"


def _amd(name: str, version: str, sub: str, filename: str, sha256: str, size: int) -> dict:
    return {"name": name, "version": version, "url": f"{AMD_REPO}/{sub}/whl-next/{name}/{filename}",
            "sha256": sha256, "size": size, "filename": filename.replace("%2B", "+")}


def _gfx_wheel(gfx: str, sha256: str, size: int) -> dict:
    return _amd(f"rocm-sdk-device-{gfx}", ROCM_VERSION, "core",
                f"rocm_sdk_device_{gfx}-{ROCM_VERSION}-py3-none-linux_x86_64.whl", sha256, size)


# Resolved once with pip against AMD's index for ROCm 10.1.0 (cp312, manylinux_2_28).
_ROCM10_WHEELS = (
    {"name": "onnxruntime", "version": "1.29.0", "size": 23136745,
     "url": "https://files.pythonhosted.org/packages/96/eb/e6968f5e41aac3125f2ff5708855f09cb0b70d85ed3115b625b0b58305ba/"
            "onnxruntime-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl",
     "sha256": "2b80d8c7ec2cc7438e4da3760b88c24568cba72c9ace96d668800a6c79419acb",
     "filename": "onnxruntime-1.29.0-cp312-cp312-manylinux_2_28_x86_64.whl"},
    _amd("onnxruntime-ep-migraphx", "1.0.0+rocm10.1.0", "onnxruntime",
         "onnxruntime_ep_migraphx-1.0.0%2Brocm10.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
         "f43aafe7581fc7f82f94518424c86418371cdcc31c2cb399cec14e7d5c8cd19c", 1808763),
    _amd("migraphx-libs", "2.18.0+rocm10.1.0", "migraphx",
         "migraphx_libs-2.18.0%2Brocm10.1.0-py3-none-manylinux_2_28_x86_64.whl",
         "0a625d0fc835db4727af8b6199a0a85bcaa0578cc2a873f6fbbe36b70d84713b", 82963723),
    _amd("rocm-sdk-core", ROCM_VERSION, "core", f"rocm_sdk_core-{ROCM_VERSION}-py3-none-linux_x86_64.whl",
         "a1a2433334ac61a526c6bae2f58f6bbc9b525f5a9e008e76a1a48d07f64f0703", 419364838),
    _amd("rocm-sdk-libraries", ROCM_VERSION, "core", f"rocm_sdk_libraries-{ROCM_VERSION}-py3-none-linux_x86_64.whl",
         "ffb36d5e1d97e96c191052a10c39a1a7a4ddb4da4a88d3166ab4bf7bfdb94cf6", 260814462),
)

# GPUs for which AMD supports ONNX Runtime + MIGraphX on ROCm 10.1 (consumer cards / APUs)
_ROCM10_DEVICES = {
    "gfx1100": _gfx_wheel("gfx1100", "55fc9d7505c78178060d11bc8dc44b52b3e715f319bd61dd7e7f37c5bec2a94d", 233374792),
    "gfx1101": _gfx_wheel("gfx1101", "518834a7706da3f448a780d6ec0f96c2d3b4246b6e2b7cc503a5ee219c786b13", 227269260),
    "gfx1102": _gfx_wheel("gfx1102", "c621497b31bba1551f725e9ae0d39406355163852a56add2e5c73ce9ea295abe", 225388997),
    "gfx1150": _gfx_wheel("gfx1150", "aee780f79922bc2cc21803ef6fe528b1882523fcc4a27bde5d389df09b9b7529", 155303000),
    "gfx1151": _gfx_wheel("gfx1151", "e4e4d9b9a5020d0b971aec956f024514ced3ba80857aad7c42ef57bee9ebba5f", 177774184),
    "gfx1152": _gfx_wheel("gfx1152", "e070fe1682572d32f60afca8220329636929a6c1d3c2638266a06be449fd20fb", 155150763),
    "gfx1153": _gfx_wheel("gfx1153", "5ee81c972a4314a8c8ba0e492dfe0f4a7bd760dc20515d535b7bd7df5a696b1c", 155302956),
    "gfx1200": _gfx_wheel("gfx1200", "60433865c41a1a0cf25f1ecf850b8109c11385d8fdb16b229efbc314f77906b1", 405199317),
    "gfx1201": _gfx_wheel("gfx1201", "78e4360ff25872598563acf4bccaab2360e9f837601c74cf273640ad395cbe01", 343467033),
}

GFX_NAMES = {
    "gfx1100": "Radeon RX 7900 / PRO W7900", "gfx1101": "Radeon RX 7700 / 7800", "gfx1102": "Radeon RX 7600",
    "gfx1103": "Radeon 780M / 760M (Phoenix)", "gfx1150": "Radeon 890M / 880M (Strix Point)",
    "gfx1151": "Radeon 8060S / 8050S (Strix Halo)", "gfx1152": "Radeon 860M / 840M (Krackan Point)",
    "gfx1153": "Radeon (gfx1153)", "gfx1200": "Radeon RX 9060", "gfx1201": "Radeon RX 9070 / AI PRO R9700",
    "gfx1030": "Radeon RX 6800 / 6900", "gfx1031": "Radeon RX 6700", "gfx1032": "Radeon RX 6600",
    "gfx1033": "Steam Deck (Van Gogh)", "gfx1034": "Radeon RX 6500 / 6400", "gfx1035": "Radeon 680M / 660M",
    "gfx1036": "Radeon (Raphael / Granite Ridge)",
}


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
    "rocm10": Profile(
        "rocm10", "amd", "AMD ROCm 10 – MIGraphX", "onnxruntime", "1.29.0", (),
        provider="MIGraphXExecutionProvider", pinned=_ROCM10_WHEELS, devices=_ROCM10_DEVICES,
        plugin="onnxruntime_ep_migraphx", python=(3, 12), disk_gb=4.5),
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


def gfx_name(target_version: int) -> str:
    """KFD gfx_target_version (major*10000 + minor*100 + stepping) -> 'gfx1100', 'gfx90a'."""
    major, minor, step = target_version // 10000, (target_version // 100) % 100, target_version % 100
    return f"gfx{major}{minor}{step:x}"


def rocm_driver(root: str = "/") -> dict | None:
    """AMD compute driver (amdgpu/KFD): GPUs as gfx targets and whether /dev/kfd is usable.
    ROCm itself does not have to be installed (ROCm 10 is downloaded by the app)."""
    kfd = os.path.join(root, "dev/kfd")
    gfx = []
    for prop in sorted(glob.glob(os.path.join(root, "sys/class/kfd/kfd/topology/nodes/*/properties"))):
        try:
            for line in open(prop, encoding="ascii", errors="replace"):
                k, _, v = line.partition(" ")
                if k == "gfx_target_version" and int(v) > 0:
                    name = gfx_name(int(v))
                    if name not in gfx:
                        gfx.append(name)
        except (OSError, ValueError):
            pass
    if not os.path.exists(kfd) and not gfx:
        return None
    return {"kfd": os.path.exists(kfd), "access": os.access(kfd, os.R_OK | os.W_OK), "gfx": gfx}


@dataclass
class Offer:
    profile: Profile
    possible: bool
    reason: str = ""
    gpus: list = field(default_factory=list)
    missing: list = field(default_factory=list)   # system libs not found
    need_lib_wheels: bool = False                 # NVIDIA libs will be downloaded too
    target: str = ""                              # AMD gfx target the download is for
    quiet: bool = False                           # nothing the user can do: no start-up pop-up


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
        offers.append(_rocm10_offer(rocm))
    return offers


def _rocm10_offer(rocm: dict) -> Offer:
    p = PROFILES["rocm10"]
    names = [f"{GFX_NAMES.get(g, 'AMD-GPU')} ({g})" for g in rocm["gfx"]] or ["AMD-GPU"]
    target = next((g for g in rocm["gfx"] if g in p.devices), "")
    if not target:
        found = ", ".join(rocm["gfx"]) or "unbekannt"
        return Offer(p, False, f"Ihre AMD-Grafik ({found}) wird von ROCm 10 mit ONNX Runtime nicht unterstützt "
                               "– sie wird weiterhin über Vulkan genutzt", names, quiet=True)
    if sys.version_info[:2] != p.python:
        return Offer(p, False, "Diese Ausgabe der App läuft mit Python "
                               f"{sys.version_info[0]}.{sys.version_info[1]}; ROCm 10 braucht Python 3.12",
                     names, quiet=True)
    if not rocm["access"]:
        why = ("/dev/kfd fehlt (amdgpu-Rechentreiber nicht geladen)" if not rocm["kfd"]
               else "kein Zugriff auf /dev/kfd")
        return Offer(p, False, f"ROCm 10 kann für {GFX_NAMES.get(target, target)} genutzt werden, aber es besteht "
                               f"{why}", names, target=target)
    return Offer(p, True, f"AMD-Grafik {target} wird von ROCm 10 unterstützt", names, target=target)


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


def _fetch_json(url: str, opener=net.urlopen):
    with opener(url, timeout=60) as r:
        return json.load(r)


def _pick_release(name: str, spec: str, opener=net.urlopen) -> dict:
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


def plan(offer: Offer, opener=net.urlopen) -> list[dict]:
    """Wheels to download for an offer (ONNX Runtime build + NVIDIA libs if needed)."""
    p = offer.profile
    if p.pinned:
        if p.devices and offer.target not in p.devices:
            raise RuntimeError(f"Keine ROCm-Pakete für {offer.target or 'diese GPU'}")
        return [dict(i) for i in p.pinned] + ([dict(p.devices[offer.target])] if p.devices else [])
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


def install(offer: Offer, progress=None, opener=net.urlopen, items=None) -> None:
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
    json.dump({"profile": p.id, "target": offer.target, "packages": [{k: i[k] for k in ("name", "version", "sha256")} for i in items]},
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

    names = ("onnxruntime",)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, [self.path])


_state = {"active": None, "error": None, "finder": None, "registered": None}


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
    if PROFILES[pid].plugin == "onnxruntime_ep_migraphx":
        # MIGraphX compiles every model for the GPU (minutes for SCUNet); keep the result,
        # so the crash-safe check in a child process and later runs reuse it
        cache = os.path.join(runtimes_dir(), pid, "migraphx-cache")
        os.makedirs(cache, exist_ok=True)
        os.environ.setdefault("ORT_MIGRAPHX_CACHE_DIR", cache)
    _preload(site, PROFILES[pid])
    finder = _OverrideFinder(site)
    sys.meta_path.insert(0, finder)
    sys.path.append(site)  # 'nvidia' namespace packages, ROCm plugin and library packages
    _state.update(active=pid, finder=finder)
    return pid


def register_plugins(ort) -> list[str]:
    """After 'import onnxruntime': register the execution-provider plugin of the active
    runtime (ROCm 10: MIGraphX). Returns the registered EP names."""
    pid = _state.get("active")
    if not pid or not PROFILES[pid].plugin or _state.get("registered") is not None:
        return _state.get("registered") or []
    _state["registered"] = []
    try:
        mod = importlib.import_module(PROFILES[pid].plugin)
        for name, lib in zip(mod.get_ep_names(), mod.get_library_paths()):
            ort.register_execution_provider_library(name, lib)
            _state["registered"].append(name)
    except Exception as exc:  # noqa: BLE001 - the bundled providers keep working
        _state["error"] = f"{PROFILES[pid].title}: {exc}"
    return _state["registered"]


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
        txt = f"Zusätzliche GPU-Laufzeit aktiv: {PROFILES[_state['active']].title}"
        if _state["error"]:
            txt += f" – Fehler: {_state['error']}"
        return txt
    if _state["error"]:
        return f"Zusätzliche GPU-Laufzeit nicht nutzbar: {_state['error']}"
    return "Keine zusätzliche GPU-Laufzeit aktiv"


def download_size(offer: Offer, opener=net.urlopen) -> int:
    return sum(i["size"] for i in plan(offer, opener))
