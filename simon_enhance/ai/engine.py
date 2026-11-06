"""Inference engine: finds accelerators and runs models on the best one.

Runtimes (all optional, the CPU always works):
  * ONNX Runtime
      - WebGPU EP    : any GPU through the installed Vulkan driver (Linux) or Direct3D 12
                       (Windows) - AMD, NVIDIA and Intel alike               (Linux build)
      - OpenVINO EP  : Intel NPU / GPU, if an onnxruntime-openvino build is used
      - DirectML EP  : any DirectX 12 GPU incl. NVIDIA Tensor cores, AMD, Intel,
                       Intel NPU                                         (Windows build)
      - CUDA / TensorRT EP : NVIDIA Tensor cores, if an onnxruntime-gpu build is used
  * OpenVINO (standalone): Intel NPU (Core Ultra "AI Boost") and Intel GPU     (Linux build)
  * ncnn + Vulkan  : any Vulkan GPU (NVIDIA, AMD, Intel). ncnn uses fp16 and, where the
                     driver offers VK_KHR/NV_cooperative_matrix, the Tensor / matrix cores.

All GPU / NPU paths use the drivers already installed on the system (Mesa, AMD, NVIDIA,
Intel NPU driver); nothing vendor specific (CUDA, ROCm) is bundled.

Every (model, device) combination is tried once; if a device cannot compile or run a
model (NPUs only support some operators, memory limits, ...) the next one is used.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from . import models as M

KIND_ORDER = {"dgpu": 0, "npu": 1, "igpu": 2, "cpu": 3}
KIND_LABEL = {"dgpu": "Grafikkarte", "npu": "NPU", "igpu": "integrierte GPU", "cpu": "CPU"}


@dataclass(frozen=True)
class Device:
    id: str        # e.g. "ort:openvino:NPU", "ort:dml:0", "vk:0", "ort:cpu"
    label: str
    kind: str      # dgpu | npu | igpu | cpu
    runtime: str   # "onnxruntime" | "ncnn"
    tensor_cores: bool = False


_lock = threading.RLock()
_devices: list[Device] | None = None
_sessions: dict = {}
_failed: set = set()
_last_used: dict = {}
_preferred = {"id": "auto"}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _ort():
    try:
        import onnxruntime as ort  # type: ignore
        ort.set_default_logger_severity(3)
        return ort
    except Exception:
        return None


def _ncnn():
    if os.environ.get("SIE_VULKAN", "1") == "0":
        return None
    try:
        import ncnn  # type: ignore
        return ncnn
    except Exception:
        return None


def _openvino():
    if os.environ.get("SIE_OPENVINO", "1") == "0":
        return None
    try:
        import openvino as ov  # type: ignore
        return ov
    except Exception:
        return None


def _vulkan_gpus() -> list[tuple[str, str]]:
    """(name, kind) of real Vulkan GPUs, used to label the WebGPU device."""
    gpus = []
    nc = _ncnn()
    if nc is None:
        return gpus
    try:
        for i in range(nc.get_gpu_count()):
            info = nc.get_gpu_info(i)
            typ = info.type()  # 0 discrete, 1 integrated, 2 virtual, 3 cpu
            if typ == 3 and os.environ.get("SIE_ALLOW_CPU_VULKAN") != "1":
                continue
            gpus.append((info.device_name().strip(), "dgpu" if typ == 0 else "igpu"))
    except Exception:
        pass
    return gpus


def _openvino_devices(ort) -> list[str]:
    try:
        from onnxruntime.capi import _pybind_state as C  # type: ignore
        return list(C.get_available_openvino_device_ids())
    except Exception:
        return ["CPU", "GPU", "NPU"]  # unknown: try them, failures are remembered


def discover(refresh: bool = False) -> list[Device]:
    global _devices
    with _lock:
        if _devices is not None and not refresh:
            return _devices
        devs: list[Device] = []
        ort = _ort()
        if ort is not None:
            prov = ort.get_available_providers()
            if "OpenVINOExecutionProvider" in prov:
                for d in _openvino_devices(ort):
                    base = d.split(".")[0].upper()
                    if base == "NPU":
                        devs.append(Device("ort:openvino:NPU", "Intel NPU (OpenVINO)", "npu", "onnxruntime"))
                    elif base == "GPU":
                        devs.append(Device(f"ort:openvino:{d}", f"Intel GPU {d} (OpenVINO)", "igpu", "onnxruntime"))
            if "DmlExecutionProvider" in prov:
                devs.append(Device("ort:dml:0", "GPU (DirectML)", "dgpu", "onnxruntime", tensor_cores=True))
                devs.append(Device("ort:dml:npu", "NPU (DirectML)", "npu", "onnxruntime"))
            if "TensorrtExecutionProvider" in prov:
                devs.append(Device("ort:tensorrt:0", "NVIDIA GPU (TensorRT)", "dgpu", "onnxruntime", tensor_cores=True))
            if "CUDAExecutionProvider" in prov:
                devs.append(Device("ort:cuda:0", "NVIDIA GPU (CUDA)", "dgpu", "onnxruntime", tensor_cores=True))
            if "WebGpuExecutionProvider" in prov:
                gpus = _vulkan_gpus()
                if gpus or os.name == "nt":
                    name, kind = gpus[0] if gpus else ("GPU", "dgpu")  # Dawn picks the fast GPU
                    api = "Direct3D 12" if os.name == "nt" else "Vulkan"
                    devs.append(Device("ort:webgpu:0", f"{name} (WebGPU/{api})", kind, "onnxruntime"))
        ov = _openvino()
        if ov is not None:
            try:
                ov_devs = ov.Core().available_devices
            except Exception:
                ov_devs = []
            for d in ov_devs:
                base = d.split(".")[0].upper()
                if base == "NPU":
                    devs.append(Device(f"ov:{d}", "Intel NPU (OpenVINO)", "npu", "openvino"))
                elif base == "GPU":
                    devs.append(Device(f"ov:{d}", f"Intel GPU {d} (OpenVINO)", "igpu", "openvino"))
                elif base == "CPU" and os.environ.get("SIE_OPENVINO_ALLOW_CPU") == "1":
                    devs.append(Device("ov:CPU", "CPU (OpenVINO)", "igpu", "openvino"))  # tests only
        nc = _ncnn()
        if nc is not None:
            try:
                for i in range(nc.get_gpu_count()):
                    info = nc.get_gpu_info(i)
                    typ = info.type()  # 0 discrete, 1 integrated, 2 virtual, 3 cpu
                    if typ == 3 and os.environ.get("SIE_ALLOW_CPU_VULKAN") != "1":
                        continue  # software Vulkan (llvmpipe) is slower than the CPU path
                    name = info.device_name().strip()
                    nvidia = info.vendor_id() == 0x10DE
                    kind = "dgpu" if typ == 0 else "igpu"
                    devs.append(Device(f"vk:{i}", f"{name} (Vulkan)", kind, "ncnn", tensor_cores=nvidia))
            except Exception:
                pass
        if ort is not None:
            devs.append(Device("ort:cpu", "CPU", "cpu", "onnxruntime"))
        rt_order = {"ncnn": 0, "onnxruntime": 1, "openvino": 2}
        devs.sort(key=lambda d: (KIND_ORDER[d.kind], 0 if d.id.startswith(("ort:cuda", "ort:tensorrt")) else 1,
                                 rt_order.get(d.runtime, 3)))
        _devices = devs
        return devs


def set_preferred(device_id: str) -> None:
    _preferred["id"] = device_id or "auto"


def preferred() -> str:
    return _preferred["id"]


def candidates(model_id: str) -> list[Device]:
    info = M.MODELS[model_id]
    devs = [d for d in discover() if d.runtime in ("onnxruntime", "openvino") or (d.runtime == "ncnn" and info.ncnn)]
    devs = [d for d in devs if (model_id, d.id) not in _failed]
    pref = _preferred["id"]
    if pref != "auto":
        chosen = [d for d in devs if d.id == pref]
        rest = [d for d in devs if d.id != pref and d.kind == "cpu"]
        return chosen + rest  # explicit choice, CPU as safety net
    return devs


def last_device(model_id: str) -> Device | None:
    return _last_used.get(model_id)


def describe(d: Device | None) -> str:
    if d is None:
        return "–"
    extra = ", Tensor-Kerne" if d.tensor_cores else ""
    return f"{d.label} [{KIND_LABEL[d.kind]}{extra}]"


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

class _OrtRunner:
    def __init__(self, model_id: str, device: Device, shape):
        ort = _ort()
        info = M.MODELS[model_id]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        parts = device.id.split(":")
        providers: list = []
        if parts[1] == "openvino":
            cache = os.path.join(M.models_dir(), "ov_cache")
            providers = [("OpenVINOExecutionProvider", {"device_type": parts[2], "cache_dir": cache}),
                         "CPUExecutionProvider"]
        elif parts[1] == "dml":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opt = {"device_filter": "npu"} if parts[2] == "npu" else {"device_id": int(parts[2])}
            providers = [("DmlExecutionProvider", opt), "CPUExecutionProvider"]
        elif parts[1] == "tensorrt":
            providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": True}), "CUDAExecutionProvider",
                         "CPUExecutionProvider"]
        elif parts[1] == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        elif parts[1] == "webgpu":
            providers = [("WebGpuExecutionProvider", {}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(M.onnx_path(model_id), sess_options=so, providers=providers)
        used = self.sess.get_providers()[0]
        if parts[1] != "cpu" and used == "CPUExecutionProvider":
            raise RuntimeError(f"{device.label}: Laufzeit nicht nutzbar")
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.shape = shape
        del info

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out], {self.inp: x})[0]


class _NcnnRunner:
    def __init__(self, model_id: str, device: Device, shape):
        nc = _ncnn()
        param, binf = M.ncnn_paths(model_id)
        self.net = nc.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.set_vulkan_device(int(device.id.split(":")[1]))
        if self.net.load_param(param) != 0 or self.net.load_model(binf) != 0:
            raise RuntimeError("ncnn-Modell konnte nicht geladen werden")
        self.nc = nc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        outs = []
        for b in range(x.shape[0]):
            ex = self.net.create_extractor()
            ex.input("in0", self.nc.Mat(np.ascontiguousarray(x[b])).clone())
            ret, out = ex.extract("out0")
            if ret != 0:
                raise RuntimeError(f"ncnn-Fehler {ret}")
            outs.append(np.array(out))
        return np.stack(outs).astype(np.float32)


class _OpenVinoRunner:
    """Standalone OpenVINO: compiles the ONNX model for the NPU / Intel GPU. NPUs need
    static shapes - every call uses one fixed tile / input size."""

    def __init__(self, model_id: str, device: Device, shape):
        ov = _openvino()
        core = ov.Core()
        core.set_property({"CACHE_DIR": os.path.join(M.models_dir(), "ov_cache")})
        model = core.read_model(M.onnx_path(model_id))
        model.reshape({model.inputs[0].get_any_name(): ov.PartialShape(list(shape))})
        self.compiled = core.compile_model(model, device.id.split(":", 1)[1])
        self.out = self.compiled.outputs[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.compiled([x])[self.out], dtype=np.float32)


def _runner(model_id: str, device: Device, shape):
    key = (model_id, device.id, shape)
    r = _sessions.get(key)
    if r is None:
        cls = {"ncnn": _NcnnRunner, "openvino": _OpenVinoRunner}.get(device.runtime, _OrtRunner)
        r = cls(model_id, device, shape)
        _sessions[key] = r
    return r


def run(model_id: str, x: np.ndarray, validate=None) -> np.ndarray:
    """Run a model on an NCHW float32 batch on the best working device."""
    if not M.is_available(model_id):
        raise FileNotFoundError(f"Modell {model_id} ist nicht installiert")
    errors = []
    with _lock:
        for dev in candidates(model_id):
            try:
                y = _runner(model_id, dev, x.shape)(x)
                if not np.all(np.isfinite(y)) or (validate is not None and not validate(y)):
                    raise RuntimeError("ungültiges Ergebnis")
                _last_used[model_id] = dev
                return y
            except Exception as exc:  # remember and fall back to the next device
                _failed.add((model_id, dev.id))
                _sessions.pop((model_id, dev.id, x.shape), None)
                errors.append(f"{dev.label}: {exc}")
    raise RuntimeError("Kein Gerät konnte das Modell ausführen:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# tiled inference (fixed tile size -> works on NPUs and keeps memory bounded)
# ---------------------------------------------------------------------------

def _window(n: int, overlap: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


def run_tiled(model_id: str, img: np.ndarray, progress=None, cancel=None) -> np.ndarray:
    """img: HxWx3 float32 sRGB in [0, 1]. Returns (H*s)x(W*s)x3."""
    info = M.MODELS[model_id]
    t, s = info.tile, info.scale
    ov = max(8, t // 8)
    step = t - 2 * ov
    h, w = img.shape[:2]
    pad = ov
    ph = max(t, int(np.ceil((h + 2 * pad - t) / step)) * step + t) if h + 2 * pad > t else t
    pw = max(t, int(np.ceil((w + 2 * pad - t) / step)) * step + t) if w + 2 * pad > t else t
    widths = ((pad, ph - h - pad), (pad, pw - w - pad), (0, 0))
    small = min(h, w) <= max(widths[0][1], widths[1][1], pad)
    src = np.pad(img, widths, mode="edge" if small else "reflect")
    out = np.zeros((ph * s, pw * s, 3), np.float32)
    wsum = np.zeros((ph * s, pw * s, 1), np.float32)
    win = _window(t * s, ov * s)
    win2 = (win[:, None] * win[None, :])[..., None]
    ys = list(range(0, ph - t + 1, step))
    xs = list(range(0, pw - t + 1, step))
    total, n = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            if cancel is not None and cancel():
                raise InterruptedError()
            tile = np.ascontiguousarray(src[y:y + t, x:x + t].transpose(2, 0, 1)[None])
            res = run(model_id, tile)[0].transpose(1, 2, 0)
            out[y * s:(y + t) * s, x * s:(x + t) * s] += res * win2
            wsum[y * s:(y + t) * s, x * s:(x + t) * s] += win2
            n += 1
            if progress is not None:
                progress(n / total)
    out /= np.maximum(wsum, 1e-6)
    return np.clip(out[pad * s:(pad + h) * s, pad * s:(pad + w) * s], 0.0, 1.0)


def reset() -> None:
    """Forget sessions and failures (e.g. after changing the accelerator)."""
    with _lock:
        _sessions.clear()
        _failed.clear()
