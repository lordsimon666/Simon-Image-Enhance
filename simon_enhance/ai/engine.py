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

import json
import os
import subprocess
import sys
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
_reasons: dict = {}          # (model, device id) -> why the device was not used
_probed: set = set()         # (model, device id) checked in this process
activity_hook = None         # callable(str): "preparing the GPU ..." messages for the UI
_IN_WORKER = False           # True inside a GPU worker / probe process
_workers: dict = {}          # device id -> (process, connection)
_preferred = {"id": "auto"}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _ort():
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        # a downloaded CUDA / ROCm runtime that cannot be loaded must never take the
        # bundled ONNX Runtime down with it
        from .. import runtimes
        if runtimes._state.get("active"):
            runtimes.deactivate_after_failure(exc)
            try:
                import onnxruntime as ort  # type: ignore
            except Exception:
                return None
        else:
            return None
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    if _IN_WORKER:
        # ROCm 10: MIGraphX is an execution-provider plugin. It is only loaded in the worker
        # processes - HIP/HSA in the app process broke Vulkan and crashed it on some systems
        from .. import runtimes
        runtimes.register_plugins(ort)
    return ort


_STRICT = {}


def _strict_session_class(ort):
    """InferenceSession that raises instead of silently falling back to the CPU when the
    GPU execution provider cannot be created (ONNX Runtime only prints that error)."""
    cls = _STRICT.get(id(ort))
    if cls is None:
        class StrictSession(ort.InferenceSession):
            def _create_inference_session(self, *args, **kwargs):
                self._enable_fallback = False
                return super()._create_inference_session(*args, **kwargs)
        cls = _STRICT[id(ort)] = StrictSession
    return cls


def _explain_ep_error(device, exc) -> str:
    text = str(exc)
    if "No devices detected that match the filter" in text:
        what = "keine NPU" if device.kind == "npu" else "keine passende DirectX-12-Grafikkarte"
        return f"{what} gefunden (Grafiktreiber aktuell?)"
    if "LoadLibrary failed" in text or "cannot open shared object" in text:
        return f"Bibliothek fehlt: {text.splitlines()[0][:200]}"
    first = text.strip().splitlines()[0] if text.strip() else type(exc).__name__
    return first[:300]


def _plugin_devices(ort, ep_name: str) -> list:
    """EP devices of a registered execution-provider plugin (ONNX Runtime >= 1.22)."""
    try:
        return [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
    except Exception:  # noqa: BLE001 - older ONNX Runtime: providers are chosen by name
        return []


def _amd_label() -> str:
    from .. import runtimes
    try:
        gfx = (runtimes.rocm_driver() or {}).get("gfx") or []
    except Exception:  # noqa: BLE001
        gfx = []
    name = next((runtimes.GFX_NAMES[g] for g in gfx if g in runtimes.GFX_NAMES), "AMD GPU")
    return f"{name} (ROCm 10 · MIGraphX)"


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
            from .. import runtimes
            gpu_name = None
            if "CUDAExecutionProvider" in prov:
                info = runtimes.cuda_driver()
                gpu_name = (info or {}).get("gpus", [None])[0] if info and info.get("gpus") else None
            if "TensorrtExecutionProvider" in prov and runtimes._find_lib("libnvinfer.so.10"):
                devs.append(Device("ort:tensorrt:0", f"{gpu_name or 'NVIDIA GPU'} (TensorRT)", "dgpu",
                                   "onnxruntime", tensor_cores=True))
            if "CUDAExecutionProvider" in prov:
                devs.append(Device("ort:cuda:0", f"{gpu_name or 'NVIDIA GPU'} (CUDA)", "dgpu", "onnxruntime",
                                   tensor_cores=True))
            if ("MIGraphXExecutionProvider" in prov or _plugin_devices(ort, "MIGraphXExecutionProvider")
                    or (not _IN_WORKER and runtimes.plugin_ready())):
                devs.append(Device("ort:migraphx:0", _amd_label(), "dgpu", "onnxruntime"))
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
        vendor_rt = ("ort:tensorrt", "ort:cuda", "ort:migraphx")  # vendor-optimised first
        devs.sort(key=lambda d: (KIND_ORDER[d.kind], 0 if d.id.startswith(vendor_rt) else 1,
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
        elif parts[1] == "migraphx":
            providers = [("MIGraphXExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            plugin_devs = _plugin_devices(ort, "MIGraphXExecutionProvider")
            if plugin_devs:  # ROCm 10 plugin: selected through its EP device
                so.add_provider_for_devices(plugin_devs[:1], {})
                providers = None
        elif parts[1] == "webgpu":
            providers = [("WebGpuExecutionProvider", {}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        cls = ort.InferenceSession if parts[1] == "cpu" else _strict_session_class(ort)
        try:
            self.sess = cls(M.onnx_path(model_id), sess_options=so, providers=providers)
        except Exception as exc:  # noqa: BLE001 - give the real reason, not "not usable"
            raise RuntimeError(_explain_ep_error(device, exc)) from None
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
            if not _probe_ok(model_id, dev):
                errors.append(f"{dev.label}: {_reasons.get((model_id, dev.id), 'Prüfung fehlgeschlagen')}")
                continue
            try:
                y = _remote(model_id, dev, x) if _isolated(dev) else _runner(model_id, dev, x.shape)(x)
                if not np.all(np.isfinite(y)) or (validate is not None and not validate(y)):
                    raise RuntimeError("ungültiges Ergebnis")
                _last_used[model_id] = dev
                return y
            except Exception as exc:  # remember and fall back to the next device
                _failed.add((model_id, dev.id))
                _reasons[(model_id, dev.id)] = _short(exc)
                _sessions.pop((model_id, dev.id, x.shape), None)
                errors.append(f"{dev.label}: {exc}")
    raise RuntimeError("Kein Gerät konnte das Modell ausführen:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# GPU worker processes: every GPU runs the models in its own process. A crashing
# driver (segfault in ROCm, Vulkan, DirectML ...) then only ends that process and the
# app continues on the next device.
# ---------------------------------------------------------------------------

WORKER_TIMEOUT = 900     # first call may compile the model (MIGraphX: minutes)


def _isolated(dev: Device) -> bool:
    return dev.kind != "cpu" and not _IN_WORKER and os.environ.get("SIE_AI_ISOLATE", "1") != "0"


def _worker_main(conn, dev_id: str) -> None:
    global _IN_WORKER
    _IN_WORKER = True
    from .. import runtimes
    runtimes.activate()
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "stop":
            return
        _cmd, model_id, x = msg
        if os.environ.get("SIE_WORKER_TEST_CRASH") == dev_id:
            os.abort()  # tests: driver crash while computing
        try:
            dev = next((d for d in discover() if d.id == dev_id), None)
            if dev is None:
                raise RuntimeError("Gerät im Hilfsprozess nicht gefunden")
            conn.send(("ok", _runner(model_id, dev, x.shape)(x)))
        except Exception as exc:  # noqa: BLE001
            conn.send(("err", _short(exc)))


def _start_worker(dev_id: str):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_worker_main, args=(child, dev_id), daemon=True, name=f"sie-gpu-{dev_id}")
    proc.start()
    child.close()
    return proc, parent


def _stop_worker(dev_id: str, kill: bool = False) -> None:
    w = _workers.pop(dev_id, None)
    if w is None:
        return
    proc, conn = w
    try:
        if kill:
            proc.kill()
        else:
            conn.send(("stop",))
        proc.join(5)
        if proc.is_alive():
            proc.kill()
    except Exception:  # noqa: BLE001
        pass


def shutdown_workers() -> None:
    for dev_id in list(_workers):
        _stop_worker(dev_id)


def _remote(model_id: str, dev: Device, x: np.ndarray) -> np.ndarray:
    w = _workers.get(dev.id)
    if w is None or not w[0].is_alive():
        w = _workers[dev.id] = _start_worker(dev.id)
    proc, conn = w
    first = (model_id, dev.id) not in _remote_seen
    try:
        conn.send(("run", model_id, np.ascontiguousarray(x, np.float32)))
        if not conn.poll(WORKER_TIMEOUT if first else WORKER_TIMEOUT // 3):
            _stop_worker(dev.id, kill=True)
            raise RuntimeError("Grafikkarte antwortet nicht (Zeitüberschreitung)")
        status, payload = conn.recv()
    except (EOFError, OSError, BrokenPipeError):
        proc.join(3)
        code = proc.exitcode
        _workers.pop(dev.id, None)
        sig = f"Signal {-code}" if code is not None and code < 0 else f"Code {code}"
        raise RuntimeError(f"Treiber ist abgestürzt ({sig}) – Gerät wird für dieses Modell nicht verwendet")
    if status != "ok":
        raise RuntimeError(payload)
    _remote_seen.add((model_id, dev.id))
    return payload


_remote_seen: set = set()
import atexit as _atexit  # noqa: E402
_atexit.register(shutdown_workers)


def _short(exc) -> str:
    text = str(exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:300]


def fallback_reasons(model_id: str) -> list[tuple[str, str]]:
    """[(device label, reason)] of devices that were skipped for this model."""
    labels = {d.id: d.label for d in discover()}
    return [(labels.get(dev, dev), r) for (m, dev), r in _reasons.items() if m == model_id]


# ---------------------------------------------------------------------------
# crash-safe device check: GPU drivers can crash the whole process (segfault in the
# Vulkan / ROCm / DirectML driver). Before a GPU is used for a model the first time,
# one tile is computed in a separate process and compared with the CPU result.
# ---------------------------------------------------------------------------

PROBE_TIMEOUT = 900   # MIGraphX compiles the model on first use - can take minutes


def probe_shape(model_id: str) -> tuple:
    info = M.MODELS[model_id]
    h, w = info.fixed if info.fixed else (info.tile, info.tile)
    return (1, 3, h, w)


def probe_input(shape) -> np.ndarray:
    """Deterministic photo-like test tile (gradients, edges, a little noise)."""
    _n, _c, h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([0.5 + 0.35 * np.sin(x / max(w, 1) * 7), 0.3 + 0.5 * y / max(h, 1),
                    0.4 + 0.3 * np.cos((x + y) / max(h + w, 1) * 9)])
    img[:, h // 4:h // 2, w // 4:w // 2] = np.array([0.9, 0.8, 0.2], np.float32)[:, None, None]
    img += np.random.default_rng(7).normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)[None].astype(np.float32)


def compare(y: np.ndarray, ref: np.ndarray) -> float:
    """Relative difference of a device result to the CPU reference."""
    if model_out_bad(y) or y.shape != ref.shape:
        return float("inf")
    return float(np.abs(y - ref).mean() / (np.abs(ref).mean() + 1e-3))


def model_out_bad(y) -> bool:
    return not isinstance(y, np.ndarray) or not np.all(np.isfinite(y))


def _probe_file() -> str:
    from ..store import data_dir
    return os.path.join(data_dir(), "ai_probe.json")


def _probe_key(model_id: str, dev: Device) -> str:
    from .. import __version__, runtimes
    return "|".join([__version__, str(runtimes._state.get("active")), model_id, dev.id, dev.label])


def _probe_cache() -> dict:
    try:
        return json.load(open(_probe_file(), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _probe_command(model_id: str, dev_id: str, result: str) -> list:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--ai-probe", model_id, dev_id, result]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
            "from simon_enhance import runtimes; runtimes.activate(); from simon_enhance.ai import engine; "
            "raise SystemExit(engine.probe_main(sys.argv[2:]))", root, model_id, dev_id, result]


def _probe_ok(model_id: str, dev: Device) -> bool:
    if dev.kind == "cpu" or (model_id, dev.id) in _probed or os.environ.get("SIE_AI_PROBE", "1") == "0":
        return True
    _probed.add((model_id, dev.id))
    key = _probe_key(model_id, dev)
    cache = _probe_cache()
    res = cache.get(key)
    if res is None:
        if activity_hook is not None:
            try:
                activity_hook(f"{dev.label} wird für „{M.MODELS[model_id].title}“ geprüft (einmalig) …")
            except Exception:  # noqa: BLE001
                pass
        res = _run_probe(model_id, dev)
        cache = _probe_cache()
        cache[key] = res
        try:
            with open(_probe_file(), "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=1, ensure_ascii=False)
        except OSError:
            pass
    if not res.get("ok"):
        _failed.add((model_id, dev.id))
        _reasons[(model_id, dev.id)] = res.get("reason", "Prüfung fehlgeschlagen")
        return False
    return True


def _run_probe(model_id: str, dev: Device) -> dict:
    import tempfile
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    fd, result = tempfile.mkstemp(prefix="sie-probe-", suffix=".json")
    os.close(fd)
    try:
        try:
            p = subprocess.run(_probe_command(model_id, dev.id, result), capture_output=True, text=True,
                               timeout=PROBE_TIMEOUT, creationflags=flags, encoding="utf-8", errors="replace",
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": f"keine Antwort nach {PROBE_TIMEOUT // 60} Minuten"}
        except OSError as exc:
            return {"ok": True, "reason": f"Prüfung nicht möglich ({exc})"}  # cannot check: just try it
        try:
            with open(result, encoding="utf-8") as f:
                data = f.read().strip()
            if data:
                return json.loads(data)
        except (OSError, ValueError):
            pass
    finally:
        try:
            os.remove(result)
        except OSError:
            pass
    if p.returncode < 0 or p.returncode > 128 or os.name == "nt" and p.returncode not in (0, 1):
        code = -p.returncode if p.returncode < 0 else p.returncode
        return {"ok": False, "reason": f"Treiber ist abgestürzt (Code {code}) – Gerät wird für dieses Modell "
                                       "nicht verwendet"}
    tail = ((p.stderr or "").strip().splitlines() or ["unbekannter Fehler"])[-1]
    return {"ok": False, "reason": tail[:300]}


def probe_main(argv) -> int:
    """Child process: run one tile on the device and on the CPU, print the verdict."""
    model_id, dev_id, result = argv[0], argv[1], argv[2]

    def out(ok, reason="", **kw):
        with open(result, "w", encoding="utf-8") as f:
            json.dump(dict(ok=ok, reason=reason, **kw), f, ensure_ascii=False)
        return 0

    global _IN_WORKER
    _IN_WORKER = True
    if os.environ.get("SIE_PROBE_TEST_CRASH") == dev_id:
        os.abort()  # tests: simulate a driver crash
    dev = next((d for d in discover() if d.id == dev_id), None)
    if dev is None:
        return out(False, "Gerät nicht gefunden")
    if not M.is_available(model_id):
        return out(True, "Modell fehlt – ungeprüft")
    import time
    x = probe_input(probe_shape(model_id))
    try:
        t = time.time()
        y = _runner(model_id, dev, x.shape)(x)
        first = time.time() - t
    except Exception as exc:  # noqa: BLE001
        return out(False, _short(exc))
    cpu = next(d for d in discover() if d.kind == "cpu")
    ref = _runner(model_id, cpu, x.shape)(x)
    diff = compare(y, ref)
    if diff > 0.05:
        return out(False, f"falsches Ergebnis (Abweichung zur CPU {diff:.0%})", diff=diff)
    return out(True, "", diff=diff, seconds=round(first, 2))


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
        _reasons.clear()
        _probed.clear()
        _remote_seen.clear()
        shutdown_workers()
