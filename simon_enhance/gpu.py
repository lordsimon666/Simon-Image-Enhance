"""Optional GPU acceleration via OpenCL (AMD / Intel via Mesa or vendor drivers,
NVIDIA drivers, Windows GPU drivers).

Only the expensive windowed filters of the background removal run on the GPU (they
dominate its run time). Everything works without a GPU: if pyopencl or an OpenCL
driver is missing, or anything fails, the NumPy implementation is used.

Environment variables
    SIE_GPU=0                 disable the GPU completely
    SIE_OPENCL_ALLOW_CPU=1    also accept CPU OpenCL devices (used by the tests)
    RUSTICL_ENABLE            Mesa OpenCL drivers to use (default here: radeonsi,iris)
"""

from __future__ import annotations

import os
import threading

import numpy as np

_KERNELS = r"""
// windowed mean along one axis, window 2r+1, shortened + renormalised at borders
__kernel void box_rows(__global const float *src, __global float *dst,
                       const int h, const int w, const int r) {
    const int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    const int y0 = max(0, y - r), y1 = min(h - 1, y + r);
    float s = 0.0f;
    for (int i = y0; i <= y1; ++i) s += src[i * w + x];
    dst[y * w + x] = s / (float)(y1 - y0 + 1);
}
__kernel void box_cols(__global const float *src, __global float *dst,
                       const int h, const int w, const int r) {
    const int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    const int x0 = max(0, x - r), x1 = min(w - 1, x + r);
    const int row = y * w;
    float s = 0.0f;
    for (int i = x0; i <= x1; ++i) s += src[row + i];
    dst[row + x] = s / (float)(x1 - x0 + 1);
}
"""

# below this many pixels the transfer overhead outweighs the gain
MIN_PIXELS = 250_000

# Mesa's OpenCL driver (Rusticl - Fedora's default for AMD and Intel GPUs) exposes no
# devices unless its drivers are enabled explicitly. Enable the mature ones unless the
# user configured something else. Must happen before the OpenCL ICD is loaded.
if os.name != "nt":
    os.environ.setdefault("RUSTICL_ENABLE", "radeonsi,iris")

_lock = threading.RLock()
_state = {"tried": False, "ctx": None, "queue": None, "prog": None, "device": None,
          "enabled": True, "error": None}


def _init():
    if _state["tried"]:
        return _state["ctx"] is not None
    _state["tried"] = True
    if os.environ.get("SIE_GPU", "1") == "0":
        _state["error"] = "per SIE_GPU=0 abgeschaltet"
        return False
    try:
        import pyopencl as cl  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the system
        _state["error"] = f"pyopencl nicht verfügbar ({exc})"
        return False
    allow_cpu = os.environ.get("SIE_OPENCL_ALLOW_CPU") == "1"
    candidates = []
    try:
        for platform in cl.get_platforms():
            for dev in platform.get_devices():
                is_gpu = bool(dev.type & cl.device_type.GPU)
                if is_gpu or allow_cpu:
                    # prefer discrete / bigger GPUs: more compute units, more memory
                    score = (is_gpu, dev.max_compute_units * dev.max_clock_frequency, dev.global_mem_size)
                    candidates.append((score, dev))
    except Exception as exc:  # no ICD / driver
        _state["error"] = f"kein OpenCL-Treiber gefunden ({exc})"
        return False
    if not candidates:
        _state["error"] = "keine OpenCL-fähige Grafikkarte gefunden"
        return False
    candidates.sort(key=lambda c: c[0], reverse=True)
    for _, dev in candidates:
        try:
            ctx = cl.Context([dev])
            queue = cl.CommandQueue(ctx)
            prog = cl.Program(ctx, _KERNELS).build()
            prog = {"rows": cl.Kernel(prog, "box_rows"), "cols": cl.Kernel(prog, "box_cols")}
            # self test against NumPy
            x = np.random.default_rng(0).random((37, 53)).astype(np.float32)
            _state.update(ctx=ctx, queue=queue, prog=prog, device=dev)
            if np.abs(_box_gpu(x, 3) - _box_reference(x, 3)).max() > 1e-4:
                raise RuntimeError("Selbsttest fehlgeschlagen")
            _state["error"] = None
            return True
        except Exception as exc:
            _state.update(ctx=None, queue=None, prog=None, device=None,
                          error=f"{dev.name.strip()}: {exc}")
    return False


def _box_reference(x, r):
    from .greenscreen import _box_axis
    return _box_axis(_box_axis(x, r, 0), r, 1)


def _box_gpu(x: np.ndarray, r: int) -> np.ndarray:
    import pyopencl as cl  # type: ignore
    ctx, queue, prog = _state["ctx"], _state["queue"], _state["prog"]
    x = np.ascontiguousarray(x, dtype=np.float32)
    h, w = x.shape
    mf = cl.mem_flags
    a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    b = cl.Buffer(ctx, mf.READ_WRITE, x.nbytes)
    c = cl.Buffer(ctx, mf.WRITE_ONLY, x.nbytes)
    gsize = (w, h)
    args = (np.int32(h), np.int32(w), np.int32(r))
    prog["rows"](queue, gsize, None, a, b, *args)
    prog["cols"](queue, gsize, None, b, c, *args)
    out = np.empty_like(x)
    cl.enqueue_copy(queue, out, c)
    queue.finish()
    return out


def set_enabled(on: bool) -> None:
    _state["enabled"] = bool(on)


def is_enabled() -> bool:
    return _state["enabled"]


def available() -> bool:
    with _lock:
        return _init()


def device_name() -> str | None:
    with _lock:
        if not _init():
            return None
        dev = _state["device"]
        mem = dev.global_mem_size / 2 ** 30
        return f"{dev.name.strip()} ({dev.platform.name.strip()}, {mem:.1f} GB)"


def status() -> str:
    """Human readable status for the UI."""
    if not is_enabled():
        return "GPU-Beschleunigung ausgeschaltet"
    name = device_name()
    if name:
        return f"GPU: {name}"
    return f"Keine GPU-Beschleunigung: {_state['error']}"


def box(x: np.ndarray, r: int) -> np.ndarray | None:
    """GPU box filter, or None if the CPU should do it (small input / no GPU / error)."""
    if not _state["enabled"] or x.size < MIN_PIXELS:
        return None
    with _lock:
        if not _init():
            return None
        try:
            return _box_gpu(x, r)
        except Exception as exc:  # out of GPU memory, driver reset, ...
            _state["error"] = f"GPU-Fehler, nutze CPU ({exc})"
            _state["enabled"] = False
            return None
