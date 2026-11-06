"""Vulkan compute (via wgpu: Vulkan on Linux, Vulkan/DirectX 12 on Windows) for the
windowed filters of the background removal. Vulkan drivers ship with every GPU driver
(Mesa RADV/ANV, NVIDIA, AMD, Intel) and are also available inside Flatpak, unlike
OpenCL which often needs extra packages.

Environment: SIE_VULKAN=0 disables it, SIE_ALLOW_CPU_VULKAN=1 also accepts software
Vulkan (llvmpipe/lavapipe - only useful for tests).
"""

from __future__ import annotations

import os
import threading

import numpy as np

_SHADER = """
struct Params { h: u32, w: u32, r: i32, axis: u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);
    let h = i32(p.h);
    let w = i32(p.w);
    if (x >= w || y >= h) { return; }
    var s: f32 = 0.0;
    var n: i32 = 0;
    if (p.axis == 0u) {
        let a = max(0, y - p.r);
        let b = min(h - 1, y + p.r);
        for (var i = a; i <= b; i = i + 1) { s = s + src[u32(i * w + x)]; }
        n = b - a + 1;
    } else {
        let a = max(0, x - p.r);
        let b = min(w - 1, x + p.r);
        let row = y * w;
        for (var i = a; i <= b; i = i + 1) { s = s + src[u32(row + i)]; }
        n = b - a + 1;
    }
    dst[u32(y * w + x)] = s / f32(n);
}
"""

_lock = threading.RLock()
_st = {"tried": False, "device": None, "pipeline": None, "name": None, "error": None, "max_bytes": 0}


def _init() -> bool:
    if _st["tried"]:
        return _st["device"] is not None
    _st["tried"] = True
    if os.environ.get("SIE_VULKAN", "1") == "0":
        _st["error"] = "per SIE_VULKAN=0 abgeschaltet"
        return False
    try:
        import wgpu  # type: ignore
    except Exception as exc:
        _st["error"] = f"wgpu nicht verfügbar ({exc})"
        return False
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("kein Grafikadapter")
        info = adapter.info
        if info.get("adapter_type") == "CPU" and os.environ.get("SIE_ALLOW_CPU_VULKAN") != "1":
            raise RuntimeError("nur Software-Vulkan (CPU) gefunden")
        limits = adapter.limits
        want = min(int(limits["max-storage-buffer-binding-size"]), int(limits["max-buffer-size"]))
        device = adapter.request_device_sync(required_limits={
            "max-storage-buffer-binding-size": want, "max-buffer-size": want})
        module = device.create_shader_module(code=_SHADER)
        pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
        _st.update(device=device, pipeline=pipeline, max_bytes=want,
                   name=f"{info.get('device', '?')} ({info.get('backend_type', '?')})")
        x = np.random.default_rng(0).random((37, 53)).astype(np.float32)
        from .greenscreen import _box_axis
        if np.abs(_box(x, 3) - _box_axis(_box_axis(x, 3, 0), 3, 1)).max() > 1e-4:
            raise RuntimeError("Selbsttest fehlgeschlagen")
        return True
    except Exception as exc:
        _st.update(device=None, pipeline=None, error=str(exc))
        return False


def _pass(device, pipeline, src_buf, dst_buf, h, w, r, axis):
    import wgpu  # type: ignore
    params = np.array([h, w, r, axis], dtype=np.uint32)
    params[2] = np.int32(r).view(np.uint32)
    ubuf = device.create_buffer_with_data(data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": src_buf}},
        {"binding": 1, "resource": {"buffer": dst_buf}},
        {"binding": 2, "resource": {"buffer": ubuf}},
    ])
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bind)
    cp.dispatch_workgroups((w + 15) // 16, (h + 15) // 16, 1)
    cp.end()
    device.queue.submit([enc.finish()])


def _box(x: np.ndarray, r: int) -> np.ndarray:
    import wgpu  # type: ignore
    device, pipeline = _st["device"], _st["pipeline"]
    x = np.ascontiguousarray(x, dtype=np.float32)
    h, w = x.shape
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    a = device.create_buffer_with_data(data=x.tobytes(), usage=usage)
    b = device.create_buffer(size=x.nbytes, usage=usage)
    _pass(device, pipeline, a, b, h, w, r, 0)
    _pass(device, pipeline, b, a, h, w, r, 1)
    out = np.frombuffer(device.queue.read_buffer(a), dtype=np.float32).reshape(h, w).copy()
    a.destroy()
    b.destroy()
    return out


def available() -> bool:
    with _lock:
        return _init()


def name() -> str | None:
    with _lock:
        return _st["name"] if _init() else None


def error() -> str | None:
    return _st["error"]


def box(x: np.ndarray, r: int) -> np.ndarray | None:
    """Box filter on the GPU via Vulkan, or None (caller falls back)."""
    with _lock:
        if not _init() or x.nbytes > _st["max_bytes"]:
            return None
        try:
            return _box(x, r)
        except Exception as exc:
            _st["error"] = f"Vulkan-Fehler ({exc})"
            _st.update(device=None, pipeline=None)
            return None
