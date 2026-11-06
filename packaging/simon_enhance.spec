# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec - builds a one-folder bundle in dist/SimonImageEnhance
# Usage (from the repository root):  pyinstaller packaging/simon_enhance.spec --noconfirm
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "packaging", "icons", "icon.ico" if sys.platform == "win32" else "icon-256.png")

# Qt modules the app never uses - keeps the bundle small
EXCLUDES = [
    "tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.Qt3DCore",
    "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtBluetooth", "PySide6.QtNetworkAuth", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtLocation",
    "PySide6.QtPositioning", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtWebSockets",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
    "matplotlib", "scipy", "pandas", "IPython", "pytest", "lxml", "PIL.ImageTk",
    "torch", "torchvision", "tensorflow", "onnx", "pnnx", "sympy", "einops",
]

# GPU acceleration (OpenCL) is optional at runtime but bundled: collect pyopencl's
# kernels sources and its bundled OpenCL ICD loader.
from PyInstaller.utils.hooks import collect_all
try:
    cl_datas, cl_binaries, cl_hidden = collect_all("pyopencl")
except Exception:
    cl_datas, cl_binaries, cl_hidden = [], [], []
# AI runtimes: ONNX Runtime (WebGPU on Linux / DirectML on Windows), OpenVINO (NPU), ncnn (Vulkan),
# wgpu (Vulkan compute). collect_all picks up their native libraries and resources.
for _pkg in ("onnxruntime", "openvino", "ncnn", "wgpu"):
    try:
        _d, _b, _h = collect_all(_pkg)
        cl_datas += _d
        cl_binaries += _b
        cl_hidden += _h
    except Exception:
        pass

# never ship a CPU OpenCL implementation that may be installed for tests
cl_binaries = [b for b in cl_binaries if "pocl" not in os.path.basename(b[0]).lower()]
cl_datas = [d for d in cl_datas if "pocl" not in os.path.basename(d[0]).lower()]

a = Analysis(
    [os.path.join(ROOT, "packaging", "launcher.py")],
    pathex=[ROOT],
    binaries=cl_binaries,
    datas=[(os.path.join(ROOT, "simon_enhance", "assets"), "simon_enhance/assets")] + cl_datas,
    hiddenimports=["rawpy", "rawpy._rawpy", "imageio.v3", "imageio.plugins.pillow", "tifffile",
                   "simon_enhance.ai.engine", "simon_enhance.ai.models", "simon_enhance.ai.features",
                   "simon_enhance.vulkan_compute"] + cl_hidden,
    excludes=EXCLUDES,
    noarchive=False,
)
# drop CPU OpenCL implementations (PoCL) that hooks may pick up from a test setup
a.binaries = [x for x in a.binaries if "pocl" not in x[0].lower()]
a.datas = [x for x in a.datas if "pocl" not in x[0].lower()]
# OpenVINO model importers the app never uses (it only loads ONNX)
_unused_ov = ("tensorflow_frontend", "tensorflow_lite_frontend", "pytorch_frontend", "paddle_frontend", "jax_frontend")
a.binaries = [x for x in a.binaries if not any(u in x[0] for u in _unused_ov)]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SimonImageEnhance",
    console=False,
    icon=ICON,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SimonImageEnhance")
