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
]

a = Analysis(
    [os.path.join(ROOT, "packaging", "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[(os.path.join(ROOT, "simon_enhance", "assets"), "simon_enhance/assets")],
    hiddenimports=["rawpy", "rawpy._rawpy", "imageio.v3", "imageio.plugins.pillow"],
    excludes=EXCLUDES,
    noarchive=False,
)
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
