"""Render icon.svg to PNG/ICO files (run once when the icon changes)."""
import os
import sys

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
app = QGuiApplication(sys.argv)
r = QSvgRenderer(os.path.join(here, "icon.svg"))
for size in (16, 32, 48, 64, 128, 256, 512):
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p)
    p.end()
    img.save(os.path.join(here, f"icon-{size}.png"))
big = Image.open(os.path.join(here, "icon-256.png"))
big.save(os.path.join(here, "icon.ico"), sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
Image.open(os.path.join(here, "icon-256.png")).save(os.path.join(root, "simon_enhance", "assets", "icon.png"))
print("ok")
