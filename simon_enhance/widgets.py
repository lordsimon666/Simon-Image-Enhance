"""Custom Qt widgets: slider rows, curve editor, histogram, image view."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
    QLabel, QSlider, QSizePolicy, QWidget,
)

from .processing import eval_curve, identity_curve


# ---------------------------------------------------------------------------
# Slider row
# ---------------------------------------------------------------------------

class SliderRow(QWidget):
    """Label + slider + spin box. Double-click the label to reset."""

    valueChanged = Signal(float)
    editingFinished = Signal()

    STEPS = 1000

    def __init__(self, label: str, vmin: float, vmax: float, default: float = 0.0,
                 decimals: int = 0, log_scale: bool = False, gradient: tuple | None = None,
                 parent=None):
        super().__init__(parent)
        self.vmin, self.vmax, self.default = vmin, vmax, default
        self.log_scale = log_scale
        self._block = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.label = QLabel(label)
        self.label.setMinimumWidth(78)
        self.label.setToolTip("Doppelklick: zurücksetzen")
        self.label.mouseDoubleClickEvent = lambda e: self.reset()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if gradient:
            a, b = gradient
            self.slider.setStyleSheet(
                "QSlider::groove:horizontal{height:4px;border-radius:2px;"
                f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {a},stop:1 {b});}}"
                "QSlider::handle:horizontal{background:#ddd;width:12px;margin:-5px 0;border-radius:6px;}")
        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(10 ** -decimals if decimals else (50 if log_scale else 1))
        self.spin.setFixedWidth(72)
        # (buttons are hidden via the style sheet; NoButtons + style sheets breaks layout)
        self.spin.setAlignment(Qt.AlignRight)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self.editingFinished.emit)
        self.spin.valueChanged.connect(self._on_spin)
        self.spin.editingFinished.connect(self.editingFinished.emit)
        self.setValue(default)

    def _to_pos(self, v: float) -> int:
        if self.log_scale:
            a, b = math.log(self.vmin), math.log(self.vmax)
            f = (math.log(max(v, self.vmin)) - a) / (b - a)
        else:
            f = (v - self.vmin) / (self.vmax - self.vmin)
        return int(round(f * self.STEPS))

    def _from_pos(self, pos: int) -> float:
        f = pos / self.STEPS
        if self.log_scale:
            a, b = math.log(self.vmin), math.log(self.vmax)
            v = math.exp(a + f * (b - a))
            return round(v / 10) * 10
        v = self.vmin + f * (self.vmax - self.vmin)
        return round(v, self.spin.decimals())

    def _on_slider(self, pos):
        if self._block:
            return
        v = self._from_pos(pos)
        self._block = True
        self.spin.setValue(v)
        self._block = False
        self.valueChanged.emit(v)

    def _on_spin(self, v):
        if self._block:
            return
        self._block = True
        self.slider.setValue(self._to_pos(v))
        self._block = False
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float, emit: bool = False):
        self._block = True
        self.spin.setValue(v)
        self.slider.setValue(self._to_pos(v))
        self._block = False
        if emit:
            self.valueChanged.emit(self.value())

    def reset(self):
        self.setValue(self.default, emit=True)
        self.editingFinished.emit()


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

class Histogram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.hist = None

    def set_image(self, img8: np.ndarray | None):
        if img8 is None:
            self.hist = None
        else:
            sub = img8[::2, ::2].reshape(-1, 3)
            h = np.stack([np.bincount(sub[:, c], minlength=256) for c in range(3)]).astype(np.float64)
            h = np.sqrt(h)
            m = np.percentile(h[:, 2:254], 99.5) if h[:, 2:254].size else h.max()
            self.hist = np.clip(h / max(m, 1e-9), 0, 1)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.fillRect(r, QColor(24, 24, 24))
        p.setPen(QPen(QColor(55, 55, 55), 1))
        for i in range(1, 4):
            x = r.left() + r.width() * i / 4
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        if self.hist is not None:
            p.setCompositionMode(QPainter.CompositionMode_Plus)
            colors = [QColor(200, 40, 40, 200), QColor(40, 180, 40, 200), QColor(50, 90, 230, 200)]
            for c in range(3):
                path = QPainterPath(QPointF(r.left(), r.bottom()))
                for i, v in enumerate(self.hist[c]):
                    path.lineTo(r.left() + r.width() * i / 255.0, r.bottom() - v * (r.height() - 4))
                path.lineTo(r.right(), r.bottom())
                path.closeSubpath()
                p.fillPath(path, colors[c])
        p.end()


# ---------------------------------------------------------------------------
# Curve editor
# ---------------------------------------------------------------------------

CURVE_COLORS = {"rgb": QColor(230, 230, 230), "r": QColor(235, 80, 80),
                "g": QColor(90, 210, 90), "b": QColor(90, 140, 255)}


class CurveEditor(QWidget):
    """Interactive tone curve. Click = add point, drag = move,
    double-click / right-click = remove point."""

    changed = Signal()
    editingFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMouseTracking(True)
        self.curves = {c: identity_curve() for c in CURVE_COLORS}
        self.channel = "rgb"
        self.drag = None
        self.hover = None
        self.hist = None

    def heightForWidth(self, w):
        return w

    def hasHeightForWidth(self):
        return True

    def set_curves(self, curves: dict):
        self.curves = {c: [list(p) for p in curves.get(c, identity_curve())] for c in CURVE_COLORS}
        self.update()

    def set_channel(self, ch: str):
        self.channel = ch
        self.update()

    def set_histogram(self, img8: np.ndarray | None):
        if img8 is None:
            self.hist = None
        else:
            lum = (img8[::3, ::3].reshape(-1, 3) @ np.array([54, 183, 19])) >> 8
            h = np.sqrt(np.bincount(lum.astype(np.int64), minlength=256)[:256].astype(np.float64))
            self.hist = h / max(h.max(), 1e-9)
        self.update()

    def reset_channel(self):
        self.curves[self.channel] = identity_curve()
        self.update()
        self.changed.emit()
        self.editingFinished.emit()

    # geometry
    def _area(self) -> QRectF:
        s = min(self.width(), self.height()) - 16
        return QRectF((self.width() - s) / 2, 8, s, s)

    def _to_screen(self, x, y) -> QPointF:
        a = self._area()
        return QPointF(a.left() + x * a.width(), a.bottom() - y * a.height())

    def _from_screen(self, pt) -> tuple[float, float]:
        a = self._area()
        return ((pt.x() - a.left()) / a.width(), (a.bottom() - pt.y()) / a.height())

    def _hit(self, pos):
        pts = self.curves[self.channel]
        for i, (x, y) in enumerate(pts):
            sp = self._to_screen(x, y)
            if abs(sp.x() - pos.x()) < 8 and abs(sp.y() - pos.y()) < 8:
                return i
        return None

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._area()
        p.fillRect(a, QColor(28, 28, 28))
        if self.hist is not None:
            path = QPainterPath(QPointF(a.left(), a.bottom()))
            for i, v in enumerate(self.hist):
                path.lineTo(a.left() + a.width() * i / 255, a.bottom() - v * a.height() * 0.9)
            path.lineTo(a.right(), a.bottom())
            p.fillPath(path, QColor(70, 70, 70, 140))
        p.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(1, 4):
            p.drawLine(QPointF(a.left() + a.width() * i / 4, a.top()), QPointF(a.left() + a.width() * i / 4, a.bottom()))
            p.drawLine(QPointF(a.left(), a.top() + a.height() * i / 4), QPointF(a.right(), a.top() + a.height() * i / 4))
        p.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        p.drawLine(self._to_screen(0, 0), self._to_screen(1, 1))
        p.setPen(QPen(QColor(90, 90, 90), 1))
        p.drawRect(a)

        # other channels faintly
        t = np.linspace(0, 1, 200)
        for ch, col in CURVE_COLORS.items():
            pts = self.curves[ch]
            if ch == self.channel or (len(pts) == 2 and pts == identity_curve()):
                continue
            c = QColor(col)
            c.setAlpha(90)
            self._draw_curve(p, pts, t, QPen(c, 1))

        col = CURVE_COLORS[self.channel]
        self._draw_curve(p, self.curves[self.channel], t, QPen(col, 2))
        for i, (x, y) in enumerate(self.curves[self.channel]):
            sp = self._to_screen(x, y)
            p.setPen(QPen(col, 1.5))
            p.setBrush(QBrush(col if i == self.hover or (self.drag == i) else QColor(30, 30, 30)))
            p.drawEllipse(sp, 5, 5)
        p.end()

    def _draw_curve(self, p, pts, t, pen):
        y = eval_curve(pts, t)
        path = QPainterPath(self._to_screen(t[0], y[0]))
        for xi, yi in zip(t[1:], y[1:]):
            path.lineTo(self._to_screen(xi, yi))
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def mousePressEvent(self, e):
        pos = e.position()
        i = self._hit(pos)
        pts = self.curves[self.channel]
        if e.button() == Qt.RightButton:
            if i is not None and 0 < i < len(pts) - 1:
                pts.pop(i)
                self.changed.emit()
                self.editingFinished.emit()
                self.update()
            return
        if e.button() != Qt.LeftButton:
            return
        if i is None:
            x, y = self._from_screen(pos)
            if not (0 <= x <= 1):
                return
            x = min(max(x, 0.0), 1.0)
            # insert keeping order
            k = next((j for j, pt in enumerate(pts) if pt[0] > x), len(pts))
            if k == 0 or k == len(pts):
                return
            if x - pts[k - 1][0] < 0.02 or pts[k][0] - x < 0.02:
                return
            y = float(eval_curve(pts, np.array([x]))[0])
            pts.insert(k, [x, y])
            i = k
            self.changed.emit()
        self.drag = i
        self.update()

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.drag is None:
            h = self._hit(pos)
            if h != self.hover:
                self.hover = h
                self.setCursor(QCursor(Qt.PointingHandCursor if h is not None else Qt.CrossCursor))
                self.update()
            return
        pts = self.curves[self.channel]
        i = self.drag
        x, y = self._from_screen(pos)
        lo = pts[i - 1][0] + 0.01 if i > 0 else 0.0
        hi = pts[i + 1][0] - 0.01 if i < len(pts) - 1 else 1.0
        pts[i] = [round(min(max(x, lo), hi), 4), round(min(max(y, 0.0), 1.0), 4)]
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, e):
        if self.drag is not None:
            self.drag = None
            self.editingFinished.emit()
            self.update()

    def mouseDoubleClickEvent(self, e):
        i = self._hit(e.position())
        pts = self.curves[self.channel]
        if i is not None and 0 < i < len(pts) - 1:
            pts.pop(i)
            self.drag = None
            self.changed.emit()
            self.editingFinished.emit()
            self.update()


# ---------------------------------------------------------------------------
# Image view
# ---------------------------------------------------------------------------

def array_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return img.copy()  # detach from numpy buffer


class ImageView(QGraphicsView):
    pixelClicked = Signal(float, float)  # normalised image coords 0..1
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.item)
        self.setBackgroundBrush(QColor(38, 38, 38))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setAcceptDrops(True)
        self.fit = True
        self.picking = False
        self.overlay_text = ""

    def set_image(self, img8: np.ndarray | None, refit: bool = False):
        if img8 is None:
            self.item.setPixmap(QPixmap())
        else:
            self.item.setPixmap(QPixmap.fromImage(array_to_qimage(img8)))
        self.scene().setSceneRect(self.item.boundingRect())
        if refit or self.fit:
            self.fit_view()

    def fit_view(self):
        self.fit = True
        if not self.item.pixmap().isNull():
            self.fitInView(self.item, Qt.KeepAspectRatio)

    def zoom_100(self):
        self.fit = False
        self.resetTransform()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.fit:
            self.fit_view()

    def wheelEvent(self, e):
        if self.item.pixmap().isNull():
            return
        factor = 1.0015 ** e.angleDelta().y()
        self.fit = False
        self.scale(factor, factor)

    def set_picking(self, on: bool):
        self.picking = on
        self.setDragMode(QGraphicsView.NoDrag if on else QGraphicsView.ScrollHandDrag)
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.OpenHandCursor)

    def mousePressEvent(self, e):
        if self.picking and e.button() == Qt.LeftButton and not self.item.pixmap().isNull():
            pt = self.mapToScene(e.position().toPoint())
            pm = self.item.pixmap()
            self.pixelClicked.emit(pt.x() / pm.width(), pt.y() / pm.height())
            return
        super().mousePressEvent(e)

    def drawForeground(self, painter, rect):
        if self.overlay_text:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor(240, 240, 240))
            f = painter.font()
            f.setPointSize(11)
            f.setBold(True)
            painter.setFont(f)
            r = painter.fontMetrics().boundingRect(self.overlay_text).adjusted(-8, -4, 8, 4)
            r.moveTopLeft(self.viewport().rect().topLeft() + QPointF(12, 12).toPoint())
            painter.fillRect(r, QColor(0, 0, 0, 150))
            painter.drawText(r, Qt.AlignCenter, self.overlay_text)
            painter.restore()

    def set_overlay(self, text: str):
        self.overlay_text = text
        self.viewport().update()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.filesDropped.emit(files)
