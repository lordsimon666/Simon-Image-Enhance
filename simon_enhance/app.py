"""Main window of Simon Image Enhance."""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPalette, QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from . import __version__
from .auto import ai_correct
from .fileio import (
    ExportOptions, ImageData, RAW_EXTS, RASTER_EXTS, export_image, is_supported, load_image,
    load_thumbnail, raw_available,
)
from .processing import (
    NEUTRAL_TEMP, Params, Renderer, TEMP_MAX, TEMP_MIN, TINT_MAX, TINT_MIN, identity_curve,
    solve_white_balance, to_uint8,
)
from .store import EditStore, Settings, is_portable
from .widgets import CurveEditor, Histogram, ImageView, SliderRow, array_to_qimage

APP_TITLE = "Simon Image Enhance"


def resource_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _Signals(QObject):
    done = Signal(object, object)       # tag, result
    failed = Signal(object, str)
    progress = Signal(int, str)


class Task(QRunnable):
    def __init__(self, tag, fn, *args):
        super().__init__()
        self.tag, self.fn, self.args = tag, fn, args
        self.signals = _Signals()

    def run(self):
        try:
            res = self.fn(*self.args)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            res, err = None, str(exc)
        else:
            err = None
        try:
            if err is None:
                self.signals.done.emit(self.tag, res)
            else:
                self.signals.failed.emit(self.tag, err)
        except RuntimeError:
            pass  # receiver already gone (application shutting down)


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------

class ExportDialog(QDialog):
    def __init__(self, settings: Settings, count: int, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(f"Exportieren ({count} Bild{'er' if count != 1 else ''})")
        form = QFormLayout(self)

        self.fmt = QComboBox()
        self.fmt.addItem("JPEG (.jpg)", "jpeg")
        self.fmt.addItem("PNG (.png)", "png")
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.bits = QComboBox()
        self.bits.addItem("8 Bit", 8)
        self.bits.addItem("16 Bit", 16)
        self.resize_cb = QCheckBox("Lange Kante begrenzen auf")
        self.long_edge = QSpinBox()
        self.long_edge.setRange(64, 20000)
        self.long_edge.setSuffix(" px")
        self.keep_exif = QCheckBox("EXIF-Metadaten übernehmen")
        self.suffix = QLineEdit()
        self.folder = QLineEdit()
        browse = QPushButton("…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        rs = QHBoxLayout()
        rs.addWidget(self.resize_cb)
        rs.addWidget(self.long_edge)

        form.addRow("Format:", self.fmt)
        form.addRow("JPEG-Qualität:", self.quality)
        form.addRow("PNG-Farbtiefe:", self.bits)
        form.addRow(rs)
        form.addRow(self.keep_exif)
        form.addRow("Dateiname-Zusatz:", self.suffix)
        form.addRow("Zielordner:", row)
        hint = QLabel("Leer lassen = Ordner des Originals")
        hint.setStyleSheet("color:#888")
        form.addRow("", hint)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Exportieren")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

        s = settings.get("export", {})
        self.fmt.setCurrentIndex(0 if s.get("fmt", "jpeg") == "jpeg" else 1)
        self.quality.setValue(s.get("quality", 92))
        self.bits.setCurrentIndex(1 if s.get("png_bits", 8) == 16 else 0)
        self.resize_cb.setChecked(bool(s.get("resize", False)))
        self.long_edge.setValue(s.get("long_edge", 3000))
        self.keep_exif.setChecked(s.get("keep_exif", True))
        self.suffix.setText(s.get("suffix", "_edit"))
        self.folder.setText(s.get("folder", ""))
        self.fmt.currentIndexChanged.connect(self._update)
        self.resize_cb.toggled.connect(self._update)
        self._update()

    def _update(self):
        jpeg = self.fmt.currentData() == "jpeg"
        self.quality.setEnabled(jpeg)
        self.keep_exif.setEnabled(jpeg)
        self.bits.setEnabled(not jpeg)
        self.long_edge.setEnabled(self.resize_cb.isChecked())

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Zielordner wählen", self.folder.text())
        if d:
            self.folder.setText(d)

    def result_options(self):
        s = {
            "fmt": self.fmt.currentData(), "quality": self.quality.value(),
            "png_bits": self.bits.currentData(), "resize": self.resize_cb.isChecked(),
            "long_edge": self.long_edge.value(), "keep_exif": self.keep_exif.isChecked(),
            "suffix": self.suffix.text(), "folder": self.folder.text().strip(),
        }
        self.settings.set("export", s)
        opts = ExportOptions(fmt=s["fmt"], quality=s["quality"], png_bits=s["png_bits"],
                             long_edge=s["long_edge"] if s["resize"] else None,
                             keep_exif=s["keep_exif"])
        return opts, s["folder"], s["suffix"]


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1500, 920)
        self.store = EditStore()
        self.settings = Settings()
        self.pool = QThreadPool.globalInstance()
        self.thumb_pool = QThreadPool(self)
        self.thumb_pool.setMaxThreadCount(2)

        self.image: ImageData | None = None
        self.params = Params()
        self.history: list[Params] = []
        self.hist_pos = -1
        self.clipboard: Params | None = None
        self.load_token = 0
        self.show_before = False
        self.current_path: str | None = None
        self._tasks = set()

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(10)
        self.render_timer.timeout.connect(self._render)
        self.commit_timer = QTimer(self)
        self.commit_timer.setSingleShot(True)
        self.commit_timer.setInterval(350)
        self.commit_timer.timeout.connect(self._commit)

        self._build_ui()
        self._build_actions()
        self._set_enabled(False)
        self.statusBar().showMessage(
            "Bild oder Ordner öffnen (Strg+O / Strg+Shift+O) – oder Dateien hier hineinziehen."
            + ("" if raw_available() else "  ·  RAW-Unterstützung nicht installiert"))

    # ----------------------------------------------------------------- UI
    def _build_ui(self):
        self.view = ImageView()
        self.view.pixelClicked.connect(self._on_pick)
        self.view.filesDropped.connect(self.open_paths)

        self.film = QListWidget()
        self.film.setViewMode(QListWidget.IconMode)
        self.film.setFlow(QListWidget.LeftToRight)
        self.film.setWrapping(False)
        self.film.setMovement(QListWidget.Static)
        self.film.setIconSize(QSize(120, 90))
        self.film.setFixedHeight(140)
        self.film.setSpacing(4)
        self.film.setSelectionMode(QListWidget.ExtendedSelection)
        self.film.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.film.currentItemChanged.connect(self._on_film_select)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.view, 1)
        cl.addWidget(self.film)

        # right panel
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(10, 10, 10, 10)
        pl.setSpacing(10)

        self.histogram = Histogram()
        pl.addWidget(self.histogram)
        self.info = QLabel("")
        self.info.setStyleSheet("color:#999;font-size:11px")
        pl.addWidget(self.info)

        # AI
        ai = QGroupBox("KI-Korrektur")
        al = QVBoxLayout(ai)
        self.ai_btn = QPushButton("✨  KI-Korrektur anwenden")
        self.ai_btn.setObjectName("accent")
        self.ai_btn.setMinimumHeight(34)
        self.ai_btn.setToolTip("Analysiert das Bild und setzt automatisch Weißabgleich und "
                               "Gradationskurven (Strg+U). Alles bleibt danach anpassbar.")
        self.ai_btn.clicked.connect(self.apply_ai)
        self.ai_strength = SliderRow("Stärke", 0, 100, 100)
        al.addWidget(self.ai_btn)
        al.addWidget(self.ai_strength)
        note = QLabel("Ändert nur Weißabgleich und Kurven.")
        note.setStyleSheet("color:#888;font-size:11px")
        al.addWidget(note)
        pl.addWidget(ai)

        # White balance
        wb = QGroupBox("Weißabgleich")
        wl = QVBoxLayout(wb)
        top = QHBoxLayout()
        self.wb_preset = QComboBox()
        self.wb_preset.addItems(["Wie Aufnahme", "Auto", "Tageslicht", "Bewölkt", "Schatten",
                                 "Kunstlicht", "Leuchtstoff", "Blitz", "Benutzerdefiniert"])
        self.wb_preset.activated.connect(self._on_wb_preset)
        self.picker_btn = QToolButton()
        self.picker_btn.setText("🎯 Pipette")
        self.picker_btn.setCheckable(True)
        self.picker_btn.setToolTip("Auf eine neutrale (graue/weiße) Stelle im Bild klicken (W)")
        self.picker_btn.toggled.connect(self.view.set_picking)
        top.addWidget(self.wb_preset, 1)
        top.addWidget(self.picker_btn)
        wl.addLayout(top)
        self.s_temp = SliderRow("Temperatur", TEMP_MIN, TEMP_MAX, NEUTRAL_TEMP, log_scale=True,
                                gradient=("#3b6fd8", "#e8c040"))
        self.s_tint = SliderRow("Tönung", TINT_MIN, TINT_MAX, 0, gradient=("#3fae3f", "#c040c0"))
        wl.addWidget(self.s_temp)
        wl.addWidget(self.s_tint)
        pl.addWidget(wb)

        # Tone
        tone = QGroupBox("Tonwert")
        tl = QVBoxLayout(tone)
        self.s_exposure = SliderRow("Belichtung", -5, 5, 0, decimals=2)
        self.s_contrast = SliderRow("Kontrast", -100, 100, 0)
        self.s_high = SliderRow("Lichter", -100, 100, 0)
        self.s_shadow = SliderRow("Tiefen", -100, 100, 0)
        self.s_whites = SliderRow("Weiß", -100, 100, 0)
        self.s_blacks = SliderRow("Schwarz", -100, 100, 0)
        for s in (self.s_exposure, self.s_contrast, self.s_high, self.s_shadow, self.s_whites, self.s_blacks):
            tl.addWidget(s)
        pl.addWidget(tone)

        pres = QGroupBox("Präsenz")
        prl = QVBoxLayout(pres)
        self.s_vib = SliderRow("Dynamik", -100, 100, 0)
        self.s_sat = SliderRow("Sättigung", -100, 100, 0)
        prl.addWidget(self.s_vib)
        prl.addWidget(self.s_sat)
        pl.addWidget(pres)

        # Curves
        cg = QGroupBox("Gradationskurve")
        cvl = QVBoxLayout(cg)
        chrow = QHBoxLayout()
        self.ch_group = QButtonGroup(self)
        for i, (key, text) in enumerate([("rgb", "RGB"), ("r", "Rot"), ("g", "Grün"), ("b", "Blau")]):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty("channel", key)
            b.setChecked(i == 0)
            self.ch_group.addButton(b, i)
            chrow.addWidget(b)
        self.ch_group.buttonClicked.connect(lambda b: self.curve.set_channel(b.property("channel")))
        cvl.addLayout(chrow)
        self.curve = CurveEditor()
        self.curve.changed.connect(self._on_curve_changed)
        self.curve.editingFinished.connect(self._schedule_commit)
        cvl.addWidget(self.curve)
        crow = QHBoxLayout()
        hint = QLabel("Klick: Punkt · Rechtsklick: löschen")
        hint.setStyleSheet("color:#888;font-size:11px")
        rb = QPushButton("Kanal zurücksetzen")
        rb.clicked.connect(self.curve.reset_channel)
        crow.addWidget(hint, 1)
        crow.addWidget(rb)
        cvl.addLayout(crow)
        pl.addWidget(cg)

        reset_all = QPushButton("Alle Einstellungen zurücksetzen")
        reset_all.clicked.connect(self.reset_all)
        pl.addWidget(reset_all)
        pl.addStretch(1)

        self.sliders = {
            "temp": self.s_temp, "tint": self.s_tint, "exposure": self.s_exposure,
            "contrast": self.s_contrast, "highlights": self.s_high, "shadows": self.s_shadow,
            "whites": self.s_whites, "blacks": self.s_blacks, "vibrance": self.s_vib,
            "saturation": self.s_sat,
        }
        for key, s in self.sliders.items():
            s.valueChanged.connect(lambda v, k=key: self._on_slider(k, v))
            s.editingFinished.connect(self._schedule_commit)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(360)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel = panel

        split = QSplitter()
        split.addWidget(center)
        split.addWidget(scroll)
        split.setStretchFactor(0, 1)
        split.setSizes([1100, 380])
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())

    def _build_actions(self):
        mb = self.menuBar()

        def act(menu, text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setStatusTip(tip)
            a.triggered.connect(slot)
            menu.addAction(a)
            self.addAction(a)
            return a

        f = mb.addMenu("&Datei")
        act(f, "Bilder öffnen…", self.open_files_dialog, "Ctrl+O")
        act(f, "Ordner öffnen…", self.open_folder_dialog, "Ctrl+Shift+O")
        f.addSeparator()
        self.a_export = act(f, "Exportieren…", self.export_current, "Ctrl+E")
        self.a_export_sel = act(f, "Auswahl exportieren…", self.export_selected, "Ctrl+Shift+E")
        f.addSeparator()
        act(f, "Beenden", self.close, "Ctrl+Q")

        e = mb.addMenu("&Bearbeiten")
        self.a_undo = act(e, "Rückgängig", self.undo, QKeySequence.Undo)
        self.a_redo = act(e, "Wiederholen", self.redo, "Ctrl+Shift+Z")
        self.addAction(self.a_redo)
        e.addSeparator()
        act(e, "Einstellungen kopieren", self.copy_settings, "Ctrl+Shift+C")
        act(e, "Einstellungen einfügen (Auswahl)", self.paste_settings, "Ctrl+Shift+V")
        e.addSeparator()
        act(e, "KI-Korrektur", self.apply_ai, "Ctrl+U")
        act(e, "KI-Korrektur auf Auswahl anwenden", self.apply_ai_selected, "Ctrl+Shift+U")
        act(e, "Weißabgleich-Pipette", lambda: self.picker_btn.toggle(), "W")
        act(e, "Alles zurücksetzen", self.reset_all, "Ctrl+Shift+R")

        v = mb.addMenu("&Ansicht")
        act(v, "Vorher / Nachher", self.toggle_before, "\\")
        act(v, "Einpassen", self.view.fit_view, "Ctrl+0")
        act(v, "100 % (Vorschau)", self.view.zoom_100, "Ctrl+1")
        act(v, "Nächstes Bild", lambda: self._step(1), "Right")
        act(v, "Vorheriges Bild", lambda: self._step(-1), "Left")

        h = mb.addMenu("&Hilfe")
        act(h, "Über…", self.about)

    def _set_enabled(self, on: bool):
        self.panel.setEnabled(on)
        for a in (self.a_export,):
            a.setEnabled(on)

    # ------------------------------------------------------------- opening
    def open_files_dialog(self):
        exts = " ".join("*" + e for e in sorted(RASTER_EXTS | RAW_EXTS))
        raw = " ".join("*" + e for e in sorted(RAW_EXTS))
        start = self.settings.get("last_dir", os.path.expanduser("~"))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Bilder öffnen", start,
            f"Alle Bilder ({exts});;JPEG/PNG (*.jpg *.jpeg *.png);;RAW ({raw});;Alle Dateien (*)")
        if files:
            self.open_paths(files)

    def open_folder_dialog(self):
        start = self.settings.get("last_dir", os.path.expanduser("~"))
        d = QFileDialog.getExistingDirectory(self, "Ordner öffnen", start)
        if d:
            self.open_paths([d])

    def open_paths(self, paths: list[str]):
        files = []
        for p in paths:
            if os.path.isdir(p):
                for name in sorted(os.listdir(p), key=str.lower):
                    fp = os.path.join(p, name)
                    if os.path.isfile(fp) and is_supported(fp):
                        files.append(fp)
            elif is_supported(p):
                files.append(p)
        if not files:
            QMessageBox.information(self, APP_TITLE, "Keine unterstützten Bilddateien gefunden.")
            return
        self.settings.set("last_dir", os.path.dirname(files[0]))
        self._save_current()
        self.film.blockSignals(True)
        self.film.clear()
        placeholder = QPixmap(120, 90)
        placeholder.fill(QColor(50, 50, 50))
        for fp in files:
            it = QListWidgetItem(QIcon(placeholder), os.path.basename(fp))
            it.setData(Qt.UserRole, fp)
            it.setToolTip(fp)
            it.setSizeHint(QSize(132, 118))
            self.film.addItem(it)
        self.film.blockSignals(False)
        for i, fp in enumerate(files):
            self._run(("thumb", i, fp), self._thumb_job, fp, pool=self.thumb_pool)
        self.film.setCurrentRow(0)

    def _thumb_job(self, fp):
        arr = load_thumbnail(fp, 180)
        p = self.store.get(fp)
        if p is not None and not p.is_default():
            # show edited thumbnail
            from .processing import srgb_to_linear
            arr = to_uint8(Renderer(p).apply(srgb_to_linear(arr.astype(np.float32) / 255.0)))
        return arr

    def _run(self, tag, fn, *args, pool=None):
        t = Task(tag, fn, *args)
        t.signals.done.connect(self._on_task_done)
        t.signals.failed.connect(self._on_task_failed)
        self._tasks.add(t.signals)
        (pool or self.pool).start(t)

    def _on_task_done(self, tag, res):
        self._tasks.discard(self.sender())
        kind = tag[0]
        if kind == "thumb":
            _, i, fp = tag
            it = self.film.item(i)
            if it is not None and it.data(Qt.UserRole) == fp:
                pm = QPixmap.fromImage(array_to_qimage(res))
                icon = QIcon(pm)
                icon.addPixmap(pm, QIcon.Selected)  # no blue tint on selected thumbnails
                it.setIcon(icon)
        elif kind == "load":
            _, token, fp = tag
            if token == self.load_token:
                self._on_loaded(res)

    def _on_task_failed(self, tag, msg):
        self._tasks.discard(self.sender())
        if tag[0] == "load" and tag[1] == self.load_token:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(f"Fehler beim Laden: {msg}")
            QMessageBox.warning(self, APP_TITLE, f"Datei konnte nicht geladen werden:\n{tag[2]}\n\n{msg}")

    def _on_film_select(self, cur, prev):
        if cur is None:
            return
        fp = cur.data(Qt.UserRole)
        if fp == self.current_path:
            return
        self._save_current()
        self.load_token += 1
        self.current_path = fp
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Lade {os.path.basename(fp)} …")
        self._run(("load", self.load_token, fp), load_image, fp, True)

    def _on_loaded(self, data: ImageData):
        QApplication.restoreOverrideCursor()
        self.image = data
        self.before_img = None
        p = self.store.get(data.path) or Params()
        self.history = [p.copy()]
        self.hist_pos = 0
        self._apply_params_to_ui(p)
        self._set_enabled(True)
        self.view.set_image(None)
        self._render(refit=True)
        w, h = data.full_size
        kind = "RAW" if data.is_raw else os.path.splitext(data.path)[1][1:].upper()
        self.info.setText(f"{os.path.basename(data.path)}  ·  {w}×{h}  ·  {kind}")
        self.setWindowTitle(f"{os.path.basename(data.path)} – {APP_TITLE}")
        self.statusBar().showMessage("Bereit", 2000)

    def _step(self, d):
        r = self.film.currentRow() + d
        if 0 <= r < self.film.count():
            self.film.setCurrentRow(r)

    # ------------------------------------------------------------- params
    def _apply_params_to_ui(self, p: Params):
        self.params = p.copy()
        for key, s in self.sliders.items():
            s.setValue(getattr(p, key))
        self.curve.set_curves(p.curves)
        self._sync_wb_preset()

    def _sync_wb_preset(self):
        p = self.params
        if abs(p.temp - NEUTRAL_TEMP) < 1 and abs(p.tint) < 0.5:
            self.wb_preset.setCurrentIndex(0)
        elif self.wb_preset.currentIndex() not in (1,):
            self.wb_preset.setCurrentText("Benutzerdefiniert")

    def _on_slider(self, key, v):
        setattr(self.params, key, float(v))
        if key in ("temp", "tint"):
            self._sync_wb_preset()
        self._request_render()
        self.commit_timer.start()

    def _on_curve_changed(self):
        self.params.curves = {c: [list(p) for p in pts] for c, pts in self.curve.curves.items()}
        self._request_render()

    def _schedule_commit(self):
        self.commit_timer.start()

    def _commit(self):
        if self.image is None:
            return
        if self.hist_pos >= 0 and self.history[self.hist_pos] == self.params:
            return
        del self.history[self.hist_pos + 1:]
        self.history.append(self.params.copy())
        self.hist_pos = len(self.history) - 1
        self._save_current()

    def _save_current(self):
        if self.image is not None:
            try:
                self.store.set(self.image.path, self.params)
                self._refresh_thumb(self.image.path)
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"Speichern fehlgeschlagen: {exc}")

    def _refresh_thumb(self, path):
        for i in range(self.film.count()):
            if self.film.item(i).data(Qt.UserRole) == path:
                self._run(("thumb", i, path), self._thumb_job, path, pool=self.thumb_pool)
                break

    def undo(self):
        self.commit_timer.stop()
        self._commit()
        if self.hist_pos > 0:
            self.hist_pos -= 1
            self._apply_params_to_ui(self.history[self.hist_pos])
            self._request_render()
            self._save_current()

    def redo(self):
        if self.hist_pos < len(self.history) - 1:
            self.hist_pos += 1
            self._apply_params_to_ui(self.history[self.hist_pos])
            self._request_render()
            self._save_current()

    def _set_params(self, p: Params):
        self._apply_params_to_ui(p)
        self._request_render()
        self.commit_timer.stop()
        self._commit()

    def reset_all(self):
        if self.image is not None:
            self._set_params(Params())

    # ----------------------------------------------------------- features
    def apply_ai(self):
        if self.image is None:
            return
        strength = self.ai_strength.value() / 100.0
        p = ai_correct(self.image.linear, self.params, strength)
        self._set_params(p)
        self.wb_preset.setCurrentText("Auto")
        self.statusBar().showMessage(
            f"KI-Korrektur: Temperatur {p.temp:.0f} K, Tönung {p.tint:+.0f}, Kurven gesetzt", 4000)

    def apply_ai_selected(self):
        paths = self._selected_paths()
        if not paths:
            return
        self._save_current()
        strength = self.ai_strength.value() / 100.0
        dlg = QProgressDialog("KI-Korrektur …", "Abbrechen", 0, len(paths), self)
        dlg.setWindowModality(Qt.WindowModal)
        for i, fp in enumerate(paths):
            if dlg.wasCanceled():
                break
            dlg.setValue(i)
            dlg.setLabelText(os.path.basename(fp))
            QApplication.processEvents()
            try:
                data = self.image if self.image and self.image.path == fp else load_image(fp, True)
                base = self.store.get(fp) or Params()
                p = ai_correct(data.linear, base, strength)
                self.store.set(fp, p)
                if self.image and self.image.path == fp:
                    self._set_params(p)
                self._refresh_thumb(fp)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, APP_TITLE, f"{fp}:\n{exc}")
        dlg.setValue(len(paths))

    def _on_wb_preset(self, idx):
        name = self.wb_preset.currentText()
        if self.image is None:
            return
        p = self.params.copy()
        if name == "Wie Aufnahme":
            p.temp, p.tint = NEUTRAL_TEMP, 0.0
        elif name == "Auto":
            from .auto import auto_white_balance
            p.temp, p.tint = auto_white_balance(self.image.linear)
        else:
            presets = {"Tageslicht": (5500, 10), "Bewölkt": (6500, 10), "Schatten": (7500, 10),
                       "Kunstlicht": (2850, 0), "Leuchtstoff": (3800, 21), "Blitz": (5500, 0)}
            if name not in presets:
                return
            # presets are relative to a daylight (5500 K) "as shot" reference
            t, tint = presets[name]
            p.temp = float(np.clip(NEUTRAL_TEMP * t / 5500.0, TEMP_MIN, TEMP_MAX))
            p.tint = float(tint)
        self._set_params(p)
        self.wb_preset.setCurrentText(name)

    def _on_pick(self, fx, fy):
        if self.image is None:
            return
        lin = self.image.linear
        h, w = lin.shape[:2]
        x, y = int(fx * w), int(fy * h)
        if not (0 <= x < w and 0 <= y < h):
            return
        r = max(2, min(w, h) // 300)
        patch = lin[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
        rgb = np.median(patch, axis=0)
        if rgb.max() < 0.002 or rgb.min() > 0.995:
            self.statusBar().showMessage("Stelle zu dunkel oder überbelichtet – bitte andere wählen.", 3000)
            return
        p = self.params.copy()
        p.temp, p.tint = solve_white_balance(rgb)
        self._set_params(p)
        self.picker_btn.setChecked(False)
        self.statusBar().showMessage(f"Weißabgleich gesetzt: {p.temp:.0f} K, Tönung {p.tint:+.0f}", 3000)

    def toggle_before(self):
        if self.image is None:
            return
        self.show_before = not self.show_before
        self.view.set_overlay("Vorher" if self.show_before else "")
        self._render()

    def copy_settings(self):
        if self.image is not None:
            self.clipboard = self.params.copy()
            self.statusBar().showMessage("Einstellungen kopiert", 2000)

    def paste_settings(self):
        if self.clipboard is None:
            return
        paths = self._selected_paths()
        for fp in paths:
            if self.image is not None and fp == self.image.path:
                self._set_params(self.clipboard)
            else:
                self.store.set(fp, self.clipboard)
                self._refresh_thumb(fp)
        self.statusBar().showMessage(f"Einstellungen auf {len(paths)} Bild(er) übertragen", 3000)

    def _selected_paths(self) -> list[str]:
        items = self.film.selectedItems()
        if not items and self.image is not None:
            return [self.image.path]
        return [it.data(Qt.UserRole) for it in items]

    # ------------------------------------------------------------- render
    def _request_render(self):
        if not self.render_timer.isActive():
            self.render_timer.start()

    def _render(self, refit=False):
        if self.image is None:
            return
        p = Params() if self.show_before else self.params
        img8 = to_uint8(Renderer(p).apply(self.image.linear.copy()))
        self.view.set_image(img8, refit=refit)
        self.histogram.set_image(img8)
        self.curve.set_histogram(img8)

    # ------------------------------------------------------------- export
    def export_current(self):
        if self.image is not None:
            self._export([self.image.path])

    def export_selected(self):
        self._export(self._selected_paths())

    def _export(self, paths):
        if not paths:
            return
        self.commit_timer.stop()
        self._commit()
        dlg = ExportDialog(self.settings, len(paths), self)
        if dlg.exec() != QDialog.Accepted:
            return
        opts, folder, suffix = dlg.result_options()
        ext = ".jpg" if opts.fmt == "jpeg" else ".png"
        jobs = []
        for fp in paths:
            params = self.params if (self.image and fp == self.image.path) else (self.store.get(fp) or Params())
            out_dir = folder or os.path.dirname(fp)
            base = os.path.splitext(os.path.basename(fp))[0] + suffix
            dst = os.path.join(out_dir, base + ext)
            n = 2
            while os.path.exists(dst) or dst in [j[2] for j in jobs]:
                dst = os.path.join(out_dir, f"{base}-{n}{ext}")
                n += 1
            jobs.append((fp, params.copy(), dst))

        prog = QProgressDialog("Exportiere …", "Abbrechen", 0, len(jobs), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        state = {"i": 0, "errors": []}

        def next_job():
            i = state["i"]
            if i >= len(jobs) or prog.wasCanceled():
                prog.setValue(len(jobs))
                errs = state["errors"]
                done = i - len(errs)
                msg = f"{done} Bild(er) exportiert."
                if errs:
                    msg += "\n\nFehler:\n" + "\n".join(errs[:10])
                    QMessageBox.warning(self, APP_TITLE, msg)
                else:
                    self.statusBar().showMessage(msg, 5000)
                return
            fp, params, dst = jobs[i]
            prog.setLabelText(f"Exportiere {os.path.basename(fp)} …")
            t = Task(("export", i), export_image, fp, params, dst, opts)

            def ok(tag, res):
                state["i"] += 1
                prog.setValue(state["i"])
                next_job()

            def fail(tag, msg):
                state["errors"].append(f"{os.path.basename(fp)}: {msg}")
                ok(tag, None)

            t.signals.done.connect(ok)
            t.signals.failed.connect(fail)
            self._tasks.add(t.signals)
            self.pool.start(t)

        next_job()

    # ------------------------------------------------------------- misc
    def about(self):
        QMessageBox.about(
            self, f"Über {APP_TITLE}",
            f"<h3>{APP_TITLE} {__version__}</h3>"
            "<p>Nicht-destruktive Fotobearbeitung: Weißabgleich, Tonwerte, Gradationskurven "
            "und KI-Korrektur.</p>"
            f"<p>RAW-Unterstützung: {'ja (LibRaw)' if raw_available() else 'nein'}<br>"
            f"Modus: {'Portabel' if is_portable() else 'Installiert'}</p>")

    def closeEvent(self, e):
        self.commit_timer.stop()
        self._commit()
        self._save_current()
        super().closeEvent(e)


# ---------------------------------------------------------------------------

STYLE = """
QWidget { font-size: 12px; }
QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 14px; padding: 8px 6px 6px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #ddd; font-weight: bold; }
QPushButton { background: #3a3a3a; border: 1px solid #4a4a4a; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #454545; }
QPushButton:checked { background: #555; border-color: #888; }
QPushButton#accent { background: #2f6fd6; border-color: #4a86e8; font-weight: bold; color: white; }
QPushButton#accent:hover { background: #3a7ee8; }
QToolButton { background: #3a3a3a; border: 1px solid #4a4a4a; border-radius: 4px; padding: 4px 8px; }
QToolButton:checked { background: #2f6fd6; }
QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; }
QSlider::handle:horizontal { background: #ddd; width: 12px; margin: -5px 0; border-radius: 6px; }
QLineEdit, QComboBox { background: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 2px 4px; }
QDoubleSpinBox, QSpinBox { background: #2a2a2a; border: 1px solid #444; border-radius: 3px; padding: 1px 4px; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; border: none; }
QListWidget { background: #1e1e1e; border: none; border-top: 1px solid #333; }
QListWidget::item:selected { background: #2f6fd6; }
QScrollArea { background: #2b2b2b; }
"""


def apply_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(43, 43, 43))
    pal.setColor(QPalette.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.Base, QColor(30, 30, 30))
    pal.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    pal.setColor(QPalette.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.Button, QColor(58, 58, 58))
    pal.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.Highlight, QColor(47, 111, 214))
    pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
    pal.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    QApplication.setApplicationName(APP_TITLE)
    QApplication.setOrganizationName("Simon")
    QApplication.setDesktopFileName("io.github.lordsimon666.SimonImageEnhance")
    app = QApplication(argv)
    apply_dark_theme(app)
    icon_file = resource_path("assets", "icon.png")
    if os.path.exists(icon_file):
        app.setWindowIcon(QIcon(icon_file))
    win = MainWindow()
    win.show()
    files = [a for a in argv[1:] if os.path.exists(a)]
    if files:
        QTimer.singleShot(0, lambda: win.open_paths(files))
    return app.exec()
