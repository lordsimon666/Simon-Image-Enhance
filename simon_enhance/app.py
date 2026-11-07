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
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

from . import __version__, gpu
from .auto import ai_correct
from .greenscreen import CHANNEL_NAMES, ai_remove_background, checkerboard, composite, has_key
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
            from . import net
            res, err = None, net.describe_error(exc)
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
        self.jpeg_bg = QComboBox()
        self.jpeg_bg.addItem("Weiß", (1.0, 1.0, 1.0))
        self.jpeg_bg.addItem("Schwarz", (0.0, 0.0, 0.0))
        self.jpeg_bg.addItem("Grau", (0.5, 0.5, 0.5))
        self.alpha_hint = QLabel("PNG speichert Transparenz (freigestellter Hintergrund).\n"
                                 "JPEG kennt keine Transparenz – dort wird die Hintergrundfarbe verwendet.")
        self.alpha_hint.setStyleSheet("color:#888")
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
        form.addRow("JPEG-Hintergrund:", self.jpeg_bg)
        form.addRow("", self.alpha_hint)
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
        self.jpeg_bg.setCurrentIndex(int(s.get("jpeg_bg", 0)))
        self.suffix.setText(s.get("suffix", "_edit"))
        self.folder.setText(s.get("folder", ""))
        self.fmt.currentIndexChanged.connect(self._update)
        self.resize_cb.toggled.connect(self._update)
        self._update()

    def _update(self):
        jpeg = self.fmt.currentData() == "jpeg"
        self.quality.setEnabled(jpeg)
        self.keep_exif.setEnabled(jpeg)
        self.jpeg_bg.setEnabled(jpeg)
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
            "jpeg_bg": self.jpeg_bg.currentIndex(),
        }
        self.settings.set("export", s)
        opts = ExportOptions(fmt=s["fmt"], quality=s["quality"], png_bits=s["png_bits"],
                             long_edge=s["long_edge"] if s["resize"] else None,
                             keep_exif=s["keep_exif"],
                             jpeg_background=tuple(self.jpeg_bg.currentData()))
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
        # the AI engine reports "checking the GPU (once) ..." from worker threads
        self._ai_sig = _Signals()
        self._ai_sig.progress.connect(lambda _n, msg: self.statusBar().showMessage(msg, 15000))
        from .ai import engine as _ai_engine
        _ai_engine.activity_hook = lambda msg: self._ai_sig.progress.emit(0, msg)
        self._block_key = False

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
        self._sync_key_ui()
        QTimer.singleShot(300, self._report_gpu)
        QTimer.singleShot(2000, self._check_gpu_runtimes)
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
        self.film.setContextMenuPolicy(Qt.CustomContextMenu)
        self.film.customContextMenuRequested.connect(self._film_menu)

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
        self.ai_method = QComboBox()
        self.ai_method.addItem("Analyse – neutral (empfohlen)", "analysis")
        self.ai_method.addItem("Neuronales Netz – Profi-Look (FiveK)", "neural")
        self.ai_method.setToolTip(
            "Analyse: physikalisch, entfernt Farbstiche am zuverlässigsten.\n"
            "Profi-Look: Netz, trainiert auf 5000 von Profis bearbeiteten Fotos (MIT-Adobe FiveK) –\n"
            "kräftiger und etwas wärmer. Beide setzen nur Weißabgleich und Kurven.")
        idx = self.ai_method.findData(self.settings.get("ai_method", "analysis"))
        self.ai_method.setCurrentIndex(max(idx, 0))
        self.ai_method.currentIndexChanged.connect(
            lambda _i: self.settings.set("ai_method", self.ai_method.currentData()))
        al.addWidget(self.ai_method)
        al.addWidget(self.ai_btn)
        al.addWidget(self.ai_strength)
        note = QLabel("Ändert nur Weißabgleich und Kurven.")
        note.setStyleSheet("color:#888;font-size:11px")
        al.addWidget(note)
        pl.addWidget(ai)

        # Background removal
        bgr = QGroupBox("Hintergrund entfernen (KI)")
        bl = QVBoxLayout(bgr)
        self.key_btn = QPushButton("✨  Greenscreen entfernen")
        self.key_btn.setObjectName("accent")
        self.key_btn.setMinimumHeight(34)
        self.key_btn.setToolTip("Erkennt Green- oder Bluescreen automatisch und macht ihn "
                                "transparent (Strg+G). PNG-Export speichert die Transparenz.")
        self.key_btn.clicked.connect(self.apply_key)
        self.ai_bg_btn = QPushButton("✨  Hintergrund entfernen (jedes Foto)")
        self.ai_bg_btn.setObjectName("accent")
        self.ai_bg_btn.setMinimumHeight(34)
        self.ai_bg_btn.setToolTip("Neuronales Netz (BiRefNet) erkennt Personen und Objekte und stellt "
                                  "sie frei – ohne Greenscreen (Strg+Shift+G).")
        self.ai_bg_btn.clicked.connect(self.apply_ai_background)
        self.key_enable = QCheckBox("Hintergrund entfernt")
        self.key_enable.toggled.connect(self._on_key_toggled)
        self.key_info = QLabel("")
        self.key_info.setStyleSheet("color:#888;font-size:11px")
        self.key_info.setWordWrap(True)
        self.s_key_tol = SliderRow("Toleranz", 0, 100, 50)
        self.s_key_tol.setToolTip("Höher = mehr vom Hintergrund wird entfernt")
        self.s_key_soft = SliderRow("Weichheit", 0, 100, 50)
        self.s_key_soft.setToolTip("Breite des weichen Übergangs an den Kanten")
        self.s_key_edge = SliderRow("Kante", -100, 100, 25)
        self.s_key_edge.setToolTip("Positiv: Kante nach innen ziehen (entfernt Farbsäume), negativ: erweitern")
        self.s_key_spill = SliderRow("Entfärben", 0, 100, 80)
        self.s_key_spill.setToolTip("Entfernt grüne/blaue Reflexe (Spill) auf Haut, Haaren und Kleidung")
        bl.addWidget(self.key_btn)
        bl.addWidget(self.ai_bg_btn)
        bl.addWidget(self.key_enable)
        for w_ in (self.s_key_tol, self.s_key_soft, self.s_key_edge, self.s_key_spill):
            bl.addWidget(w_)
        bl.addWidget(self.key_info)
        self.key_controls = (self.s_key_tol, self.s_key_soft, self.s_key_edge, self.s_key_spill)
        pl.addWidget(bgr)

        # AI enhance (new file)
        enh = QGroupBox("KI-Verbessern (neue Datei)")
        el = QVBoxLayout(enh)
        self.enh_denoise = QCheckBox("Entrauschen")
        self.enh_denoise.setChecked(True)
        self.s_enh_strength = SliderRow("Stärke", 0, 100, 80)
        up_row = QHBoxLayout()
        up_row.addWidget(QLabel("Hochskalieren"))
        self.enh_scale = QComboBox()
        self.enh_scale.addItem("Aus", 1)
        self.enh_scale.addItem("2× (doppelte Kantenlänge)", 2)
        self.enh_scale.addItem("4× (vierfache Kantenlänge)", 4)
        up_row.addWidget(self.enh_scale, 1)
        self.enh_btn = QPushButton("Verbessertes Bild erstellen …")
        self.enh_btn.setToolTip("Erzeugt aus dem Original eine neue 16-Bit-TIFF-Datei (entrauscht "
                                "und/oder hochskaliert) und öffnet sie mit denselben Einstellungen.")
        self.enh_btn.clicked.connect(self.ai_enhance)
        self.enh_denoise.toggled.connect(self.s_enh_strength.setEnabled)
        el.addWidget(self.enh_denoise)
        el.addWidget(self.s_enh_strength)
        el.addLayout(up_row)
        el.addWidget(self.enh_btn)
        note2 = QLabel("Rechnet auf NPU / Grafikkarte, falls vorhanden (Einstellungen → KI-Beschleuniger).")
        note2.setStyleSheet("color:#888;font-size:11px")
        note2.setWordWrap(True)
        el.addWidget(note2)
        pl.addWidget(enh)

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
            "key_tolerance": self.s_key_tol, "key_softness": self.s_key_soft,
            "key_edge": self.s_key_edge, "key_despill": self.s_key_spill,
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
        self.a_close = act(f, "Bild schließen", self.close_current, "Ctrl+W")
        self.a_close_sel = act(f, "Ausgewählte Bilder schließen", self.close_selected)
        self.a_close_all = act(f, "Alle Bilder schließen", self.close_all, "Ctrl+Shift+W")
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
        act(e, "Greenscreen entfernen (KI)", self.apply_key, "Ctrl+G")
        act(e, "Hintergrund entfernen – jedes Foto (KI)", self.apply_ai_background, "Ctrl+Shift+G")
        act(e, "KI-Verbessern (Entrauschen / Hochskalieren) …", self.ai_enhance, "Ctrl+Alt+E")
        act(e, "Weißabgleich-Pipette", lambda: self.picker_btn.toggle(), "W")
        act(e, "Alles zurücksetzen", self.reset_all, "Ctrl+Shift+R")

        v = mb.addMenu("&Ansicht")
        act(v, "Vorher / Nachher", self.toggle_before, "\\")
        act(v, "Einpassen", self.view.fit_view, "Ctrl+0")
        act(v, "100 % (Vorschau)", self.view.zoom_100, "Ctrl+1")
        act(v, "Nächstes Bild", lambda: self._step(1), "Right")
        act(v, "Vorheriges Bild", lambda: self._step(-1), "Left")

        opt = mb.addMenu("&Einstellungen")
        self.a_gpu = QAction("GPU-Beschleunigung (OpenCL)", self, checkable=True)
        self.a_gpu.setStatusTip("Rechenintensive Filter (Hintergrund entfernen) auf der Grafikkarte ausführen")
        self.a_gpu.setChecked(bool(self.settings.get("gpu", True)))
        self.a_gpu.toggled.connect(self._on_gpu_toggled)
        opt.addAction(self.a_gpu)
        gpu.set_enabled(self.a_gpu.isChecked())
        a_rt = QAction("GPU-Laufzeiten und Treiber …", self)
        a_rt.setStatusTip("Herstellerspezifische GPU-Unterstützung für NVIDIA (CUDA) und AMD (ROCm) nachladen")
        a_rt.triggered.connect(self.gpu_runtime_dialog)
        opt.addAction(a_rt)
        self.accel_menu = opt.addMenu("KI-Beschleuniger (NPU / GPU / CPU)")
        self.accel_menu.aboutToShow.connect(self._fill_accel_menu)
        from .ai import engine as ai_engine
        ai_engine.set_preferred(self.settings.get("ai_device", "auto"))

        h = mb.addMenu("&Hilfe")
        act(h, "Über…", self.about)

    def _set_enabled(self, on: bool):
        self.panel.setEnabled(on)
        for a in (self.a_export, self.a_close):
            a.setEnabled(on)
        has_items = self.film.count() > 0
        for a in (self.a_export_sel, self.a_close_sel, self.a_close_all):
            a.setEnabled(has_items)

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
        # add to the filmstrip (already open images are not added twice)
        open_now = {os.path.abspath(self.film.item(i).data(Qt.UserRole)) for i in range(self.film.count())}
        first = None
        placeholder = QPixmap(120, 90)
        placeholder.fill(QColor(50, 50, 50))
        self.film.blockSignals(True)
        for fp in files:
            if os.path.abspath(fp) in open_now:
                first = first or self._item_for(fp)
                continue
            open_now.add(os.path.abspath(fp))
            it = QListWidgetItem(QIcon(placeholder), os.path.basename(fp))
            it.setData(Qt.UserRole, fp)
            it.setToolTip(fp)
            it.setSizeHint(QSize(132, 118))
            self.film.addItem(it)
            first = first or it
            self._run(("thumb", fp), self._thumb_job, fp, pool=self.thumb_pool)
        self.film.blockSignals(False)
        self._set_enabled(self.image is not None)
        if first is not None:
            self.film.clearSelection()  # only the newly shown image is selected
            self.film.setCurrentItem(first)
            first.setSelected(True)
            self.film.scrollToItem(first)

    def _item_for(self, path):
        for i in range(self.film.count()):
            if self.film.item(i).data(Qt.UserRole) == path:
                return self.film.item(i)
        return None

    # ------------------------------------------------------------- closing
    def close_current(self):
        if self.image is not None:
            it = self._item_for(self.image.path)
            if it is not None:
                self._close_items([it])

    def close_selected(self):
        items = self.film.selectedItems()
        if not items and self.image is not None:
            items = [self._item_for(self.image.path)]
        self._close_items([i for i in items if i is not None])

    def close_all(self):
        self._close_items([self.film.item(i) for i in range(self.film.count())])

    def _close_items(self, items):
        """Remove images from the filmstrip. Edits stay saved and come back when the
        image is opened again; the files themselves are never touched."""
        if not items:
            return
        self.commit_timer.stop()
        self._commit()
        self._save_current()
        closing = {it.data(Qt.UserRole) for it in items}
        current = self.image.path if self.image is not None else self.current_path
        cur_row = self.film.currentRow()
        self.film.blockSignals(True)
        for it in items:
            self.film.takeItem(self.film.row(it))
        self.film.blockSignals(False)
        if current in closing or current is None:
            self._clear_view()
            if self.film.count():
                self.film.blockSignals(True)
                self.film.setCurrentRow(min(max(cur_row, 0), self.film.count() - 1))
                self.film.blockSignals(False)
                # load explicitly: Qt may already have moved the current row silently
                self._on_film_select(self.film.currentItem(), None)
        n = len(closing)
        self._set_enabled(self.image is not None)
        self.statusBar().showMessage(f"{n} Bild{'er' if n != 1 else ''} geschlossen", 3000)

    def _clear_view(self):
        self.load_token += 1  # ignore loads still running in the background
        QApplication.restoreOverrideCursor()
        self.image = None
        self.current_path = None
        self.params = Params()
        self.history, self.hist_pos = [], -1
        self.show_before = False
        self.view.set_overlay("")
        self.view.set_image(None)
        self.histogram.set_image(None)
        self.curve.set_histogram(None)
        self.info.setText("")
        self.setWindowTitle(APP_TITLE)
        self._apply_params_to_ui(Params())
        self._set_enabled(False)

    def _film_menu(self, pos):
        it = self.film.itemAt(pos)
        menu = QMenu(self)
        if it is not None:
            menu.addAction("Schließen", lambda: self._close_items([it]))
        if len(self.film.selectedItems()) > 1:
            menu.addAction(f"{len(self.film.selectedItems())} ausgewählte schließen", self.close_selected)
        if self.film.count():
            menu.addAction("Alle schließen", self.close_all)
        if not menu.isEmpty():
            menu.exec(self.film.mapToGlobal(pos))

    def _thumb_job(self, fp):
        arr = load_thumbnail(fp, 180)
        p = self.store.get(fp)
        if p is not None and not p.is_default():
            # show edited thumbnail
            from .processing import srgb_to_linear
            rgb, a = Renderer(p).apply_rgba(srgb_to_linear(arr.astype(np.float32) / 255.0))
            if a is not None:
                rgb = composite(rgb, a, checkerboard(*a.shape, size=8))
            arr = to_uint8(rgb)
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
            _, fp = tag
            it = self._item_for(fp)
            if it is not None:
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
        msg = getattr(self, "_pending_status", None)
        self._pending_status = None
        if msg:
            self.statusBar().showMessage(msg, 10000)
        else:
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
        self._sync_key_ui()

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
        if self._item_for(path) is not None:
            self._run(("thumb", path), self._thumb_job, path, pool=self.thumb_pool)

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
        try:
            p = self._auto_color(self.image.linear, self.params, strength)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_TITLE, f"KI-Korrektur fehlgeschlagen:\n{exc}")
            return
        self._set_params(p)
        self.wb_preset.setCurrentText("Auto")
        extra = ""
        if self.ai_method.currentData() == "neural":
            from .ai import engine as ai_engine
            extra = f" – Netz lief auf {ai_engine.describe(ai_engine.last_device('cidnet'))}"
        self.statusBar().showMessage(
            f"KI-Korrektur: Temperatur {p.temp:.0f} K, Tönung {p.tint:+.0f}, Kurven gesetzt{extra}", 6000)

    def _auto_color(self, linear, params, strength):
        if self.ai_method.currentData() == "neural":
            from .auto import neural_color
            return neural_color(linear, params, strength)
        return ai_correct(linear, params, strength)

    def apply_key(self):
        if self.image is None:
            return
        p, info = ai_remove_background(self.image.linear, self.params)
        if p is not None:
            p.key_mode, p.key_mask = 0.0, ""
        if p is None:
            QMessageBox.information(
                self, APP_TITLE,
                "Kein Green- oder Bluescreen erkannt.\n\nDer Hintergrund muss am oberen, linken "
                "und rechten Bildrand überwiegend aus einer kräftig grünen oder blauen Fläche bestehen.")
            return
        self._set_params(p)
        self.statusBar().showMessage(
            f"{CHANNEL_NAMES[info['channel']]}screen entfernt – ca. {info['coverage'] * 100:.0f} % "
            "des Bildes sind jetzt transparent. PNG-Export behält die Transparenz.", 6000)

    def _on_key_toggled(self, on):
        if self._block_key:
            return
        self.params.key_enabled = 1.0 if on else 0.0
        self._sync_key_ui()
        self._request_render()
        self.commit_timer.start()

    def _sync_key_ui(self):
        p = self.params
        ai_mode = p.key_mode == 1.0
        analysed = bool(p.key_mask) if ai_mode else p.key_level > 0
        self._block_key = True
        self.key_enable.setChecked(bool(p.key_enabled) and analysed)
        self._block_key = False
        self.key_enable.setEnabled(analysed)
        for w_ in self.key_controls:
            w_.setEnabled(analysed and bool(p.key_enabled))
        if ai_mode:
            self.s_key_spill.setEnabled(False)  # no screen colour to remove
        if analysed and ai_mode:
            self.key_info.setText("KI-Maske (BiRefNet). Für Transparenz als PNG exportieren.")
        elif analysed:
            name = CHANNEL_NAMES.get(int(p.key_channel), "Grün")
            self.key_info.setText(f"Erkannt: {name}screen. Für Transparenz als PNG exportieren.")
        else:
            self.key_info.setText("Greenscreen: schnell, ohne Download. Jedes Foto: neuronales Netz "
                                  "(einmaliger Download, 224 MB).")

    # ------------------------------------------------------------- AI models
    def _fill_accel_menu(self):
        from .ai import engine as ai_engine
        self.accel_menu.clear()
        devices = ai_engine.discover()
        current = ai_engine.preferred()
        group = []
        entries = [("auto", "Automatisch (beste verfügbare Hardware)")] + \
                  [(d.id, ai_engine.describe(d)) for d in devices]
        for dev_id, text in entries:
            a = QAction(text, self, checkable=True)
            a.setChecked(dev_id == current)
            a.triggered.connect(lambda _=False, i=dev_id: self._set_accel(i))
            self.accel_menu.addAction(a)
            group.append(a)
        if not devices:
            self.accel_menu.addAction("Keine KI-Laufzeit gefunden").setEnabled(False)

    def _set_accel(self, dev_id):
        from .ai import engine as ai_engine
        ai_engine.set_preferred(dev_id)
        ai_engine.reset()
        self.settings.set("ai_device", dev_id)
        self.statusBar().showMessage(f"KI-Beschleuniger: {dev_id if dev_id != 'auto' else 'Automatisch'}", 4000)

    def _run_with_progress(self, title, fn, on_done, determinate=True):
        """Run ``fn(progress, cancelled)`` in the background with a progress dialog."""
        dlg = QProgressDialog(title, "Abbrechen", 0, 1000 if determinate else 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        state = {"cancel": False}
        dlg.canceled.connect(lambda: state.update(cancel=True))
        t = Task(("ai",), fn, None, None)
        sig = t.signals
        t.args = (lambda f, sig=sig: sig.progress.emit(int(f * 1000), ""), lambda: state["cancel"])
        if determinate:
            sig.progress.connect(lambda v, _t: dlg.setValue(min(v, 999)))

        def done(tag, res):
            self._tasks.discard(sig)
            dlg.reset()
            QApplication.restoreOverrideCursor()
            if not state["cancel"]:
                on_done(res)

        def fail(tag, msg):
            self._tasks.discard(sig)
            dlg.reset()
            QApplication.restoreOverrideCursor()
            if not state["cancel"]:
                QMessageBox.warning(self, APP_TITLE, msg)

        sig.done.connect(done)
        sig.failed.connect(fail)
        self._tasks.add(sig)
        self.pool.start(t)

    def _ensure_model(self, model_id, then):
        """Download a model on first use (asks first, shows size and licence)."""
        from .ai import models as ai_models
        if ai_models.is_available(model_id):
            then()
            return
        m = ai_models.MODELS[model_id]
        size = m.download_size / 1e6
        if QMessageBox.question(
                self, APP_TITLE,
                f"Für diese Funktion wird das KI-Modell „{m.title}“ benötigt.\n\n"
                f"Download: {size:.0f} MB (einmalig)\nLizenz: {m.license}\nQuelle: {m.source}\n\n"
                "Jetzt herunterladen?") != QMessageBox.Yes:
            return

        def job(progress, cancelled):
            ai_models.download(model_id, lambda d, t: (progress(d / max(t, 1)), not cancelled())[1])
            return True

        self._run_with_progress(f"Lade {m.title} ({size:.0f} MB) …", job, lambda _r: then())

    # ------------------------------------------------------- CUDA / ROCm runtimes
    @staticmethod
    def _rt_size_text(offer, items):
        if items is None:
            return "Größe unbekannt (keine Verbindung zu PyPI)"
        if offer.profile.pinned:  # ROCm 10: the whole AMD user space comes along
            gb = lambda v: f"{v:.1f}".replace(".", ",")  # noqa: E731
            total = sum(i["size"] for i in items) / 1e9
            return (f"ca. {gb(total)} GB von AMD ({offer.target}); entpackt ca. {gb(offer.profile.disk_gb)} GB im "
                    "Benutzerordner – am System wird nichts verändert")
        rt = items[0]["size"] / 1e6
        libs = sum(i["size"] for i in items[1:]) / 1e6
        txt = f"ca. {rt:.0f} MB"
        if libs:
            gb = f"{libs / 1000:.1f}".replace(".", ",")
            txt += f" + {gb} GB {offer.profile.vendor.upper()}-Bibliotheken (sie sind nicht installiert)"
        return txt

    def _check_gpu_runtimes(self):
        """Startup: offer CUDA / ROCm support once when the drivers are found."""
        from . import runtimes

        def job(progress, cancelled):
            res = []
            try:
                from . import drivers
                self._driver_issues = (drivers.find_issues(), drivers.read_os())
            except Exception:  # noqa: BLE001
                self._driver_issues = ([], None)
            for o in runtimes.detect():
                items = None
                if o.possible and o.profile.id not in runtimes.installed():
                    try:
                        items = runtimes.plan(o)
                    except Exception:
                        items = None
                res.append((o, items))
            return res

        t = Task(("rt",), job, None, None)
        sig = t.signals
        sig.done.connect(lambda _tag, res: (self._tasks.discard(sig), self._offer_gpu_runtimes(res)))
        sig.failed.connect(lambda *_a: self._tasks.discard(sig))
        self._tasks.add(sig)
        self.pool.start(t)

    def _offer_gpu_runtimes(self, results):
        from . import runtimes
        installed = runtimes.installed()
        if self._offer_driver_guides():
            return  # one message per start
        for offer, items in results:
            vendor = offer.profile.vendor
            key = f"gpu_runtime_asked_{vendor}"
            if self.settings.get(key) or any(runtimes.PROFILES[p].vendor == vendor for p in installed):
                continue
            if offer.quiet:
                continue  # e.g. an AMD GPU that ROCm does not support: Vulkan is used, nothing to do
            self.settings.set(key, True)  # ask only once - the settings menu remains
            name = "NVIDIA-CUDA" if vendor == "nvidia" else "AMD-ROCm"
            later = ("\n\nWenn Sie das jetzt nicht tun, können Sie es jederzeit unter "
                     "Einstellungen → GPU-Laufzeiten und Treiber nachholen.")
            gpus = f" ({', '.join(offer.gpus)})" if offer.gpus else ""
            if not offer.possible:
                issues, osi = getattr(self, "_driver_issues", ([], None))
                rocm_issue = next((i for i in issues if i.id == "rocm"), None) if vendor == "amd" else None
                box = QMessageBox(self)
                box.setWindowTitle("GPU-Unterstützung")
                box.setIcon(QMessageBox.Information)
                box.setText(f"Auf diesem Computer wurde {'eine AMD-Grafik' if vendor == 'amd' else f'ein {name}-Treiber'}"
                            f" gefunden{gpus}.\n\n{offer.reason}.")
                box.setInformativeText(
                    (f"{offer.profile.install_hint}\n\n" if offer.profile.install_hint else "")
                    + ("Die Anleitung zeigt, wie Sie das mit einem Befehl einrichten." if rocm_issue else "")
                    + later)
                show = box.addButton("Anleitung anzeigen", QMessageBox.AcceptRole) if rocm_issue else None
                box.addButton("OK" if show is None else "Nicht jetzt", QMessageBox.RejectRole)
                box.exec()
                if show is not None and box.clickedButton() is show:
                    self.show_driver_guide(rocm_issue, osi)
                return  # one message per start
            box = QMessageBox(self)
            box.setWindowTitle("GPU-Unterstützung verfügbar")
            box.setIcon(QMessageBox.Question)
            found = (f"eine AMD-Grafik gefunden, die ROCm 10 unterstützt{gpus}" if vendor == "amd"
                     else f"{name}-Treiber gefunden{gpus}")
            box.setText(f"Auf diesem Computer wurde{'n' if vendor != 'amd' else ''} {found}.\n\n"
                        f"Möchten Sie die {name.split('-')[1]}-Unterstützung nachladen?")
            box.setInformativeText(
                f"{offer.profile.title}: Die KI-Funktionen laufen damit über die für Ihre Grafikkarte "
                f"optimierten Bibliotheken – meist deutlich schneller.\n\nDownload: {self._rt_size_text(offer, items)}"
                + later)
            yes = box.addButton("Nachladen", QMessageBox.AcceptRole)
            box.addButton("Nicht jetzt", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is yes:
                self._install_gpu_runtime(offer)
            return  # one question per start

    def _offer_driver_guides(self) -> bool:
        """A compatible GPU/NPU without driver: offer an installation guide (once per issue)."""
        issues, osi = getattr(self, "_driver_issues", ([], None))
        for issue in issues:
            if issue.optional:
                continue  # optional extras only appear in the settings dialog
            key = f"driver_guide_asked_{issue.id}"
            if self.settings.get(key):
                continue
            self.settings.set(key, True)
            box = QMessageBox(self)
            box.setWindowTitle("Treiber fehlt")
            box.setIcon(QMessageBox.Information)
            box.setText(issue.message)
            box.setInformativeText(
                f"{issue.benefit}\n\nMöchten Sie eine Installationsanleitung für "
                f"{osi.name if osi else 'Ihr System'} anzeigen?\n\nSie finden die Anleitung auch jederzeit "
                "unter Einstellungen → GPU-Laufzeiten und Treiber.")
            show = box.addButton("Anleitung anzeigen", QMessageBox.AcceptRole)
            box.addButton("Nicht jetzt", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is show:
                self.show_driver_guide(issue, osi)
            return True
        return False

    def show_driver_guide(self, issue, osi=None):
        from PySide6.QtGui import QDesktopServices, QGuiApplication
        from PySide6.QtCore import QUrl
        from . import drivers
        osi = osi or drivers.read_os()
        g = drivers.guide(issue, osi)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Installationsanleitung – {g.title}")
        lay = QVBoxLayout(dlg)
        head = QLabel(f"<b>{g.title}</b><br><span style='color:#999'>Erkanntes System: {g.os_label}</span>")
        head.setWordWrap(True)
        lay.addWidget(head)
        intro = QLabel(g.intro.replace("\n", "<br>"))
        intro.setWordWrap(True)
        lay.addWidget(intro)
        mono = QApplication.font()
        mono.setFamily("monospace")
        for i, st in enumerate(g.steps, 1):
            t = QLabel(f"<b>{i}.</b> {st.text}")
            t.setWordWrap(True)
            lay.addWidget(t)
            if st.command:
                row = QHBoxLayout()
                cmd = QLineEdit(st.command)
                cmd.setReadOnly(True)
                cmd.setFont(mono)
                cmd.setCursorPosition(0)
                cp = QPushButton("Kopieren")
                cp.clicked.connect(lambda _=False, c=st.command: (QGuiApplication.clipboard().setText(c),
                                                                  self.statusBar().showMessage("Befehl kopiert", 2000)))
                row.addWidget(cmd, 1)
                row.addWidget(cp)
                lay.addLayout(row)
        for n in g.notes:
            lbl = QLabel(f"ℹ️ {n}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#bbb")
            lay.addWidget(lbl)
        if g.links:
            links = QLabel(" · ".join(f"<a href='{u}' style='color:#6aa0ff'>{t}</a>" for t, u in g.links))
            links.setOpenExternalLinks(False)
            links.linkActivated.connect(lambda u: QDesktopServices.openUrl(QUrl(u)))
            lay.addWidget(links)
        bb = QDialogButtonBox()
        bb.addButton("Schließen", QDialogButtonBox.RejectRole)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.setMinimumWidth(720)
        dlg.adjustSize()
        dlg.exec()

    def _install_gpu_runtime(self, offer):
        from . import runtimes

        def job(progress, cancelled):
            runtimes.install(offer, lambda f: (progress(f), not cancelled())[1])
            return True

        def done(_r):
            box = QMessageBox(self)
            box.setWindowTitle(APP_TITLE)
            box.setText(f"{offer.profile.title} wurde installiert.\n\nSie wird nach einem Neustart der App verwendet.")
            now = box.addButton("Jetzt neu starten", QMessageBox.AcceptRole)
            box.addButton("Später", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is now:
                self._restart()

        self._run_with_progress(f"Lade {offer.profile.title} …", job, done)

    def _restart(self):
        from PySide6.QtCore import QProcess
        if getattr(sys, "frozen", False):
            QProcess.startDetached(sys.executable, sys.argv[1:])
        else:
            QProcess.startDetached(sys.executable, ["-m", "simon_enhance"] + sys.argv[1:])
        self.close()

    def gpu_runtime_dialog(self):
        from . import runtimes
        dlg = QDialog(self)
        dlg.setWindowTitle("GPU-Laufzeiten und Treiber")
        lay = QVBoxLayout(dlg)
        intro = QLabel("Die App nutzt jede Grafikkarte über Vulkan und die Intel-NPU über OpenVINO. "
                       "Für NVIDIA- und AMD-Karten gibt es zusätzlich herstellerspezifische Laufzeiten "
                       "(CUDA bzw. ROCm), die meist deutlich schneller sind. Sie werden nur bei Bedarf "
                       "geladen und nutzen die installierten Treiber.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        status = QLabel(runtimes.status())
        status.setStyleSheet("color:#aaa")
        lay.addWidget(status)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            offers = runtimes.detect()
        finally:
            QApplication.restoreOverrideCursor()
        installed = runtimes.installed()
        from . import drivers
        try:
            issues, osi = drivers.find_issues(), drivers.read_os()
        except Exception:  # noqa: BLE001
            issues, osi = [], None
        if issues:
            hdr = QLabel("<b>Fehlende Treiber</b>" + (f" <span style='color:#999'>– {osi.name}</span>" if osi else ""))
            lay.addWidget(hdr)
        for issue in issues:
            row = QHBoxLayout()
            lbl = QLabel(f"{issue.title}<br><span style='color:#999'>{issue.message}</span>")
            lbl.setWordWrap(True)
            row.addWidget(lbl, 1)
            b = QPushButton("Anleitung anzeigen")
            b.clicked.connect(lambda _=False, i=issue: self.show_driver_guide(i, osi))
            row.addWidget(b)
            lay.addLayout(row)
        if issues:
            lay.addWidget(QLabel("<b>GPU-Laufzeiten</b>"))
        if not offers and not installed:
            lay.addWidget(QLabel("Keine NVIDIA-CUDA-Treiber und keine AMD-Grafik mit Rechentreiber (/dev/kfd) gefunden."))
        for o in offers:
            row = QHBoxLayout()
            gpus = f" – {', '.join(o.gpus)}" if o.gpus else ""
            lbl = QLabel(f"<b>{o.profile.title}</b>{gpus}<br><span style='color:#999'>{o.reason}"
                         + ("" if o.possible else f"<br>{o.profile.install_hint}") + "</span>")
            lbl.setWordWrap(True)
            row.addWidget(lbl, 1)
            if o.possible and o.profile.id not in installed:
                b = QPushButton("Nachladen …")
                b.clicked.connect(lambda _=False, o=o: (dlg.accept(), self._install_gpu_runtime(o)))
                row.addWidget(b)
            lay.addLayout(row)
        for pid in installed:
            row = QHBoxLayout()
            act = " (aktiv)" if runtimes.active() == pid else ""
            row.addWidget(QLabel(f"Installiert: <b>{runtimes.PROFILES[pid].title}</b>{act}"), 1)
            b = QPushButton("Entfernen")

            def rm(_=False, pid=pid):
                runtimes.remove(pid)
                dlg.accept()
                QMessageBox.information(self, APP_TITLE, "Entfernt. Nach einem Neustart wird wieder die "
                                                         "mitgelieferte Laufzeit (Vulkan) verwendet.")
            b.clicked.connect(rm)
            row.addWidget(b)
            lay.addLayout(row)
        bb = QDialogButtonBox()
        bb.addButton("Schließen", QDialogButtonBox.RejectRole)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.setMinimumWidth(640)
        dlg.adjustSize()
        dlg.exec()

    def apply_ai_background(self):
        if self.image is None:
            return
        self._ensure_model("birefnet", self._run_ai_background)

    def _run_ai_background(self):
        from .ai import features as ai_features
        image = self.image
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("KI erkennt das Motiv …")

        def job(progress, cancelled):
            return ai_features.compute_mask(image.path, image.linear)

        def done(key):
            if self.image is not image:
                return
            p = self.params.copy()
            p.key_enabled, p.key_mode, p.key_mask = 1.0, 1.0, key
            self._set_params(p)
            self.statusBar().showMessage(
                f"Hintergrund entfernt (BiRefNet) – berechnet auf {self._ai_device_text('birefnet')}. "
                "PNG-Export behält die Transparenz.", 8000)

        self._run_with_progress("Hintergrund wird erkannt …", job, done, determinate=False)

    def ai_enhance(self):
        if self.image is None:
            return
        denoise = self.enh_denoise.isChecked()
        scale = int(self.enh_scale.currentData())
        if not denoise and scale == 1:
            QMessageBox.information(self, APP_TITLE, "Bitte Entrauschen und/oder Hochskalieren auswählen.")
            return
        needed = (["scunet"] if denoise else []) + (["realesr"] if scale > 1 else [])

        def chain(i=0):
            if i < len(needed):
                self._ensure_model(needed[i], lambda: chain(i + 1))
            else:
                self._run_enhance(denoise, scale)
        chain()

    def _run_enhance(self, denoise, scale):
        from .ai import features as ai_features
        src = self.image.path
        w, h = self.image.full_size
        if w * h * scale * scale > ai_features.MAX_OUTPUT_PIXELS:
            QMessageBox.warning(self, APP_TITLE, f"Das Ergebnis wäre {w * scale}×{h * scale} Pixel – zu groß. "
                                "Bitte eine kleinere Vergrößerung wählen.")
            return
        strength = self.s_enh_strength.value() / 100.0 if denoise else 0.0
        dst = ai_features.enhance_output_path(src, denoise, scale)
        params = self.params.copy()
        if params.key_mode == 1.0:  # the AI mask belongs to the old file
            params.key_enabled, params.key_mask = 0.0, ""

        def job(progress, cancelled):
            try:
                return ai_features.enhance(src, dst, strength, scale, progress, cancelled)
            except InterruptedError:
                if os.path.exists(dst):
                    os.remove(dst)
                raise

        def done(path):
            self.store.set(path, params)
            used = self._ai_device_text("scunet" if denoise else "realesr")
            self._pending_status = f"Gespeichert: {os.path.basename(path)} – berechnet auf {used}"
            self.open_paths([path])

        self._run_with_progress(f"KI-Verbessern: {os.path.basename(src)} …", job, done)

    @staticmethod
    def _ai_device_text(model_id: str) -> str:
        """Device the model ran on - and why a GPU was skipped, if it ran on the CPU."""
        from .ai import engine as ai_engine
        dev = ai_engine.last_device(model_id)
        txt = ai_engine.describe(dev)
        if dev is not None and dev.kind == "cpu":
            why = ai_engine.fallback_reasons(model_id)
            if why:
                txt += " – Grafikkarte nicht genutzt: " + "; ".join(f"{l}: {r}" for l, r in why[:2])
        return txt

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
                p = self._auto_color(data.linear, base, strength)
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
        rgb, alpha = Renderer(p).apply_rgba(self.image.linear, self.image.alpha)
        hist8 = to_uint8(rgb if alpha is None else rgb[alpha > 0.5][None])
        if alpha is not None:
            h, w = alpha.shape
            rgb = composite(rgb, alpha, checkerboard(h, w, max(8, round(max(h, w) / 120))))
        img8 = to_uint8(rgb)
        self.view.set_image(img8, refit=refit)
        self.histogram.set_image(hist8 if hist8.size else img8)
        self.curve.set_histogram(hist8 if hist8.size else img8)

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

        prog = QProgressDialog("Exportiere …", "Abbrechen", 0, len(jobs) * 100, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        state = {"i": 0, "errors": []}

        def next_job():
            i = state["i"]
            if i >= len(jobs) or prog.wasCanceled():
                prog.setValue(len(jobs) * 100)
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
            prog.setLabelText(f"Exportiere {os.path.basename(fp)} ({i + 1}/{len(jobs)}) …")
            t = Task(("export", i), export_image, fp, params, dst, opts)
            sig = t.signals
            t.args = (*t.args, lambda f, sig=sig: sig.progress.emit(int(f * 100), ""))
            sig.progress.connect(lambda v, _txt, i=i: prog.setValue(i * 100 + min(v, 99)))

            def ok(tag, res):
                state["i"] += 1
                prog.setValue(state["i"] * 100)
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
            f"Modus: {'Portabel' if is_portable() else 'Installiert'}<br>"
            f"{gpu.status()}</p>")

    def _on_gpu_toggled(self, on):
        gpu.set_enabled(on)
        self.settings.set("gpu", bool(on))
        self.statusBar().showMessage(gpu.status(), 5000)

    def _report_gpu(self):
        # runs once shortly after start: OpenCL driver discovery can take a moment
        self.a_gpu.setToolTip(gpu.status())
        if self.a_gpu.isChecked() and gpu.available():
            self.statusBar().showMessage(gpu.status(), 5000)

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
    # a downloaded CUDA / ROCm ONNX Runtime must be hooked in before anything imports it
    diag = ("--ai-info", "--gpu-info", "--net-check", "--version")
    if any(f in argv for f in diag) and (sys.stdout is None or os.environ.get("SIE_DIAG_FILE")):
        # Windows build has no console: write the diagnostics to a file instead
        from .store import data_dir
        path = os.environ.get("SIE_DIAG_FILE") or os.path.join(data_dir(), "diagnose.txt")
        sys.stdout = sys.stderr = open(path, "w", encoding="utf-8", buffering=1)
    from . import runtimes
    runtimes.activate()
    if "--ai-probe" in argv:  # child process of the crash-safe device check (see ai/engine.py)
        from .ai import engine as ai_engine
        i = argv.index("--ai-probe")
        return ai_engine.probe_main(argv[i + 1:i + 4])
    if "--gpu-info" in argv:  # quick diagnostics: is the GPU usable?
        print(gpu.status())
        return 0 if gpu.available() else 1
    if "--ai-info" in argv:  # diagnostics: which accelerators can run which AI model?
        from .ai import engine as ai_engine, models as ai_models
        print(f"{APP_TITLE} {__version__}")
        print(gpu.status())
        print(runtimes.status())
        for o in runtimes.detect():
            print(f"Erkannt: {o.profile.title} – {o.reason}" + ("" if o.possible else " (nicht nachladbar)"))
        if "--download" in argv:
            from . import net
            for k in ai_models.MODELS:
                if not ai_models.is_available(k):
                    try:
                        ai_models.download(k)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Download {k} fehlgeschlagen: {net.describe_error(exc)}")
        models = [k for k in ai_models.MODELS if ai_models.is_available(k)]
        devices = ai_engine.discover()
        print("KI-Beschleuniger (jedes Gerät wird in einem eigenen Prozess geprüft):")
        for d in devices:
            print(f"  {ai_engine.describe(d)}")
            for k in models:
                if d.kind == "cpu":
                    print(f"      {k:9s} Referenz")
                    continue
                if d.runtime == "ncnn" and not ai_models.MODELS[k].ncnn:
                    continue
                r = ai_engine._run_probe(k, d)
                if r.get("ok"):
                    extra = f" ({r['seconds']} s erster Durchlauf)" if r.get("seconds") is not None else ""
                    print(f"      {k:9s} funktioniert{extra}")
                else:
                    print(f"      {k:9s} NICHT nutzbar: {r.get('reason')}")
        print("Modelle:", ", ".join(f"{m.title}: {'installiert' if ai_models.is_available(k) else 'wird bei Bedarf geladen'}"
                                     for k, m in ai_models.MODELS.items()))
        if "--download" in argv and ai_models.is_available("scunet"):  # end-to-end: denoise a photo
            import tempfile
            import numpy as _np
            from PIL import Image as _Image
            from .ai import features as ai_features
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "test.png")
                rng = _np.random.default_rng(1)
                yy, xx = _np.mgrid[0:300, 0:400] / 300.0
                img = _np.stack([0.5 + 0.3 * _np.sin(xx * 6), 0.4 + 0.3 * yy, 0.5 + 0.2 * _np.cos(yy * 5)], -1)
                img = _np.clip(img + rng.normal(0, 0.05, img.shape), 0, 1)
                _Image.fromarray((img * 255).astype(_np.uint8)).save(src)
                dst = ai_features.enhance_output_path(src, True, 1)
                try:
                    ai_features.enhance(src, dst, 0.8, 1)
                    print(f"Entrauschen (Ende-zu-Ende): OK – {os.path.getsize(dst)} Bytes, "
                          f"berechnet auf {ai_engine.describe(ai_engine.last_device('scunet'))}")
                    for label, why in ai_engine.fallback_reasons("scunet"):
                        print(f"  übersprungen: {label}: {why}")
                except Exception as exc:  # noqa: BLE001
                    print(f"Entrauschen (Ende-zu-Ende): FEHLER – {exc}")
                    return 1
        return 0
    if "--net-check" in argv:  # diagnostics: do HTTPS downloads work (certificates)?
        from . import net
        from .ai import models as ai_models
        print("Zertifikate:", net.source())
        urls = ["https://pypi.org/simple/onnxruntime/", f"{runtimes.AMD_REPO}/core/whl-next/rocm-sdk-core/",
                next(f.url for m in ai_models.MODELS.values() for f in m.files if f.url)]
        failed = 0
        for u in urls:
            try:
                with net.urlopen(u, timeout=30) as r:
                    r.read(1)
                print("OK    ", u)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FEHLER", u, "–", net.describe_error(exc).splitlines()[0])
        return 1 if failed else 0
    if "--version" in argv:
        print(f"{APP_TITLE} {__version__}")
        return 0
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
