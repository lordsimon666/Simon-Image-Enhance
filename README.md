# Simon Image Enhance

Fotobearbeitung im Stil von Lightroom – für **Windows** und **Linux**.
Nicht-destruktiv: Die Originale werden nie verändert, alle Einstellungen pro Bild werden
gespeichert und erst beim Export angewendet.

![Icon](packaging/icons/icon-128.png)

## Funktionen

| Bereich | Was es kann |
|---|---|
| **Weißabgleich** | Temperatur (Kelvin) & Tönung, Pipette (auf neutrale Stelle klicken), Voreinstellungen (Tageslicht, Bewölkt, Kunstlicht …) |
| **Gradationskurve** | Interaktive Kurven für RGB, Rot, Grün, Blau (Klick = Punkt, Ziehen = verschieben, Rechtsklick/Doppelklick = löschen) |
| **KI-Korrektur** | Analysiert das Bild und setzt **nur Weißabgleich und Gradationskurven** automatisch. Stärke einstellbar, danach alles feinjustierbar. Auch für mehrere Bilder auf einmal. |
| **Tonwert / Präsenz** | Belichtung, Kontrast, Lichter, Tiefen, Weiß, Schwarz, Dynamik, Sättigung |
| **Import** | JPEG, PNG (8/16 Bit, Transparenz), TIFF, WebP und **RAW** (DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2, PEF, SRW … über LibRaw) |
| **Export** | JPEG (Qualität, EXIF) und PNG (8/16 Bit), Größe begrenzen, Stapel-Export |
| **Workflow** | Filmstreifen (Ordner öffnen), Vorher/Nachher (`\`), Rückgängig/Wiederholen, Einstellungen kopieren & auf mehrere Bilder einfügen, Drag & Drop, Histogramm |

### Wie die KI-Korrektur arbeitet

Läuft komplett offline auf deinem Rechner, ohne Cloud.

1. **Weißabgleich:** Sucht Flächen im Bild, deren Farbe auf der Kurve physikalisch möglicher
   Lichtquellen liegt (Schwarzkörper 2000–14000 K) – das sind fast immer neutrale Oberflächen
   (Wände, Wolken, Straße, Papier). Das Ergebnis wird je nach Sicherheit mit klassischen
   Farbkonstanz-Verfahren kombiniert (Gray-World, Shades-of-Gray, Gray-Edge, White-Patch).
   Bei sehr warmem/kaltem Licht (Sonnenuntergang, Kerzenlicht) bleibt etwas Stimmung erhalten.
2. **Kurven:** Aus dem Histogramm werden Schwarz-/Weißpunkt, Mitteltöne und ein angepasster
   S-Kontrast bestimmt; Farbstiche in Tiefen und Lichtern werden über die R/G/B-Kurven entfernt.

## Downloads

Fertige Pakete erzeugt GitHub Actions bei jedem Push (Tab **Actions** → Artefakte) und bei
Tags `v*` automatisch als **Release**:

| Datei | System |
|---|---|
| `SimonImageEnhance-<ver>-Setup.exe` | Windows-Installer (ohne Admin-Rechte möglich, optional Dateizuordnung) |
| `SimonImageEnhance-<ver>-Windows-Portable.zip` | Windows portabel – entpacken & starten, Einstellungen bleiben im Ordner |
| `simon-image-enhance_<ver>_amd64.deb` | Debian / Ubuntu / Mint |
| `SimonImageEnhance-<ver>-x86_64.AppImage` | Jede Linux-Distribution (`chmod +x` und starten) |
| `SimonImageEnhance.flatpak` | Flatpak (`flatpak install --user SimonImageEnhance.flatpak`) |

Release erstellen:

```bash
git tag v1.0.0
```

```bash
git push origin v1.0.0
```

## Tastenkürzel

| Taste | Aktion |
|---|---|
| `Strg+O` / `Strg+Shift+O` | Bilder / Ordner öffnen |
| `Strg+E` / `Strg+Shift+E` | Exportieren / Auswahl exportieren |
| `Strg+U` / `Strg+Shift+U` | KI-Korrektur / auf Auswahl anwenden |
| `W` | Weißabgleich-Pipette |
| `\` | Vorher / Nachher |
| `Strg+Z` / `Strg+Shift+Z` | Rückgängig / Wiederholen |
| `Strg+Shift+C` / `Strg+Shift+V` | Einstellungen kopieren / einfügen |
| `←` / `→` | Vorheriges / nächstes Bild |
| `Strg+0` / `Strg+1` | Einpassen / 100 % |

Doppelklick auf einen Slider-Namen setzt ihn zurück.

## Aus dem Quellcode starten

```bash
pip install -r requirements.txt
```

```bash
python -m simon_enhance
```

Tests:

```bash
pip install pytest && python -m pytest -q
```

## Selbst bauen

Alles basiert auf einem PyInstaller-Bundle (`packaging/simon_enhance.spec`).

**Windows** (PowerShell, Python 3.10+ und [Inno Setup 6](https://jrsoftware.org/isinfo.php)):

```powershell
.\packaging\windows\build.ps1
```

**Linux:**

```bash
pip install -r requirements.txt pyinstaller && pyinstaller packaging/simon_enhance.spec --noconfirm
```

```bash
packaging/linux/build_deb.sh dist/SimonImageEnhance 1.0.0 dist
```

```bash
packaging/linux/build_appimage.sh dist/SimonImageEnhance 1.0.0 dist
```

```bash
tar -C dist/SimonImageEnhance -czf packaging/flatpak/SimonImageEnhance-linux.tar.gz . && flatpak-builder --user --install --force-clean build-flatpak packaging/flatpak/io.github.lordsimon666.SimonImageEnhance.yml
```

## Datenablage

Bearbeitungen werden als JSON gespeichert (Originale bleiben unberührt):

- Windows: `%APPDATA%\SimonImageEnhance\edits`
- Linux: `~/.local/share/SimonImageEnhance/edits`
- Portabel: Ordner `data` neben der `.exe`

## Projektstruktur

```
simon_enhance/
  processing.py   Bildpipeline (Weißabgleich, Tonwerte, Kurven, LUTs)
  auto.py         KI-Korrektur (Weißabgleich + Kurven)
  fileio.py       Laden (JPEG/PNG/RAW) und Export (JPEG/PNG)
  store.py        Speicherung der Bearbeitungen, Portabel-Modus
  widgets.py      Kurveneditor, Histogramm, Slider, Bildansicht
  app.py          Hauptfenster
packaging/        PyInstaller, Inno Setup, .deb, AppImage, Flatpak, Icons
.github/workflows/build.yml   CI: Tests + alle Pakete + Release
```
