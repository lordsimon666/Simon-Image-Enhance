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
| **KI-Korrektur** | Setzt **nur Weißabgleich und Gradationskurven** automatisch – wahlweise per physikalischer *Analyse* (neutral, empfohlen) oder per *neuronalem Netz* („Profi-Look“, trainiert auf 5000 von Profis bearbeiteten Fotos). Stärke einstellbar, danach alles feinjustierbar. Auch für mehrere Bilder auf einmal. |
| **Hintergrund entfernen (KI)** | Erkennt Green- und Bluescreens automatisch und macht sie transparent – auch bei Falten, Schatten und ungleichmäßiger Ausleuchtung. Regler für Toleranz, Weichheit, Kante und Entfärben (entfernt grüne Reflexe). PNG-Export speichert die Transparenz (8 und 16 Bit), JPEG bekommt eine wählbare Hintergrundfarbe. |
| **Hintergrund entfernen – jedes Foto** | Neuronales Netz (BiRefNet) stellt Personen und Objekte frei, ganz ohne Greenscreen; Kanten werden in voller Auflösung verfeinert. Export als transparentes PNG. |
| **KI-Verbessern** | Entrauschen (SCUNet) und Hochskalieren 2×/4× (Real-ESRGAN) des Originals in eine **neue 16-Bit-TIFF-Datei** – wie „Verbessern“ in Lightroom |
| **Tonwert / Präsenz** | Belichtung, Kontrast, Lichter, Tiefen, Weiß, Schwarz, Dynamik, Sättigung |
| **Import** | JPEG, PNG (mit Transparenz; 16-Bit-PNGs werden derzeit mit 8 Bit gelesen), TIFF, WebP, Fotos bis 1 Gigapixel (z. B. 200-MP-Handykameras) und **RAW** (DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2, PEF, SRW … über LibRaw) |
| **Export** | JPEG (Qualität, EXIF) und PNG (8/16 Bit), Größe begrenzen, Stapel-Export |
| **Workflow** | Filmstreifen (Ordner öffnen, Bilder hinzufügen und schließen), Vorher/Nachher (`\`), Rückgängig/Wiederholen, Einstellungen kopieren & auf mehrere Bilder einfügen, Drag & Drop, Histogramm |

### Wie die KI-Korrektur arbeitet

Läuft komplett offline auf deinem Rechner, ohne Cloud.

1. **Weißabgleich:** Sucht Flächen im Bild, deren Farbe zu einer physikalisch möglichen
   Lichtquelle passt (Schwarzkörper 2000–14000 K, leichte Grün/Magenta-Abweichung erlaubt) –
   das sind fast immer neutrale Oberflächen (Wände, Wolken, Straße, Papier). Gewählt wird die
   *dominante* Lichtquelle, nicht ein Mittelwert. Stark gesättigte Bereiche (Monitore, LEDs)
   werden ignoriert. Wie stark korrigiert wird, hängt von der Sicherheit ab: Bei Mischlicht
   oder wenig neutralen Flächen korrigiert die KI vorsichtiger, und bei sehr warmem/kaltem
   Licht (Sonnenuntergang, Kerzenlicht) bleibt etwas Stimmung erhalten.
**Methode „Neuronales Netz – Profi-Look“:** Das Netz HVI-CIDNet (trainiert auf dem
MIT-Adobe-FiveK-Datensatz) erzeugt eine „Profi-Version“ des Fotos. Daraus werden der
Weißabgleich und die RGB-/Kanal-Kurven berechnet, die dieses Ergebnis nachbilden – die
Korrektur bleibt also auf Weißabgleich und Kurven beschränkt und voll editierbar.
Gemessen an Fotos ist das ein *Stil* (kräftiger, etwas wärmer), keine neutralere
Korrektur; für reines Entfernen von Farbstichen ist die Analyse besser.

2. **Kurven:** Aus dem Histogramm werden Schwarz-/Weißpunkt, Mitteltöne (behutsam, damit
   Abend- und Low-Key-Bilder ihre Stimmung behalten) und ein angepasster S-Kontrast bestimmt.
   Farbstiche in Tiefen und Lichtern werden über die R/G/B-Kurven entfernt – gemessen nur an
   neutralen Bildbereichen, damit z. B. ein blauer Himmel nicht als Farbstich gilt.

### Wie der Greenscreen-Entferner arbeitet

1. Am oberen, linken und rechten Bildrand wird geprüft, ob dort eine kräftig grüne oder
   blaue Fläche ist; Farbe und Stärke des Schirms werden gemessen (inkl. Falten/Schatten).
2. Für jeden Pixel wird der *relative* Grün- bzw. Blauüberschuss berechnet – unabhängig von der
   Helligkeit, daher stören Falten und Schatten im Stoff nicht. Die Schwellwerte kommen aus der
   Statistik des gemessenen Schirms.
3. Die Maske wird kantenbewusst verfeinert (Guided Filter), einzelne Fehlpixel und grüne
   Reflexe im Motiv werden gefüllt, Einzelpixel im Hintergrund entfernt.
4. Halbtransparente Kanten werden mit der *lokalen* Hintergrundfarbe entmischt, verbleibende
   grüne Reflexe auf Haut und Kleidung entfärbt (Despill).

Die KI-Farbkorrektur analysiert bei entferntem Hintergrund nur noch das Motiv.

## KI-Funktionen und Beschleuniger (NPU, Tensor-Kerne, Vulkan)

Die neuronalen Netze laufen auf der stärksten verfügbaren Hardware; kann ein Gerät ein
Modell nicht ausführen, wird automatisch das nächste genutzt (bis zur CPU).
Auswahl: *Einstellungen → KI-Beschleuniger*. Prüfen, was auf deinem Rechner funktioniert:

```bash
simon-image-enhance --ai-info
```

`--ai-info` prüft jedes Modell auf jedem Gerät **in einem eigenen Prozess** und nennt den Grund, falls
ein Gerät nicht geht (Absturz des Treibers, falsches Ergebnis, Fehlermeldung). Dieselbe Prüfung macht
die App vor der ersten Nutzung einer Grafikkarte für ein Modell (einmalig, Ergebnis wird gemerkt): Stürzt
der Grafiktreiber dabei ab, stürzt nicht die App ab, sondern das Gerät wird für dieses Modell
übersprungen. Unter Windows (ohne Konsole) steht die Ausgabe in `%APPDATA%\SimonImageEnhance\diagnose.txt`.

Funktionieren Downloads (Modelle, CUDA/ROCm) nicht, prüft `simon-image-enhance --net-check` die
HTTPS-Verbindungen. Die App nutzt die Zertifikate des Systems (Fedora/Bazzite: `/etc/pki/tls/certs`,
Debian/Ubuntu/Arch: `/etc/ssl/certs`) und als Reserve die mitgelieferten von `certifi`;
eigene Zertifikate (Firmen-Proxy) über `SSL_CERT_FILE`.

| Weg | Hardware | Plattform |
|---|---|---|
| ONNX Runtime + **WebGPU** | jede GPU über den installierten **Vulkan**-Treiber: **AMD**, NVIDIA, Intel | Linux |
| **OpenVINO** | **Intel-NPU** (Core Ultra „AI Boost“), Intel-GPU | Linux |
| ONNX Runtime + **DirectML** | jede DirectX-12-GPU: NVIDIA (**Tensor-Kerne**), AMD, Intel; Intel-NPU | Windows |
| **ncnn + Vulkan** | jede Vulkan-GPU: NVIDIA (**Tensor-Kerne** über *cooperative matrix*), AMD, Intel – auch im Flatpak | Linux, Windows |
| ONNX Runtime CPU | jeder Prozessor | überall |

Alle Wege nutzen die **bereits installierten Treiber** (Mesa/RADV, AMD, NVIDIA, Intel-NPU-Treiber);
herstellerspezifische Rechenbibliotheken wie CUDA oder ROCm werden nicht mitgeliefert.

| Funktion | Modell (Lizenz) | Download | läuft auf |
|---|---|---|---|
| Hintergrund entfernen | BiRefNet-lite (MIT) | 224 MB, einmalig | WebGPU-GPU*, NPU*, DirectML-GPU, CPU |
| Entrauschen | SCUNet real (Apache-2.0/MIT) | 77 MB, einmalig | WebGPU-GPU, NPU*, DirectML-GPU, CPU |
| Hochskalieren | Real-ESRGAN general-x4v3 (BSD-3) | in der App enthalten | Vulkan-GPU (ncnn), WebGPU-GPU, NPU*, DirectML-GPU, CPU |
| KI-Korrektur „Profi-Look“ | HVI-CIDNet FiveK (MIT) | in der App enthalten | WebGPU-GPU, NPU*, DirectML-GPU, CPU |

\* Nicht jedes Gerät kann jedes Modell übersetzen – dann übernimmt automatisch die nächste Hardware.
BiRefNet ließ sich im Test auf einem Software-Vulkan-Treiber nicht über WebGPU ausführen; ob es auf
echten Grafikkarten läuft, zeigt `--ai-info` bzw. die Statusleiste (sonst rechnet die CPU).

### Optional: NVIDIA CUDA und AMD ROCm nachladen (Linux)

Erkennt die App beim Start einen **NVIDIA-Treiber (CUDA)** oder eine von **AMD ROCm 10** unterstützte
Radeon-Grafik, fragt sie einmalig,
ob die herstellerspezifische Laufzeit nachgeladen werden soll – meist deutlich schneller als
Vulkan und mit Tensor- bzw. Matrix-Kernen. Später jederzeit über
*Einstellungen → GPU-Laufzeiten und Treiber* nachholen oder wieder entfernen.

| Laufzeit | Download | Voraussetzung auf dem System |
|---|---|---|
| NVIDIA CUDA 13 | ~256 MB | NVIDIA-Treiber mit CUDA ≥ 13; cuBLAS/cuDNN werden vom System genutzt oder (~1,3 GB) mitgeladen |
| NVIDIA CUDA 12 | ~277 MB | älterer NVIDIA-Treiber (CUDA 12); sonst wie oben |
| AMD ROCm 10.1 – MIGraphX | ~0,95–1,2 GB (je nach GPU), entpackt ~4 GB | nur der amdgpu-Kerneltreiber und Zugriff auf `/dev/kfd` – ROCm selbst muss **nicht** installiert sein |

**ROCm 10:** AMD liefert ROCm seit Version 10 vollständig als Python-Pakete aus. Die App lädt
`rocm-sdk-core`, `rocm-sdk-libraries`, das Paket für die erkannte GPU (`rocm-sdk-device-gfxNNNN`),
`migraphx-libs` und das MIGraphX-Plugin für ONNX Runtime 1.29 (`onnxruntime-ep-migraphx`) von
`stable.repo.amd.com` – als feste Dateien mit fest eingetragenen SHA-256-Prüfsummen – und registriert
das Plugin beim Start. Unterstützt (laut AMD für ONNX Runtime freigegeben): Radeon RX 7000
(gfx1100/1101/1102), RX 9000 (gfx1200/1201) und Ryzen-AI-Grafik (gfx1150–1153). Ältere Karten
(RX 6000, Radeon 780M, Steam Deck) laufen weiter über Vulkan. Die GPU wird über
`/sys/class/kfd` erkannt; fehlt der Zugriff auf `/dev/kfd`, zeigt die App eine Anleitung
(Gruppen `render`/`video` bzw. auf atomaren Systemen wie Bazzite eine udev-Regel). Das Flatpak
hat dafür `--device=all`. Die früheren ROCm-7-Laufzeiten (System-ROCm nötig) entfallen.

Die NVIDIA-Pakete kommen offiziell von PyPI (ONNX Runtime von Microsoft, CUDA-Bibliotheken von NVIDIA),
die AMD-Pakete von AMD; alle werden per SHA-256 geprüft und im Datenordner unter `runtimes`
abgelegt. Der Kernel-Grafiktreiber selbst kommt immer vom System. Lässt sich die nachgeladene Laufzeit nicht starten, nutzt die
App automatisch wieder die mitgelieferte (Vulkan). `SIE_GPU_RUNTIME=0` schaltet sie ab.

### Fehlende Treiber: Installationsanleitung für das eigene System

Findet die App eine Grafikkarte oder NPU, deren Treiber fehlt (z. B. NVIDIA-Karte mit dem freien
„nouveau“-Treiber, kein Vulkan-Treiber, kein Zugriff auf den AMD-Rechentreiber, fehlender Intel-NPU-Treiber), bietet sie
einmalig eine **Installationsanleitung für das erkannte Betriebssystem** an – mit Befehlen zum
Kopieren. Jederzeit erreichbar unter *Einstellungen → GPU-Laufzeiten und Treiber*.

Berücksichtigte Systeme: **Bazzite** und andere Universal-Blue-Images (Bluefin, Aurora), Fedora Atomic (Silverblue, Kinoite),
Fedora, Ubuntu, Linux Mint, Pop!_OS, Debian, Arch, Manjaro, CachyOS, openSUSE; für andere
Systeme allgemeine Hinweise mit Links. Läuft die App als Flatpak, wird das Wirtssystem erkannt.

**Bazzite** (nach der offiziellen Bazzite-Dokumentation):
- NVIDIA-Treiber werden nicht installiert, sondern man wechselt auf das passende Image
  (`brh rebase IMAGE:TAG`, gleichwertig `rpm-ostree rebase ostree-image-signed:docker://ghcr.io/ublue-os/IMAGE:TAG`).
  Die App wählt das Image anhand der Karte: `-nvidia-open` für GTX 16xx/RTX, `-nvidia` für
  GTX 9xx/10xx (Maxwell/Pascal/Volta), behält KDE bzw. GNOME und den Update-Zweig bei und nimmt für
  Bazzite-Deck `bazzite-deck-nvidia` bzw. `bazzite-deck-nvidia-gnome` (ältere Karten: Desktop-Ausgabe).
  Für Kepler-Karten (GTX 7xx) gibt es keinen Treiber mehr – das sagt die Anleitung auch.
- Hinweise zu Secure Boot (`ujust enroll-secure-boot-key`), Rückweg (`brh rollback`), Treiber-Updates
  über die normalen Systemupdates und zu gelayerten Paketen, die einen Rebase blockieren (`rpm-ostree reset`).
- ROCm: Nichts wird ins Image gelayert – die App lädt ROCm 10 selbst; nötig ist nur die udev-Regel für `/dev/kfd`.

Weitere Hinweise:
- **AMD-NPUs** (Ryzen AI / XDNA) werden nicht unterstützt.
- Für die **Intel-NPU** unter Linux muss der Intel-NPU-Treiber installiert sein.
- Modelle werden von Hugging Face geladen (feste Version, SHA-256-geprüft), danach offline nutzbar.

## GPU-Beschleunigung der Filter

Die rechenintensiven Filter der Hintergrundentfernung laufen über **Vulkan** (wgpu) oder
**OpenCL** auf der Grafikkarte. Ohne passende Grafikkarte oder Treiber rechnet die App automatisch auf dem
Prozessor – mit identischem Ergebnis. Ein-/Ausschalten: *Einstellungen → GPU-Beschleunigung*.

Prüfen, ob die Grafikkarte erkannt wird:

```bash
simon-image-enhance --gpu-info
```

| System | Benötigter OpenCL-Treiber |
|---|---|
| Fedora, AMD / Intel | `mesa-libOpenCL` (wird mit dem RPM automatisch mitinstalliert); für Intel alternativ `intel-compute-runtime` |
| Fedora, NVIDIA | NVIDIA-Treiber mit OpenCL, z. B. `xorg-x11-drv-nvidia-cuda-libs` (RPM Fusion) |
| Ubuntu / Debian | `mesa-opencl-icd` (AMD/Intel), `intel-opencl-icd` oder den NVIDIA-Treiber |
| Windows | im normalen Grafikkartentreiber enthalten |

Für Mesa (Rusticl) aktiviert die App die Treiber `radeonsi` und `iris` automatisch
(`RUSTICL_ENABLE`), sofern nichts anderes eingestellt ist. Liefert ein Treiber beim
Selbsttest falsche Ergebnisse, wird automatisch die CPU verwendet. Mit `SIE_GPU=0` lässt
sich die GPU komplett abschalten. Die Flatpak-Version hat im Sandbox-Modus in der Regel
keinen Zugriff auf OpenCL-Treiber und rechnet daher auf der CPU.

## Downloads

Fertige Pakete erzeugt GitHub Actions bei jedem Push (Tab **Actions** → Artefakte) und bei
Tags `v*` automatisch als **Release**:

| Datei | System |
|---|---|
| `SimonImageEnhance-Setup.exe` | Windows-Installer (ohne Admin-Rechte möglich, optional Dateizuordnung) |
| `SimonImageEnhance-<ver>-Windows-Portable.zip` | Windows portabel – entpacken & starten, Einstellungen bleiben im Ordner |
| `simon-image-enhance_amd64.deb` | Debian / Ubuntu / Mint |
| `simon-image-enhance.x86_64.rpm` | Fedora / openSUSE / RHEL (`sudo dnf install ./simon-image-enhance.x86_64.rpm`) |
| `SimonImageEnhance-x86_64.AppImage` | Jede Linux-Distribution (`chmod +x` und starten) |
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
| `Strg+W` / `Strg+Shift+W` | Bild schließen / alle schließen (Rechtsklick im Filmstreifen: weitere Optionen) |
| `Strg+E` / `Strg+Shift+E` | Exportieren / Auswahl exportieren |
| `Strg+U` / `Strg+Shift+U` | KI-Korrektur / auf Auswahl anwenden |
| `Strg+G` / `Strg+Shift+G` | Greenscreen entfernen / Hintergrund entfernen (jedes Foto) |
| `Strg+Alt+E` | KI-Verbessern (Entrauschen / Hochskalieren) |
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
packaging/linux/build_rpm.sh dist/SimonImageEnhance 1.0.0 dist
```

```bash
tar -C dist/SimonImageEnhance -czf packaging/flatpak/SimonImageEnhance-linux.tar.gz . && flatpak-builder --user --install --force-clean build-flatpak packaging/flatpak/io.github.lordsimon666.SimonImageEnhance.yml
```

## Datenablage

Bearbeitungen werden als JSON gespeichert (Originale bleiben unberührt):

- Windows: `%APPDATA%\SimonImageEnhance\edits`
- Linux: `~/.local/share/SimonImageEnhance/edits`
- Portabel: Ordner `data` neben der `.exe`

Im selben Ordner liegen unter `models` die heruntergeladenen KI-Modelle und unter `masks`
die berechneten Freisteller-Masken (beides kann gelöscht werden und wird bei Bedarf neu erzeugt).

## Projektstruktur

```
simon_enhance/
  processing.py   Bildpipeline (Weißabgleich, Tonwerte, Kurven, LUTs)
  auto.py         KI-Korrektur (Weißabgleich + Kurven)
  greenscreen.py  KI-Hintergrundentfernung (Green-/Bluescreen → Transparenz)
  gpu.py          GPU-Beschleunigung der Filter (Vulkan/OpenCL) mit CPU-Rückfall
  vulkan_compute.py  Vulkan-Compute-Shader (wgpu)
  ai/engine.py    KI-Laufzeiten: OpenVINO (NPU), DirectML, CUDA, ncnn-Vulkan, CPU
  ai/models.py    Modellregister, Download mit Prüfsumme
  ai/features.py  Hintergrund-KI, Entrauschen, Hochskalieren
  runtimes.py     optionale CUDA-/ROCm-Laufzeiten: Erkennung, Download, Aktivierung
  drivers.py      fehlende Treiber erkennen, Installationsanleitungen je Betriebssystem
  fileio.py       Laden (JPEG/PNG/RAW) und Export (JPEG/PNG)
  store.py        Speicherung der Bearbeitungen, Portabel-Modus
  widgets.py      Kurveneditor, Histogramm, Slider, Bildansicht
  app.py          Hauptfenster
packaging/        PyInstaller, Inno Setup, .deb, AppImage, Flatpak, Icons
.github/workflows/build.yml   CI: Tests + alle Pakete + Release
```
