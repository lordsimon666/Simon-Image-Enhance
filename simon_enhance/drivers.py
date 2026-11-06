"""Find GPUs / NPUs whose drivers are missing and build installation guides for the
operating system in use (Bazzite and other Universal Blue images, Fedora Atomic,
Fedora, Ubuntu/Mint/Pop!_OS, Debian, Arch family, openSUSE, generic).

Detection looks at the PCI bus (works without any driver) and compares it with what
the app can actually use (NVIDIA driver/libcuda, Vulkan devices, ROCm, Intel compute
runtime, Intel NPU via OpenVINO). AMD ROCm 10 is downloaded by the app itself, so for
AMD only the kernel driver and access to /dev/kfd matter.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

NVIDIA, AMD, INTEL = 0x10DE, 0x1002, 0x8086
_DISPLAY_CLASSES = (0x0300, 0x0302, 0x0380)
_NPU_CLASS = 0x1200  # "processing accelerator" (Intel NPU / VPU)

DOCS = {
    "bazzite": "https://docs.bazzite.gg",
    "bazzite_images": "https://bazzite.gg/#image-picker",
    "bazzite_rebase": "https://docs.bazzite.gg/Installing_and_Managing_Software/Updates_Rollbacks_and_Rebasing/rebase_guide/",
    "bazzite_secure_boot": "https://docs.bazzite.gg/General/Installation_Guide/secure_boot/",
    "bazzite_layering": "https://docs.bazzite.gg/Installing_and_Managing_Software/rpm-ostree/",
    "ublue": "https://universal-blue.org",
    "rpmfusion_nvidia": "https://rpmfusion.org/Howto/NVIDIA",
    "ubuntu_nvidia": "https://ubuntu.com/server/docs/nvidia-drivers-installation",
    "debian_nvidia": "https://wiki.debian.org/NvidiaGraphicsDrivers",
    "arch_nvidia": "https://wiki.archlinux.org/title/NVIDIA",
    "opensuse_nvidia": "https://en.opensuse.org/SDB:NVIDIA_drivers",
    "nvidia": "https://www.nvidia.com/drivers",
    "rocm": "https://rocm.docs.amd.com/en/latest/",
    "intel_npu": "https://github.com/intel/linux-npu-driver",
    "intel_compute": "https://github.com/intel/compute-runtime",
}


# ---------------------------------------------------------------------------
# operating system
# ---------------------------------------------------------------------------

@dataclass
class OSInfo:
    id: str = "unknown"
    like: tuple = ()
    name: str = "Linux"
    version: str = ""
    family: str = "other"         # bazzite | ublue | fedora-atomic | fedora | ubuntu | debian | arch | opensuse | other
    atomic: bool = False          # image based (rpm-ostree / bootc)
    image_name: str = ""          # Universal Blue image, e.g. "bazzite-gnome"
    image_tag: str = "stable"
    image_flavor: str = ""        # "", "nvidia", "nvidia-open", ...
    flatpak: bool = False
    variant: str = ""             # Pop!_OS, Linux Mint, Manjaro, ... (for texts)

    @property
    def nvidia_image(self) -> bool:
        return "nvidia" in self.image_name or "nvidia" in self.image_flavor


def _parse_os_release(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def read_os(root: str = "/") -> OSInfo:
    flatpak = os.path.exists(os.path.join(root, ".flatpak-info"))
    candidates = (["run/host/os-release"] if flatpak else []) + ["etc/os-release", "usr/lib/os-release"]
    rel = {}
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            rel = _parse_os_release(open(p, encoding="utf-8", errors="replace").read())
            break
    info = OSInfo(id=rel.get("ID", "unknown").lower(), like=tuple(rel.get("ID_LIKE", "").lower().split()),
                  name=rel.get("PRETTY_NAME") or rel.get("NAME", "Linux"), version=rel.get("VERSION_ID", ""),
                  flatpak=flatpak)
    host = os.path.join(root, "run/host") if flatpak else root
    info.atomic = os.path.exists(os.path.join(host, "run/ostree-booted")) or \
        rel.get("VARIANT_ID", "").lower() in ("silverblue", "kinoite", "sericea", "onyx", "cosmic-atomic")
    img = os.path.join(host, "usr/share/ublue-os/image-info.json")
    if os.path.exists(img):
        try:
            data = json.load(open(img, encoding="utf-8"))
            info.image_name = str(data.get("image-name", "")).lower()
            info.image_tag = str(data.get("image-tag", "stable")) or "stable"
            info.image_flavor = str(data.get("image-flavor", "")).lower()
        except Exception:
            pass
        info.atomic = True
    blob = " ".join([info.id, *info.like, info.name.lower(), rel.get("VARIANT_ID", "").lower(), info.image_name])
    ids = {info.id, *info.like}
    if "bazzite" in blob:
        info.family = "bazzite"
    elif info.image_name or any(x in blob for x in ("bluefin", "aurora", "ublue")):
        info.family = "ublue"
    elif "fedora" in ids and info.atomic:
        info.family = "fedora-atomic"
    elif "fedora" in ids or ids & {"rhel", "centos", "rocky", "almalinux", "nobara"}:
        info.family = "fedora"
    elif ids & {"ubuntu", "linuxmint", "pop", "zorin", "elementary", "neon"}:
        info.family = "ubuntu"
        info.variant = {"pop": "Pop!_OS", "linuxmint": "Linux Mint"}.get(info.id, "")
    elif "debian" in ids:
        info.family = "debian"
    elif ids & {"arch", "manjaro", "endeavouros", "cachyos", "garuda", "archarm"}:
        info.family = "arch"
        info.variant = {"manjaro": "Manjaro"}.get(info.id, "")
    elif any(i.startswith("opensuse") or i in ("suse", "sles") for i in ids):
        info.family = "opensuse"
    return info


# ---------------------------------------------------------------------------
# hardware and what is missing
# ---------------------------------------------------------------------------

@dataclass
class PciDevice:
    vendor: int
    device: int
    cls: int


def pci_devices(root: str = "/") -> list[PciDevice]:
    out = []
    for d in glob.glob(os.path.join(root, "sys/bus/pci/devices/*")):
        try:
            v = int(open(os.path.join(d, "vendor")).read(), 16)
            dev = int(open(os.path.join(d, "device")).read(), 16)
            c = int(open(os.path.join(d, "class")).read(), 16) >> 8
            out.append(PciDevice(v, dev, c))
        except Exception:
            pass
    return out


@dataclass
class Issue:
    id: str          # nvidia_driver | rocm | vulkan_* | intel_compute | intel_npu
    title: str
    message: str
    benefit: str
    optional: bool = False
    device: int = 0  # PCI device id (NVIDIA: picks the right driver generation)


def nvidia_generation(device_id: int) -> str:
    """'open' (Turing and newer: GTX 16xx, RTX), 'legacy' (Maxwell, Pascal, Volta) or 'kepler'
    (and older - no longer supported by current NVIDIA drivers), from the PCI device id."""
    if device_id >= 0x1E00:
        return "open"
    if device_id >= 0x1340:
        return "legacy"
    return "kepler"


@dataclass
class Facts:
    """What the app can use right now (injected in tests)."""
    libcuda: bool = False
    vulkan_vendors: set = field(default_factory=set)  # PCI vendor ids of real Vulkan GPUs
    kfd_access: bool = False                          # /dev/kfd usable (AMD compute)
    amd_gfx: list = field(default_factory=list)       # gfx targets reported by the kernel driver
    opencl_gpu: bool = False
    openvino_npu: bool = False
    openvino_gpu: bool = False


def current_facts() -> Facts:
    from . import runtimes
    rocm = runtimes.rocm_driver() or {}
    f = Facts(libcuda=bool(runtimes._find_lib("libcuda.so.1")),
              kfd_access=bool(rocm.get("access")), amd_gfx=list(rocm.get("gfx") or []))
    try:
        import ncnn  # type: ignore
        for i in range(ncnn.get_gpu_count()):
            info = ncnn.get_gpu_info(i)
            if info.type() != 3:
                f.vulkan_vendors.add(info.vendor_id())
    except Exception:
        pass
    try:
        import openvino as ov  # type: ignore
        devs = ov.Core().available_devices
        f.openvino_npu = any(d.startswith("NPU") for d in devs)
        f.openvino_gpu = any(d.startswith("GPU") for d in devs)
    except Exception:
        pass
    try:
        from . import gpu
        f.opencl_gpu = gpu.device_name() is not None
    except Exception:
        pass
    return f


def find_issues(devices: list[PciDevice] | None = None, facts: Facts | None = None) -> list[Issue]:
    import sys
    if not sys.platform.startswith("linux"):
        return []
    devices = pci_devices() if devices is None else devices
    facts = current_facts() if facts is None else facts
    gpus = {d.vendor for d in devices if d.cls in _DISPLAY_CLASSES}
    issues = []
    if NVIDIA in gpus and not facts.libcuda:
        nv = max((d.device for d in devices if d.vendor == NVIDIA and d.cls in _DISPLAY_CLASSES), default=0)
        issues.append(Issue(
            "nvidia_driver", "NVIDIA-Treiber fehlt",
            "Es wurde eine NVIDIA-Grafikkarte gefunden, aber der NVIDIA-Treiber ist nicht installiert "
            "(es läuft vermutlich der freie Treiber „nouveau“).",
            "Mit dem NVIDIA-Treiber nutzt die App die Grafikkarte für alle KI-Funktionen und Filter – "
            "auf Wunsch zusätzlich mit CUDA und den Tensor-Kernen.", device=nv))
    for vendor, name in ((AMD, "AMD"), (INTEL, "Intel"), (NVIDIA, "NVIDIA")):
        if vendor in gpus and vendor not in facts.vulkan_vendors and not (vendor == NVIDIA and not facts.libcuda):
            issues.append(Issue(
                f"vulkan_{name.lower()}", f"Vulkan-Treiber für {name} fehlt",
                f"Es wurde eine {name}-Grafikkarte gefunden, aber kein Vulkan-Treiber dafür.",
                "Über Vulkan laufen KI-Funktionen und Filter auf der Grafikkarte."))
    from .runtimes import PROFILES
    rocm_ok = not facts.amd_gfx or any(g in PROFILES["rocm10"].devices for g in facts.amd_gfx)
    if AMD in gpus and rocm_ok and not facts.kfd_access:
        issues.append(Issue(
            "rocm", "AMD ROCm 10: Zugriff einrichten (optional)",
            "Die App kann für die AMD-Grafik ROCm 10 selbst nachladen, bekommt aber keinen Zugriff auf "
            "den Rechentreiber (/dev/kfd). Die Grafikkarte wird trotzdem bereits über Vulkan genutzt.",
            "Mit ROCm 10 laufen die KI-Modelle über AMDs MIGraphX – auf Radeon RX 7000/9000 und Ryzen-AI-"
            "Grafik meist deutlich schneller.", optional=True))
    if INTEL in gpus and not facts.openvino_gpu:
        issues.append(Issue(
            "intel_compute", "Intel Compute Runtime (optional)",
            "Für die Intel-Grafik fehlt die Intel Compute Runtime (OpenCL / Level Zero).",
            "Damit kann OpenVINO die Intel-Grafik für die KI-Modelle nutzen.", optional=True))
    if any(d.vendor == INTEL and d.cls == _NPU_CLASS for d in devices) and not facts.openvino_npu:
        issues.append(Issue(
            "intel_npu", "Intel-NPU-Treiber fehlt",
            "Es wurde eine Intel-NPU (KI-Beschleuniger, z. B. „AI Boost“) gefunden, aber der "
            "NPU-Treiber ist nicht installiert.",
            "Mit dem Treiber kann die App KI-Modelle stromsparend auf der NPU ausführen.", optional=True))
    return issues


# ---------------------------------------------------------------------------
# guides
# ---------------------------------------------------------------------------

@dataclass
class Step:
    text: str
    command: str = ""


@dataclass
class Guide:
    title: str
    os_label: str
    intro: str
    steps: list
    notes: list = field(default_factory=list)
    links: list = field(default_factory=list)    # (label, url)


_RPMFUSION = ("https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm "
              "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm")
_REBOOT = Step("Computer neu starten.", "systemctl reboot")
_SECURE_BOOT = ("Bei aktiviertem Secure Boot muss der Schlüssel der selbst gebauten Treiber-Module "
                "registriert werden (siehe RPM-Fusion-Anleitung), sonst lädt der Treiber nicht.")


def _ublue_nvidia_image(osi: OSInfo, gen: str = "open") -> tuple[str, str]:
    """Image with NVIDIA driver that matches the installed one (same desktop, same edition).
    Returns (image, note)."""
    name = osi.image_name or ("bazzite" if osi.family == "bazzite" else "bluefin")
    if osi.family != "bazzite":
        base = name.replace("-nvidia-open", "").replace("-nvidia", "")
        return f"{base}-{'nvidia-open' if gen == 'open' else 'nvidia'}", ""
    gnome = "gnome" in name
    if "deck" in name:
        if gen == "open":
            return ("bazzite-deck-nvidia-gnome" if gnome else "bazzite-deck-nvidia"), ""
        return (("bazzite-gnome-nvidia" if gnome else "bazzite-nvidia"),
                "Für ältere NVIDIA-Karten gibt es kein Handheld-/HTPC-Image (Bazzite-Deck). Vorgeschlagen ist "
                "daher die Desktop-Ausgabe – sie startet in den normalen Desktop statt in den Steam-Spielmodus.")
    base = "bazzite-gnome" if gnome else "bazzite"
    return f"{base}-{'nvidia-open' if gen == 'open' else 'nvidia'}", ""


_GEN_TEXT = {"open": "GeForce GTX 16xx / RTX (Turing oder neuer)",
             "legacy": "GeForce GTX 9xx / 10xx, Titan V (Maxwell, Pascal, Volta)",
             "kepler": "GeForce GTX 7xx oder älter (Kepler)"}


def _ublue_nvidia_guide(g: Guide, osi: OSInfo, issue: Issue) -> None:
    bazzite = osi.family == "bazzite"
    distro = "Bazzite" if bazzite else "Universal-Blue-Systemen (Bluefin, Aurora)"
    gen = nvidia_generation(issue.device) if issue.device else "open"
    if gen == "kepler":
        g.intro += (f"\n\nIhre Karte gehört zur Generation {_GEN_TEXT['kepler']}. Dafür gibt es auf {distro} "
                    "keinen NVIDIA-Treiber mehr – NVIDIA unterstützt diese Karten mit aktuellen Treibern nicht, "
                    "und der letzte passende Treiber (470) läuft nicht unter Wayland.")
        g.steps = [Step("Nichts installieren: Die Karte läuft weiter mit dem freien Treiber nouveau. Die App "
                        "nutzt sie über den freien Vulkan-Treiber (NVK), soweit Mesa ihn für die Karte "
                        "bereitstellt, sonst rechnet sie auf dem Prozessor.")]
        g.links = [("Bazzite-FAQ", DOCS["bazzite"] + "/General/FAQ/")] if bazzite else [("Universal Blue", DOCS["ublue"])]
        return
    image, note = _ublue_nvidia_image(osi, gen)
    tag = osi.image_tag or "stable"
    g.intro += (f"\n\nAuf {distro} wird der NVIDIA-Treiber nicht einzeln installiert: Er steckt fertig "
                "getestet in eigenen System-Images, auf die man umsteigt („rebase“). Persönliche Dateien, "
                "Einstellungen und Apps bleiben erhalten; ein Zurückwechseln ist jederzeit möglich.\n\n"
                f"Erkannte Karte: {_GEN_TEXT[gen]} → passendes Image: {image}:{tag}")
    if bazzite:
        g.steps = [
            Step("Aktuelles Image und Update-Zweig anzeigen (Bazzite Rollback Helper):", "brh current"),
            Step(f"Auf das Image mit NVIDIA-Treiber wechseln. Ihre Desktop-Umgebung und der Update-Zweig "
                 f"„{tag}“ bleiben gleich; es werden einige GB geladen:", f"brh rebase {image}:{tag}"),
            Step("Gleichwertiger Standardbefehl, falls „brh“ nicht vorhanden ist:",
                 f"rpm-ostree rebase ostree-image-signed:docker://ghcr.io/ublue-os/{image}:{tag}"),
            _REBOOT,
            Step("Nach dem Neustart prüfen – es sollte die Grafikkarte mit Treiberversion erscheinen:", "nvidia-smi"),
        ]
        g.notes += [
            "Den NVIDIA-Treiber aktualisiert Bazzite mit den normalen Systemupdates (Bazzite Updater bzw. "
            "„ujust update“) – ein manuelles Update ist weder nötig noch möglich.",
            "Bricht der Wechsel wegen selbst hinzugefügter Pakete ab („rpm-ostree install“), zeigt "
            "„rpm-ostree status“ diese an; „rpm-ostree reset“ entfernt alle zusätzlichen Pakete.",
            "Nicht zwischen KDE- und GNOME-Images wechseln – das unterstützt Bazzite nicht. Der Befehl oben "
            "behält Ihre Desktop-Umgebung bei.",
            "Secure Boot: Die Treiber in den Bazzite-Images sind mit dem Universal-Blue-Schlüssel signiert. "
            "Startet Bazzite bei Ihnen bereits mit Secure Boot, ist alles erledigt. Wollen Sie Secure Boot "
            "erst einschalten, vorher „ujust enroll-secure-boot-key“ ausführen (Passwort: universalblue).",
            "Zurück zum vorherigen Stand: „brh rollback“ und neu starten.",
        ]
        if note:
            g.notes.insert(0, note)
        g.links = [("Bazzite: Rebase-Anleitung", DOCS["bazzite_rebase"]),
                   ("Image-Auswahlhelfer", DOCS["bazzite_images"]),
                   ("Secure Boot", DOCS["bazzite_secure_boot"])]
    else:
        g.steps = [
            Step("Aktuelles Image anzeigen:", "rpm-ostree status"),
            Step("Auf das Image mit NVIDIA-Treiber wechseln (einige GB Download):",
                 f"rpm-ostree rebase ostree-image-signed:docker://ghcr.io/ublue-os/{image}:{tag}"),
            _REBOOT,
            Step("Nach dem Neustart prüfen:", "nvidia-smi"),
        ]
        g.notes.append("Bitte den Image-Namen mit der Universal-Blue-Dokumentation abgleichen. Zurück zum "
                       "vorherigen Stand: „rpm-ostree rollback“.")
        g.links = [("Universal Blue", DOCS["ublue"])]


def _rocm_guide(g: Guide, osi: OSInfo) -> None:
    f = osi.family
    g.intro += ("\n\nSeit ROCm 10 bringt AMD alle Rechenbibliotheken als Pakete mit, die die App selbst in "
                "Ihren Benutzerordner lädt (ca. 1 GB Download). ROCm muss also nicht mehr ins System "
                "installiert werden. Nötig sind nur der amdgpu-Kerneltreiber – in jedem aktuellen Kernel "
                "enthalten – und das Recht, den Rechentreiber /dev/kfd zu öffnen.")
    if f == "bazzite":
        g.intro += ("\n\nAuf Bazzite heißt das: nichts mit „rpm-ostree install“ ins Image aufnehmen. Das "
                    "System-Image bleibt unverändert, Updates und Rebase laufen weiter wie gewohnt.")
    g.steps = [Step("Prüfen, ob der Rechentreiber da ist und welche Rechte er hat:", "ls -l /dev/kfd")]
    if osi.atomic:
        g.steps += [
            Step("Zugriff für die Grafikkarte freigeben (udev-Regel nach AMD-Anleitung; sie liegt in /etc und "
                 "bleibt bei Updates erhalten):",
                 "printf 'KERNEL==\"kfd\", GROUP=\"render\", MODE=\"0666\"\\nSUBSYSTEM==\"drm\", "
                 "KERNEL==\"renderD*\", GROUP=\"render\", MODE=\"0666\"\\n' | sudo tee "
                 "/etc/udev/rules.d/70-amdgpu.rules"),
            Step("Regel sofort anwenden (oder neu starten):",
                 "sudo udevadm control --reload-rules && sudo udevadm trigger"),
        ]
        g.notes.append("Warum nicht einfach „usermod -a -G render“? Auf atomaren Systemen wie Bazzite stehen die "
                       "Systemgruppen in /usr/lib/group, deshalb meldet usermod dort oft „Gruppe existiert "
                       "nicht“. Die udev-Regel umgeht das.")
    else:
        g.steps += [
            Step("Den eigenen Benutzer in die Gruppen „render“ und „video“ aufnehmen:",
                 "sudo usermod -a -G render,video $LOGNAME"),
            Step("Abmelden und wieder anmelden (oder neu starten), damit die Gruppen gelten."),
        ]
    g.steps.append(Step("Die App neu starten und unter Einstellungen → GPU-Laufzeiten und Treiber "
                        "„AMD ROCm 10 – MIGraphX“ nachladen."))
    if osi.flatpak:
        g.notes.append("Im Flatpak braucht die App dafür Zugriff auf alle Geräte (ist voreingestellt). Falls Sie "
                       "das z. B. mit Flatseal abgeschaltet haben: "
                       "„flatpak override --user --device=all io.github.lordsimon666.SimonImageEnhance“.")
    g.notes.append("ROCm 10 mit ONNX Runtime unterstützt Radeon RX 7000 (gfx1100–1102), RX 9000 (gfx1200/1201) "
                   "und die Ryzen-AI-Grafik (gfx1150–1153). Ältere Karten wie RX 6000, Radeon 780M oder das "
                   "Steam Deck laufen weiterhin über Vulkan.")
    if f in ("ubuntu", "debian"):
        g.notes.append("Ryzen-AI-Prozessoren (Strix Point / Strix Halo) brauchen Kernel 6.14 oder neuer; unter "
                       "Ubuntu 24.04 also den HWE- bzw. OEM-Kernel.")
    if f == "bazzite":
        g.links = [("Bazzite: Pakete layern (warum nicht nötig)", DOCS["bazzite_layering"])]
    g.links.append(("AMD ROCm-Dokumentation", DOCS["rocm"]))


def _os_label(osi: OSInfo) -> str:
    label = osi.name
    if osi.family in ("bazzite", "ublue") and osi.image_name:
        label += f" (Image: {osi.image_name}:{osi.image_tag})"
    elif osi.atomic:
        label += " (atomares System)"
    if osi.flatpak:
        label += " – App läuft als Flatpak"
    return label


def guide(issue: Issue, osi: OSInfo) -> Guide:
    f = osi.family
    g = Guide(issue.title, _os_label(osi), f"{issue.message}\n\n{issue.benefit}", [])
    if osi.flatpak:
        g.notes.append("Die App läuft als Flatpak: Treiber werden auf dem System installiert, nicht im Flatpak. "
                       "Danach einmal „flatpak update“ ausführen – Flatpak lädt dann automatisch die zum "
                       "Treiber passende Grafik-Erweiterung.")

    if issue.id == "nvidia_driver":
        if f in ("bazzite", "ublue"):
            _ublue_nvidia_guide(g, osi, issue)
        elif f == "fedora-atomic":
            g.steps = [
                Step("RPM-Fusion-Paketquellen hinzufügen:", f"sudo rpm-ostree install {_RPMFUSION}"),
                _REBOOT,
                Step("NVIDIA-Treiber mit CUDA-Unterstützung installieren:",
                     "sudo rpm-ostree install akmod-nvidia xorg-x11-drv-nvidia-cuda"),
                Step("Den freien Treiber nouveau abschalten:",
                     "sudo rpm-ostree kargs --append=rd.driver.blacklist=nouveau "
                     "--append=modprobe.blacklist=nouveau --append=nvidia-drm.modeset=1"),
                _REBOOT,
            ]
            g.notes.append(_SECURE_BOOT)
            g.links = [("RPM Fusion: NVIDIA", DOCS["rpmfusion_nvidia"])]
        elif f == "fedora":
            g.steps = [
                Step("RPM-Fusion-Paketquellen hinzufügen:", f"sudo dnf install {_RPMFUSION}"),
                Step("NVIDIA-Treiber mit CUDA-Unterstützung installieren:",
                     "sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda"),
                Step("Einige Minuten warten, bis das Treibermodul gebaut ist – fertig, wenn dieser Befehl "
                     "eine Versionsnummer ausgibt:", "modinfo -F version nvidia"),
                _REBOOT,
            ]
            g.notes.append(_SECURE_BOOT)
            g.links = [("RPM Fusion: NVIDIA", DOCS["rpmfusion_nvidia"])]
        elif f == "ubuntu":
            if osi.id == "pop":
                g.steps = [Step("NVIDIA-Treiber installieren:", "sudo apt install system76-driver-nvidia"), _REBOOT]
            else:
                g.steps = [Step("Empfohlenen NVIDIA-Treiber automatisch installieren:", "sudo ubuntu-drivers install"),
                           _REBOOT]
                if osi.id == "linuxmint":
                    g.notes.append("Alternativ in Linux Mint: Startmenü → „Treiberverwaltung“.")
            g.links = [("Ubuntu: NVIDIA-Treiber", DOCS["ubuntu_nvidia"])]
        elif f == "debian":
            g.steps = [Step("In /etc/apt/sources.list die Bereiche „contrib non-free non-free-firmware“ "
                            "aktivieren, dann:", "sudo apt update"),
                       Step("NVIDIA-Treiber installieren:", "sudo apt install nvidia-driver firmware-misc-nonfree"),
                       _REBOOT]
            g.links = [("Debian-Wiki: NVIDIA", DOCS["debian_nvidia"])]
        elif f == "arch":
            if osi.id == "manjaro":
                g.steps = [Step("Proprietären Treiber automatisch einrichten:", "sudo mhwd -a pci nonfree 0300"), _REBOOT]
            else:
                g.steps = [Step("Treiber installieren (GTX 16xx / RTX 20xx und neuer):",
                                "sudo pacman -S nvidia-open nvidia-utils"), _REBOOT]
                g.notes.append("Für ältere Karten (GTX 9xx / 10xx) siehe Arch-Wiki – dort sind andere Pakete nötig.")
            g.links = [("Arch-Wiki: NVIDIA", DOCS["arch_nvidia"])]
        elif f == "opensuse":
            repo = "tumbleweed" if "tumbleweed" in osi.id or "tumbleweed" in osi.name.lower() else "leap/$releasever"
            g.steps = [Step("NVIDIA-Paketquelle hinzufügen:",
                            f"sudo zypper addrepo --refresh https://download.nvidia.com/opensuse/{repo} NVIDIA"),
                       Step("Passenden Treiber installieren:", "sudo zypper install-new-recommends --repo NVIDIA"),
                       _REBOOT]
            g.links = [("openSUSE: NVIDIA-Treiber", DOCS["opensuse_nvidia"])]
        else:
            g.steps = [Step("Den NVIDIA-Treiber über die Treiberverwaltung bzw. Paketverwaltung Ihrer "
                            "Distribution installieren und neu starten.")]
            g.links = [("NVIDIA-Treiber", DOCS["nvidia"])]
        if not (f in ("bazzite", "ublue") and issue.device and nvidia_generation(issue.device) == "kepler"):
            g.notes.append("Nach dem Neustart erkennt die App den Treiber und bietet die CUDA-Unterstützung "
                           "automatisch an (fehlende CUDA-Bibliotheken lädt sie selbst nach).")

    elif issue.id.startswith("vulkan_"):
        vendor = issue.id.split("_", 1)[1]
        if f in ("bazzite", "ublue"):
            g.intro += ("\n\nDie Vulkan-Treiber sind fester Bestandteil des System-Images und werden mit jedem "
                        "Update mitgeliefert – einzeln installieren muss (und kann) man sie nicht.")
            g.steps = [Step("System auf den neuesten Stand bringen:", "ujust update"), _REBOOT]
            if vendor == "nvidia":
                g.steps.insert(0, Step("Prüfen, ob ein Image mit NVIDIA-Treiber läuft (Name endet auf "
                                       "„-nvidia“ oder „-nvidia-open“):",
                                       "brh current" if f == "bazzite" else "rpm-ostree status"))
            g.notes.append("Läuft die App als Flatpak, zusätzlich „flatpak update“ ausführen: Die Grafiktreiber "
                           "im Flatpak kommen aus der Flatpak-Laufzeit.")
        elif f in ("fedora", "fedora-atomic"):
            tool = "sudo rpm-ostree install" if f == "fedora-atomic" else "sudo dnf install"
            g.steps = [Step("Vulkan-Treiber installieren:", f"{tool} mesa-vulkan-drivers vulkan-loader")]
            if f == "fedora-atomic":
                g.steps.append(_REBOOT)
        elif f in ("ubuntu", "debian"):
            g.steps = [Step("Vulkan-Treiber installieren:", "sudo apt install mesa-vulkan-drivers libvulkan1")]
        elif f == "arch":
            pkg = {"amd": "vulkan-radeon", "intel": "vulkan-intel", "nvidia": "nvidia-utils"}[vendor]
            g.steps = [Step("Vulkan-Treiber installieren:", f"sudo pacman -S {pkg} vulkan-icd-loader")]
        elif f == "opensuse":
            g.steps = [Step("Vulkan-Treiber installieren:", "sudo zypper install Mesa-vulkan-device-select libvulkan1 libvulkan_radeon libvulkan_intel")]
        else:
            g.steps = [Step("Das Mesa-Vulkan-Paket Ihrer Distribution installieren (oft „mesa-vulkan-drivers“).")]

    elif issue.id == "rocm":
        _rocm_guide(g, osi)

    elif issue.id == "intel_compute":
        if f in ("bazzite", "ublue", "fedora-atomic"):
            g.steps = [Step("Prüfen, ob die Runtime schon im Image ist:", "rpm -q intel-compute-runtime"),
                       Step("Falls nicht: Runtime ins Image aufnehmen („layern“):",
                            "rpm-ostree install intel-compute-runtime oneapi-level-zero"), _REBOOT]
            g.notes.append("Zusätzlich gelayerte Pakete können Updates und einen späteren Rebase aufhalten. "
                           "Wieder entfernen: „rpm-ostree uninstall intel-compute-runtime oneapi-level-zero“.")
        elif f == "fedora":
            g.steps = [Step("Intel Compute Runtime installieren:", "sudo dnf install intel-compute-runtime oneapi-level-zero")]
        elif f in ("ubuntu", "debian"):
            g.steps = [Step("Intel Compute Runtime installieren:", "sudo apt install intel-opencl-icd")]
        elif f == "arch":
            g.steps = [Step("Intel Compute Runtime installieren:", "sudo pacman -S intel-compute-runtime level-zero-loader")]
        else:
            g.steps = [Step("Das Paket „intel-compute-runtime“ (OpenCL / Level Zero) Ihrer Distribution installieren.")]
        g.links = [("Intel Compute Runtime", DOCS["intel_compute"])]

    elif issue.id == "intel_npu":
        g.intro += ("\n\nDer Kernel bringt den NPU-Treiber („intel_vpu“) ab Linux 6.3 mit; zusätzlich wird der "
                    "Benutzerteil von Intel benötigt (linux-npu-driver).")
        g.steps = [Step("Kernel-Treiber prüfen – es sollte „/dev/accel/accel0“ existieren:", "ls /dev/accel/"),
                   Step("Den Intel-NPU-Treiber (Benutzerteil) nach Intels Anleitung installieren; für Ubuntu gibt "
                        "es fertige Pakete, für andere Distributionen teils Pakete in der Paketverwaltung.")]
        if f in ("bazzite", "ublue", "fedora-atomic"):
            g.notes.append("Auf atomaren Systemen müssen zusätzliche Pakete mit „rpm-ostree install“ ins Image "
                           "aufgenommen werden.")
        g.links = [("Intel linux-npu-driver", DOCS["intel_npu"])]
    return g
