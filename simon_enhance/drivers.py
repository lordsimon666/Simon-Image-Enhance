"""Find GPUs / NPUs whose drivers are missing and build installation guides for the
operating system in use (Bazzite and other Universal Blue images, Fedora Atomic,
Fedora, Ubuntu/Mint/Pop!_OS, Debian, Arch family, openSUSE, generic).

Detection looks at the PCI bus (works without any driver) and compares it with what
the app can actually use (NVIDIA driver/libcuda, Vulkan devices, ROCm, Intel compute
runtime, Intel NPU via OpenVINO).
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
    "ublue": "https://universal-blue.org",
    "rpmfusion_nvidia": "https://rpmfusion.org/Howto/NVIDIA",
    "ubuntu_nvidia": "https://ubuntu.com/server/docs/nvidia-drivers-installation",
    "debian_nvidia": "https://wiki.debian.org/NvidiaGraphicsDrivers",
    "arch_nvidia": "https://wiki.archlinux.org/title/NVIDIA",
    "opensuse_nvidia": "https://en.opensuse.org/SDB:NVIDIA_drivers",
    "nvidia": "https://www.nvidia.com/drivers",
    "rocm": "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/",
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
    id: str          # nvidia_driver | rocm | vulkan | intel_compute | intel_npu
    title: str
    message: str
    benefit: str
    optional: bool = False


@dataclass
class Facts:
    """What the app can use right now (injected in tests)."""
    libcuda: bool = False
    vulkan_vendors: set = field(default_factory=set)  # PCI vendor ids of real Vulkan GPUs
    rocm7: bool = False
    opencl_gpu: bool = False
    openvino_npu: bool = False
    openvino_gpu: bool = False


def current_facts() -> Facts:
    from . import runtimes
    f = Facts(libcuda=bool(runtimes._find_lib("libcuda.so.1")),
              rocm7=bool(runtimes._find_lib("libamdhip64.so.7")))
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
        issues.append(Issue(
            "nvidia_driver", "NVIDIA-Treiber fehlt",
            "Es wurde eine NVIDIA-Grafikkarte gefunden, aber der NVIDIA-Treiber ist nicht installiert "
            "(es läuft vermutlich der freie Treiber „nouveau“).",
            "Mit dem NVIDIA-Treiber nutzt die App die Grafikkarte für alle KI-Funktionen und Filter – "
            "auf Wunsch zusätzlich mit CUDA und den Tensor-Kernen."))
    for vendor, name in ((AMD, "AMD"), (INTEL, "Intel"), (NVIDIA, "NVIDIA")):
        if vendor in gpus and vendor not in facts.vulkan_vendors and not (vendor == NVIDIA and not facts.libcuda):
            issues.append(Issue(
                f"vulkan_{name.lower()}", f"Vulkan-Treiber für {name} fehlt",
                f"Es wurde eine {name}-Grafikkarte gefunden, aber kein Vulkan-Treiber dafür.",
                "Über Vulkan laufen KI-Funktionen und Filter auf der Grafikkarte."))
    if AMD in gpus and not facts.rocm7:
        issues.append(Issue(
            "rocm", "AMD ROCm (optional)",
            "Für die AMD-Grafikkarte ist ROCm 7 nicht installiert. Die Grafikkarte wird trotzdem bereits "
            "über Vulkan genutzt.",
            "ROCm ist AMDs Rechenplattform; damit kann die App die besonders schnelle MIGraphX-Laufzeit "
            "nachladen.", optional=True))
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


def _ublue_nvidia_image(osi: OSInfo) -> str:
    base = osi.image_name or ("bazzite" if osi.family == "bazzite" else "bluefin")
    for suffix in ("-nvidia-open", "-nvidia"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return f"{base}-nvidia-open"


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
            target = _ublue_nvidia_image(osi)
            distro = "Bazzite" if f == "bazzite" else "Universal-Blue-Systeme (Bluefin, Aurora)"
            g.intro += (f"\n\nAuf {distro} wird der NVIDIA-Treiber nicht einzeln installiert: Es gibt fertige "
                        "Images mit NVIDIA-Treiber, auf die man umsteigt („rebase“). Daten und Einstellungen "
                        "bleiben erhalten, ein Zurückwechseln ist jederzeit möglich.")
            if "deck" in osi.image_name:
                g.notes.append("Die Deck-/Handheld-Images sind für AMD-Geräte gedacht – bitte im "
                               "Image-Auswahlhelfer das passende Image mit NVIDIA-Treiber wählen.")
            g.steps = [
                Step("Passendes Image prüfen: „-nvidia-open“ für GeForce GTX 16xx / RTX 20xx und neuer, "
                     "„-nvidia“ für ältere Karten (GTX 9xx / 10xx). Im Zweifel den offiziellen "
                     "Image-Auswahlhelfer verwenden (Link unten)."),
                Step("Auf das NVIDIA-Image wechseln (Download einige GB, dauert etwas):",
                     f"rpm-ostree rebase ostree-image-signed:docker://ghcr.io/ublue-os/{target}:{osi.image_tag}"),
                _REBOOT,
            ]
            g.notes.append("Bitte den Befehl mit der offiziellen Bazzite-/Universal-Blue-Dokumentation "
                           "abgleichen – Image-Namen können sich ändern.")
            g.links = [("Bazzite-Dokumentation", DOCS["bazzite"]), ("Image-Auswahlhelfer", DOCS["bazzite_images"])] \
                if f == "bazzite" else [("Universal Blue", DOCS["ublue"])]
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
        g.notes.append("Nach dem Neustart erkennt die App den Treiber und bietet die CUDA-Unterstützung "
                       "automatisch an (fehlende CUDA-Bibliotheken lädt sie selbst nach).")

    elif issue.id.startswith("vulkan_"):
        vendor = issue.id.split("_", 1)[1]
        if f in ("bazzite", "ublue"):
            g.steps = [Step("Vulkan ist im Image enthalten – das System aktualisieren:", "ujust update"), _REBOOT]
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
        g.intro += ("\n\nROCm ist groß (mehrere GB) und nicht für jede Distribution und jede Karte offiziell "
                    "verfügbar. Die App braucht ROCm 7 mit MIGraphX. Ohne ROCm nutzt sie die Grafikkarte "
                    "weiterhin über Vulkan – für die meisten ist das ausreichend.")
        if f in ("bazzite", "ublue", "fedora-atomic", "fedora"):
            tool = "rpm-ostree install" if f != "fedora" else "dnf install"
            query = "dnf info rocm-hip rocm-runtime" if f == "fedora" else "rpm-ostree search rocm-hip"
            g.steps = [Step("Prüfen, welche ROCm-Version Ihre Fedora-Version anbietet "
                            "(die App braucht ROCm 7):", query),
                       Step("Falls ROCm 7 angeboten wird, installieren:",
                            f"sudo {tool} rocm-hip rocm-runtime rocminfo")]
            if f != "fedora":
                g.steps.append(_REBOOT)
                g.notes.append("Auf atomaren Systemen (Bazzite, Silverblue, Kinoite) wird ROCm als zusätzliche "
                               "Pakete in das Image „gelayert“; jedes Update dauert danach etwas länger.")
            g.notes.append("MIGraphX ist nicht in allen Fedora-Versionen enthalten. Ohne MIGraphX bleibt "
                           "Vulkan der empfohlene Weg.")
        elif f == "ubuntu":
            g.steps = [Step("AMD stellt ROCm für Ubuntu LTS offiziell bereit – bitte der AMD-Anleitung folgen "
                            "(„amdgpu-install“, Anwendungsfall „rocm“) und dabei MIGraphX mitinstallieren.")]
        else:
            g.steps = [Step("ROCm 7 mit MIGraphX nach der offiziellen AMD-Anleitung installieren, falls Ihre "
                            "Distribution unterstützt wird.")]
        g.links = [("AMD ROCm: Installation unter Linux", DOCS["rocm"])]

    elif issue.id == "intel_compute":
        if f in ("bazzite", "ublue", "fedora-atomic"):
            g.steps = [Step("Intel Compute Runtime ins Image aufnehmen:",
                            "sudo rpm-ostree install intel-compute-runtime oneapi-level-zero"), _REBOOT]
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
