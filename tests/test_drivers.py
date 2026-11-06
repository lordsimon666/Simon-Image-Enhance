"""Missing-driver detection and the per-OS installation guides."""
import json
import os

import pytest

from simon_enhance import drivers as D


def make_root(tmp_path, os_release: dict, image_info: dict | None = None, ostree=False, flatpak=False):
    root = tmp_path / "root"
    rel = "\n".join(f'{k}="{v}"' for k, v in os_release.items())
    target = root / ("run/host/os-release" if flatpak else "etc/os-release")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rel)
    host = root / "run/host" if flatpak else root
    if image_info is not None:
        p = host / "usr/share/ublue-os/image-info.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(image_info))
    if ostree:
        (host / "run").mkdir(parents=True, exist_ok=True)
        (host / "run/ostree-booted").write_text("")
    if flatpak:
        (root / ".flatpak-info").write_text("[Application]\n")
    return str(root)


BAZZITE = {"ID": "bazzite", "ID_LIKE": "fedora", "NAME": "Bazzite", "PRETTY_NAME": "Bazzite 43 (FROM Fedora Kinoite)",
           "VERSION_ID": "43", "VARIANT_ID": "bazzite"}


@pytest.mark.parametrize("rel,info,ostree,family", [
    (BAZZITE, {"image-name": "bazzite", "image-tag": "stable"}, True, "bazzite"),
    ({"ID": "fedora", "NAME": "Fedora Linux", "VARIANT_ID": "kinoite"}, {"image-name": "bazzite-gnome"}, True, "bazzite"),
    ({"ID": "fedora", "NAME": "Bluefin"}, {"image-name": "bluefin", "image-tag": "stable"}, True, "ublue"),
    ({"ID": "fedora", "NAME": "Fedora Linux", "VARIANT_ID": "silverblue"}, None, True, "fedora-atomic"),
    ({"ID": "fedora", "NAME": "Fedora Linux", "VARIANT_ID": "workstation"}, None, False, "fedora"),
    ({"ID": "nobara", "ID_LIKE": "rhel centos fedora", "NAME": "Nobara"}, None, False, "fedora"),
    ({"ID": "ubuntu", "ID_LIKE": "debian", "NAME": "Ubuntu"}, None, False, "ubuntu"),
    ({"ID": "linuxmint", "ID_LIKE": "ubuntu debian", "NAME": "Linux Mint"}, None, False, "ubuntu"),
    ({"ID": "pop", "ID_LIKE": "ubuntu debian", "NAME": "Pop!_OS"}, None, False, "ubuntu"),
    ({"ID": "debian", "NAME": "Debian GNU/Linux"}, None, False, "debian"),
    ({"ID": "arch", "NAME": "Arch Linux"}, None, False, "arch"),
    ({"ID": "cachyos", "ID_LIKE": "arch", "NAME": "CachyOS"}, None, False, "arch"),
    ({"ID": "manjaro", "ID_LIKE": "arch", "NAME": "Manjaro"}, None, False, "arch"),
    ({"ID": "opensuse-tumbleweed", "ID_LIKE": "opensuse suse", "NAME": "openSUSE Tumbleweed"}, None, False, "opensuse"),
    ({"ID": "gentoo", "NAME": "Gentoo"}, None, False, "other"),
])
def test_os_family(tmp_path, rel, info, ostree, family):
    assert D.read_os(make_root(tmp_path, rel, info, ostree)).family == family


def test_flatpak_reads_host_os(tmp_path):
    osi = D.read_os(make_root(tmp_path, BAZZITE, {"image-name": "bazzite-deck"}, True, flatpak=True))
    assert osi.flatpak and osi.family == "bazzite" and osi.image_name == "bazzite-deck"


def _facts(**kw):
    return D.Facts(**kw)


def test_issues_nvidia_without_driver():
    devs = [D.PciDevice(D.NVIDIA, 0x2786, 0x0300)]
    ids = [i.id for i in D.find_issues(devs, _facts(libcuda=False))]
    assert ids == ["nvidia_driver"]  # no separate Vulkan issue: the NVIDIA driver brings Vulkan
    assert D.find_issues(devs, _facts(libcuda=True, vulkan_vendors={D.NVIDIA})) == []


def test_issues_amd_and_intel():
    devs = [D.PciDevice(D.AMD, 0x744C, 0x0300), D.PciDevice(D.INTEL, 0x7D55, 0x0300),
            D.PciDevice(D.INTEL, 0x7D1D, 0x1200)]
    issues = {i.id: i for i in D.find_issues(devs, _facts(vulkan_vendors={D.AMD, D.INTEL}))}
    assert set(issues) == {"rocm", "intel_compute", "intel_npu"}
    assert all(i.optional for i in issues.values())        # nothing blocks the GPU: no pop-up
    issues = {i.id for i in D.find_issues(devs, _facts(vulkan_vendors=set(), kfd_access=True,
                                                       openvino_gpu=True, openvino_npu=True))}
    assert issues == {"vulkan_amd", "vulkan_intel"}
    # an AMD GPU that ROCm 10 does not support (Steam Deck): nothing to set up
    deck = [D.PciDevice(D.AMD, 0x163F, 0x0300)]
    assert D.find_issues(deck, _facts(vulkan_vendors={D.AMD}, amd_gfx=["gfx1033"])) == []
    assert [i.id for i in D.find_issues(deck, _facts(vulkan_vendors={D.AMD}, amd_gfx=["gfx1201"]))] == ["rocm"]


RTX4070, GTX1080, GTX780 = 0x2786, 0x1B80, 0x1004


@pytest.mark.parametrize("image,dev,target", [
    ("bazzite", RTX4070, "bazzite-nvidia-open"),
    ("bazzite-gnome", RTX4070, "bazzite-gnome-nvidia-open"),
    ("bazzite-nvidia", RTX4070, "bazzite-nvidia-open"),
    ("bazzite", GTX1080, "bazzite-nvidia"),                 # Pascal: legacy image
    ("bazzite-gnome", GTX1080, "bazzite-gnome-nvidia"),
    ("bazzite-deck", RTX4070, "bazzite-deck-nvidia"),       # handheld/HTPC edition with NVIDIA
    ("bazzite-deck-gnome", RTX4070, "bazzite-deck-nvidia-gnome"),
    ("bazzite-deck", GTX1080, "bazzite-nvidia"),            # no legacy deck image: desktop edition
])
def test_bazzite_nvidia_guide_uses_rebase_to_matching_image(tmp_path, image, dev, target):
    osi = D.read_os(make_root(tmp_path, BAZZITE, {"image-name": image, "image-tag": "testing"}, True))
    g = D.guide(D.find_issues([D.PciDevice(D.NVIDIA, dev, 0x0300)], _facts())[0], osi)
    cmds = [s.command for s in g.steps if s.command]
    assert cmds[0] == "brh current"
    assert f"brh rebase {target}:testing" in cmds
    assert f"rpm-ostree rebase ostree-image-signed:docker://ghcr.io/ublue-os/{target}:testing" in cmds
    assert "nvidia-smi" in cmds
    assert not any("dnf" in c or "rpm-ostree install" in c for c in cmds)  # never layer the driver
    assert any("docs.bazzite.gg" in u for _t, u in g.links)
    assert any("brh rollback" in n for n in g.notes) and any("enroll-secure-boot-key" in n for n in g.notes)
    assert ("Desktop-Ausgabe" in " ".join(g.notes)) == ("deck" in image and target.startswith("bazzite-") and "deck" not in target)


def test_bazzite_kepler_gets_no_rebase(tmp_path):
    osi = D.read_os(make_root(tmp_path, BAZZITE, {"image-name": "bazzite"}, True))
    g = D.guide(D.find_issues([D.PciDevice(D.NVIDIA, GTX780, 0x0300)], _facts())[0], osi)
    assert not any("rebase" in s.command for s in g.steps) and "Kepler" in g.intro


def test_nvidia_generation():
    assert D.nvidia_generation(0x2786) == "open" and D.nvidia_generation(0x2182) == "open"   # RTX 4070, GTX 1660 Ti
    assert D.nvidia_generation(0x1B80) == "legacy" and D.nvidia_generation(0x1381) == "legacy"  # GTX 1080, 750 Ti
    assert D.nvidia_generation(0x1004) == "kepler"


@pytest.mark.parametrize("rel,info,ostree,expect,avoid", [
    (BAZZITE, {"image-name": "bazzite"}, True, "udev", "usermod"),
    ({"ID": "fedora", "VARIANT_ID": "silverblue"}, None, True, "udev", "usermod"),
    ({"ID": "ubuntu", "ID_LIKE": "debian"}, None, False, "usermod -a -G render,video", "udev"),
    ({"ID": "arch"}, None, False, "usermod -a -G render,video", "rpm-ostree"),
])
def test_rocm10_guide(tmp_path, rel, info, ostree, expect, avoid):
    osi = D.read_os(make_root(tmp_path, rel, info, ostree))
    issue = D.find_issues([D.PciDevice(D.AMD, 0x7550, 0x0300)], _facts(vulkan_vendors={D.AMD}))[0]
    g = D.guide(issue, osi)
    cmds = " ".join(s.command for s in g.steps)
    assert issue.id == "rocm" and "ls -l /dev/kfd" in cmds and expect in cmds and avoid not in cmds
    assert "rpm-ostree install" not in cmds and "dnf" not in cmds   # ROCm 10 comes with the app
    assert "ROCm 10" in g.intro
    if osi.family == "bazzite":
        assert "Image bleibt unverändert" in g.intro.replace("System-Image bleibt unverändert", "Image bleibt unverändert")


@pytest.mark.parametrize("rel,ostree,expect,avoid", [
    ({"ID": "fedora", "NAME": "Fedora", "VARIANT_ID": "silverblue"}, True, "rpm-ostree install akmod-nvidia", "dnf"),
    ({"ID": "fedora", "NAME": "Fedora", "VARIANT_ID": "workstation"}, False, "dnf install akmod-nvidia", "rpm-ostree"),
    ({"ID": "ubuntu", "ID_LIKE": "debian", "NAME": "Ubuntu"}, False, "ubuntu-drivers install", "dnf"),
    ({"ID": "pop", "ID_LIKE": "ubuntu debian", "NAME": "Pop!_OS"}, False, "system76-driver-nvidia", "ubuntu-drivers"),
    ({"ID": "debian", "NAME": "Debian"}, False, "apt install nvidia-driver", "dnf"),
    ({"ID": "arch", "NAME": "Arch Linux"}, False, "pacman -S nvidia-open", "dnf"),
    ({"ID": "manjaro", "ID_LIKE": "arch", "NAME": "Manjaro"}, False, "mhwd", "pacman"),
    ({"ID": "opensuse-tumbleweed", "ID_LIKE": "opensuse", "NAME": "openSUSE Tumbleweed"}, False, "zypper", "dnf"),
])
def test_nvidia_guides_per_distribution(tmp_path, rel, ostree, expect, avoid):
    osi = D.read_os(make_root(tmp_path, rel, None, ostree))
    issue = D.find_issues([D.PciDevice(D.NVIDIA, RTX4070, 0x0300)], _facts())[0]
    cmds = " ".join(s.command for s in D.guide(issue, osi).steps)
    assert expect in cmds and avoid not in cmds


def test_every_issue_has_a_guide_on_every_family(tmp_path):
    devs = [D.PciDevice(D.AMD, 1, 0x0300), D.PciDevice(D.INTEL, 2, 0x0300), D.PciDevice(D.INTEL, 3, 0x1200),
            D.PciDevice(D.NVIDIA, 4, 0x0300)]
    issues = D.find_issues(devs, _facts())
    rels = [BAZZITE, {"ID": "fedora", "VARIANT_ID": "silverblue"}, {"ID": "fedora"}, {"ID": "ubuntu"},
            {"ID": "debian"}, {"ID": "arch"}, {"ID": "opensuse-leap", "ID_LIKE": "opensuse"}, {"ID": "gentoo"}]
    for n, rel in enumerate(rels):
        osi = D.read_os(make_root(tmp_path / str(n), rel, {"image-name": "bazzite"} if rel is BAZZITE else None,
                                  ostree=rel.get("VARIANT_ID") == "silverblue"))
        for issue in issues:
            g = D.guide(issue, osi)
            assert g.steps and g.title, (osi.family, issue.id)
            if osi.atomic:
                assert not any(s.command.startswith("sudo dnf") for s in g.steps), (osi.family, issue.id)


def test_flatpak_note(tmp_path):
    osi = D.read_os(make_root(tmp_path, {"ID": "fedora"}, None, False, flatpak=True))
    g = D.guide(D.find_issues([D.PciDevice(D.NVIDIA, 1, 0x0300)], _facts())[0], osi)
    assert any("flatpak update" in n for n in g.notes)


def test_real_system_does_not_crash():
    D.read_os()
    D.find_issues()
