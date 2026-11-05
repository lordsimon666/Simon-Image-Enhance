#!/usr/bin/env bash
# Build a .deb from the PyInstaller bundle.
# Usage: packaging/linux/build_deb.sh <bundle-dir> <version> [out-dir]
set -euo pipefail
BUNDLE=$(realpath "$1"); VERSION="$2"; OUT=$(realpath -m "${3:-dist}")
HERE=$(cd "$(dirname "$0")" && pwd)
APPID=io.github.lordsimon666.SimonImageEnhance
PKG=simon-image-enhance
ARCH=$(dpkg --print-architecture 2>/dev/null || echo amd64)
ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT

install -d "$ROOT/DEBIAN" "$ROOT/opt/$PKG" "$ROOT/usr/bin" "$ROOT/usr/share/applications" \
           "$ROOT/usr/share/metainfo"
cp -a "$BUNDLE/." "$ROOT/opt/$PKG/"
ln -s "/opt/$PKG/SimonImageEnhance" "$ROOT/usr/bin/$PKG"
install -m644 "$HERE/$APPID.desktop" "$ROOT/usr/share/applications/$APPID.desktop"
install -m644 "$HERE/$APPID.metainfo.xml" "$ROOT/usr/share/metainfo/$APPID.metainfo.xml"
for s in 16 32 48 64 128 256 512; do
  install -Dm644 "$HERE/../icons/icon-$s.png" "$ROOT/usr/share/icons/hicolor/${s}x${s}/apps/$APPID.png"
done
install -Dm644 "$HERE/../icons/icon.svg" "$ROOT/usr/share/icons/hicolor/scalable/apps/$APPID.svg"

SIZE=$(du -sk "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<CTRL
Package: $PKG
Version: $VERSION
Section: graphics
Priority: optional
Architecture: $ARCH
Installed-Size: $SIZE
Maintainer: Simon <noreply@github.com>
Depends: libc6 (>= 2.35), libegl1, libgl1, libfontconfig1, libfreetype6, libdbus-1-3, libxkbcommon0, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, libxcb-image0, libxcb-keysyms1, libxcb-randr0, libxcb-render-util0, libxcb-shape0, libxcb-xinerama0, libxcb-xkb1, libglib2.0-0 | libglib2.0-0t64
Homepage: https://github.com/lordsimon666/Simon-Image-Enhance
Description: Lightroom-style photo editor with AI auto correction
 Non-destructive editing of JPEG, PNG and camera RAW files: white balance,
 tone controls, RGB tone curves and an AI correction that automatically sets
 white balance and tone curves. Exports JPEG and PNG.
CTRL
cat > "$ROOT/DEBIAN/postinst" <<'POST'
#!/bin/sh
set -e
command -v update-desktop-database >/dev/null && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
POST
cp "$ROOT/DEBIAN/postinst" "$ROOT/DEBIAN/postrm"
chmod 755 "$ROOT/DEBIAN/postinst" "$ROOT/DEBIAN/postrm"
find "$ROOT/opt" -type d -exec chmod 755 {} +

mkdir -p "$OUT"
DEB="$OUT/${PKG}_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group --build "$ROOT" "$DEB"
echo "$DEB"
