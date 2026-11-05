#!/usr/bin/env bash
# Build an AppImage from the PyInstaller bundle.
# Usage: packaging/linux/build_appimage.sh <bundle-dir> <version> [out-dir]
set -euo pipefail
BUNDLE=$(realpath "$1"); VERSION="$2"; OUT=$(realpath -m "${3:-dist}")
HERE=$(cd "$(dirname "$0")" && pwd)
APPID=io.github.lordsimon666.SimonImageEnhance
ARCH=$(uname -m)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
APPDIR="$WORK/SimonImageEnhance.AppDir"

install -d "$APPDIR/usr/lib/simon-image-enhance" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/metainfo"
cp -a "$BUNDLE/." "$APPDIR/usr/lib/simon-image-enhance/"
sed 's/^Exec=.*/Exec=SimonImageEnhance %F/' "$HERE/$APPID.desktop" > "$APPDIR/$APPID.desktop"
cp "$APPDIR/$APPID.desktop" "$APPDIR/usr/share/applications/"
cp "$HERE/$APPID.metainfo.xml" "$APPDIR/usr/share/metainfo/$APPID.appdata.xml"
cp "$HERE/../icons/icon-256.png" "$APPDIR/$APPID.png"
install -Dm644 "$HERE/../icons/icon-256.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APPID.png"
ln -s "$APPID.png" "$APPDIR/.DirIcon"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/simon-image-enhance/SimonImageEnhance" "$@"
RUN
chmod +x "$APPDIR/AppRun"

TOOL="${APPIMAGETOOL:-$WORK/appimagetool}"
if [ ! -x "$TOOL" ]; then
  curl -fsSL -o "$TOOL" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
  chmod +x "$TOOL"
fi
mkdir -p "$OUT"
DST="$OUT/SimonImageEnhance-$VERSION-$ARCH.AppImage"
ARCH=$ARCH APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --no-appstream "$APPDIR" "$DST"
echo "$DST"
