#!/usr/bin/env bash
# Build an .rpm (Fedora, openSUSE, RHEL) from the PyInstaller bundle.
# Needs rpmbuild (Fedora: dnf install rpm-build, Ubuntu: apt install rpm).
# Usage: packaging/linux/build_rpm.sh <bundle-dir> <version> [out-dir]
set -euo pipefail
BUNDLE=$(realpath "$1"); VERSION="$2"; OUT=$(realpath -m "${3:-dist}")
HERE=$(cd "$(dirname "$0")" && pwd)
TOP=$(mktemp -d)
trap 'rm -rf "$TOP"' EXIT
mkdir -p "$TOP"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
sed -e "s|@VERSION@|$VERSION|g" -e "s|@BUNDLE@|$BUNDLE|g" -e "s|@HERE@|$HERE|g" \
    "$HERE/simon-image-enhance.spec.in" > "$TOP/SPECS/simon-image-enhance.spec"
rpmbuild --define "_topdir $TOP" --define "dist %{nil}" -bb "$TOP/SPECS/simon-image-enhance.spec" >&2
mkdir -p "$OUT"
DST="$OUT/simon-image-enhance.x86_64.rpm"
cp "$TOP"/RPMS/x86_64/*.rpm "$DST"
echo "$DST"
