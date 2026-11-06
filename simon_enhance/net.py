"""HTTPS downloads that work on every Linux distribution.

The Linux bundles are built on Ubuntu, so the bundled OpenSSL looks for the root
certificates in Ubuntu's places (/usr/lib/ssl). Fedora, Bazzite, openSUSE, Arch ...
keep them elsewhere, which made every download fail with CERTIFICATE_VERIFY_FAILED.
This module loads the system's CA bundle explicitly (so company / self-added
certificates keep working) and falls back to the bundled certifi store.
"""

from __future__ import annotations

import os
import ssl
import urllib.request

# CA bundles of the common distributions (first existing one wins)
SYSTEM_CA_FILES = (
    "/etc/pki/tls/certs/ca-bundle.crt",                    # Fedora, Bazzite, RHEL
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",   # Fedora (extracted)
    "/etc/ssl/certs/ca-certificates.crt",                  # Debian, Ubuntu, Arch, Gentoo, Flatpak runtimes
    "/etc/ssl/ca-bundle.pem",                              # openSUSE
    "/var/lib/ca-certificates/ca-bundle.pem",              # openSUSE
    "/etc/ssl/cert.pem",                                   # Alpine, macOS, BSD
)

_ctx: ssl.SSLContext | None = None
_source = ""


def ca_file() -> str | None:
    """Certificate bundle to use: $SSL_CERT_FILE, the system bundle or certifi's."""
    env = os.environ.get("SSL_CERT_FILE")
    if env and os.path.isfile(env):
        return env
    for f in SYSTEM_CA_FILES:
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            return f
    try:
        import certifi  # bundled with the app
        f = certifi.where()
        return f if os.path.isfile(f) else None
    except Exception:  # noqa: BLE001
        return None


def context() -> ssl.SSLContext:
    global _ctx, _source
    if _ctx is None:
        ctx = ssl.create_default_context()
        f = ca_file()
        if f:
            try:
                ctx.load_verify_locations(cafile=f)
                _source = f
            except (OSError, ssl.SSLError):
                pass
        _ctx = ctx
    return _ctx


def source() -> str:
    context()
    return _source or "OpenSSL-Standard"


def urlopen(url, timeout=60):
    """urllib.request.urlopen with a working certificate store (same signature as
    used throughout the app: opener(url, timeout=...))."""
    return urllib.request.urlopen(url, timeout=timeout, context=context())


def describe_error(exc: BaseException) -> str:
    """Readable German message for download errors."""
    text = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" in text:
        return ("Die sichere Verbindung konnte nicht geprüft werden (Zertifikat unbekannt). "
                f"Verwendete Zertifikate: {source()}. Ist ein Proxy oder Virenscanner mit eigenem "
                "Zertifikat aktiv, dessen Datei über die Umgebungsvariable SSL_CERT_FILE angeben.\n\n"
                f"Details: {text}")
    if "urlopen error" in text or isinstance(exc, (TimeoutError, ConnectionError)):
        return f"Download fehlgeschlagen – besteht eine Internetverbindung?\n\nDetails: {text}"
    return text
