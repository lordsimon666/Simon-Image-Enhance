"""HTTPS certificate store: the bundle is built on Ubuntu but must work on Fedora/Bazzite."""
import ssl

from simon_enhance import net


def _reset(monkeypatch):
    monkeypatch.setattr(net, "_ctx", None)
    monkeypatch.setattr(net, "_source", "")


def test_prefers_system_bundle(tmp_path, monkeypatch):
    import certifi
    fedora = tmp_path / "ca-bundle.crt"
    fedora.write_text(open(certifi.where()).read())
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(net, "SYSTEM_CA_FILES", (str(tmp_path / "missing.crt"), str(fedora)))
    assert net.ca_file() == str(fedora)
    _reset(monkeypatch)
    ctx = net.context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.cert_store_stats()["x509_ca"] > 50
    assert net.source() == str(fedora)


def test_env_overrides_and_certifi_fallback(tmp_path, monkeypatch):
    import certifi
    monkeypatch.setattr(net, "SYSTEM_CA_FILES", (str(tmp_path / "none.crt"),))
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert net.ca_file() == certifi.where()          # no system bundle at all: bundled store
    own = tmp_path / "proxy.pem"
    own.write_text(open(certifi.where()).read())
    monkeypatch.setenv("SSL_CERT_FILE", str(own))
    assert net.ca_file() == str(own)


def test_readable_error():
    msg = net.describe_error(OSError("<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed>"))
    assert msg.startswith("Die sichere Verbindung") and "SSL_CERT_FILE" in msg
