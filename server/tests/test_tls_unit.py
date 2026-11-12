"""server/tls.py - HTTPS trusts the system store, verification stays on."""

from __future__ import annotations

import ssl
import sys
import types

import pytest

import server
from server import tls


@pytest.fixture
def fake_truststore(monkeypatch):
    calls = []
    module = types.ModuleType("truststore")
    module.inject_into_ssl = lambda: calls.append("inject")
    monkeypatch.setitem(sys.modules, "truststore", module)
    monkeypatch.setattr(tls, "_injected", False)
    return calls


def test_injects_once(fake_truststore):
    assert tls.use_system_certificates(True) is True
    assert tls.use_system_certificates(True) is True
    assert fake_truststore == ["inject"]


def test_can_be_turned_off(fake_truststore):
    assert tls.use_system_certificates(False) is False
    assert fake_truststore == []


def test_defaults_to_the_config_flag(fake_truststore, monkeypatch):
    import server.config
    monkeypatch.setattr(server.config, "SYSTEM_CERTS", False)
    assert tls.use_system_certificates() is False
    monkeypatch.setattr(server.config, "SYSTEM_CERTS", True)
    assert tls.use_system_certificates() is True


def test_on_by_default():
    import server.config
    assert "SYSTEM_CERTS" in server.config.known_variables()


def test_missing_truststore_is_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)   # import fails
    monkeypatch.setattr(tls, "_injected", False)
    assert tls.use_system_certificates(True) is False


def test_importing_the_package_turns_it_on():
    pytest.importorskip("truststore")
    assert server.tls._injected is True
    # Verification is still required: truststore swaps the trust source,
    # it does not accept everything.
    context = ssl.create_default_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
