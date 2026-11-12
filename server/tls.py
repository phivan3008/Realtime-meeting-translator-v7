"""Trust the operating system's certificates for outgoing HTTPS.

The pipeline downloads its CPU checkpoints (ECAPA, VoxLingua107, AST) from
huggingface.co the first time it runs. Python verifies those connections
against ``certifi``'s bundle, not the Windows certificate store. On a network
that inspects HTTPS, the certificate huggingface.co presents is signed by the
company's own root, which Windows (and every browser) trusts and ``certifi``
does not, so every download fails with ``CERTIFICATE_VERIFY_FAILED``.

``truststore`` makes Python's ``ssl`` verify against the system store
instead. Verification stays on: a certificate nobody trusts is still refused.

This runs from ``server/__init__.py``, so every entry point that can download
a model - the app, the real tests, the analysis tools - gets it before the
first request. ``SYSTEM_CERTS=0`` (server/config.py) turns it off.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

_injected = False


def use_system_certificates(enabled: Optional[bool] = None) -> bool:
    """Verify HTTPS against the OS store. True when it is now in effect."""
    global _injected
    if enabled is None:
        from server.config import SYSTEM_CERTS
        enabled = SYSTEM_CERTS
    if not enabled:
        return False
    if _injected:
        return True
    try:
        import truststore
    except ImportError:
        log.debug("truststore not installed; HTTPS uses certifi's bundle")
        return False
    truststore.inject_into_ssl()
    _injected = True
    return True
