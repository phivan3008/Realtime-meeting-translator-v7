"""The meeting pipeline.

Importing the package makes outgoing HTTPS trust the operating system's
certificates (see ``server/tls.py``), before anything downloads a model.
"""

from server.tls import use_system_certificates

use_system_certificates()
