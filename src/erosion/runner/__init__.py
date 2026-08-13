"""CSHORE execution boundary — the runner contract and its implementations.

``CSHORERunner`` (base.py) is the interface the storm loop calls, with
``MockCSHORERunner`` alongside it as the test stand-in; ``LocalCSHORERunner``
(local.py) runs the real binary in a subprocess via the vendored fixed-format
file I/O (cshore_io.py).  ``vfall.py`` computes the sediment fall-velocity
input.
"""

from .base import CSHOREResult, CSHORERunner, MockCSHORERunner
from .local import CSHOREParams, LocalCSHORERunner

__all__ = [
    "CSHOREParams",
    "CSHOREResult",
    "CSHORERunner",
    "LocalCSHORERunner",
    "MockCSHORERunner",
]
