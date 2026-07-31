"""CSHORE execution boundary — the runner contract and its implementations.

``CSHORERunner`` (base.py) is the interface the storm loop calls;
``LocalCSHORERunner`` (local.py) runs the real binary in a subprocess via the
vendored fixed-format file I/O (cshore_io.py), and ``MockCSHORERunner``
(mock.py) is the test stand-in.  ``vfall.py`` computes the sediment
fall-velocity input.
"""

from .base import CSHOREResult, CSHORERunner
from .local import CSHOREParams, LocalCSHORERunner
from .mock import MockCSHORERunner

__all__ = [
    "CSHOREParams",
    "CSHOREResult",
    "CSHORERunner",
    "LocalCSHORERunner",
    "MockCSHORERunner",
]
