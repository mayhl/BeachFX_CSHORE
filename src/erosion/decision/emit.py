"""The one path from a reach-scope decision to the audit sink and the console."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..results import ResultsSink
    from ..types import DecisionKind

log = logging.getLogger(__name__)


def emit(sink: ResultsSink, kind: DecisionKind, t: float, **payload) -> None:
    """Record one decision to the structured audit AND the operational console.

    Every decision goes through here: a second emission path lets the parquet and
    the log drift apart (reach.py's CYCLE_DEFER once wrote the parquet row but
    never the console line).
    """
    sink.record_decision(kind, t, **payload)
    log.info("decision t=%.1fd — %s %s", t, kind.value, payload)
