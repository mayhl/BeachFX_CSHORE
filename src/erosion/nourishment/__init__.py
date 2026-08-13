"""Nourishment physics and execution: Tier-1 assessment and the campaign executor.

The decision layer (gate, ordering, calendar, placement planning) lives in
``erosion.decision``; this package owns the assessors, the restore synthesis,
the work bundles, and the crew that applies what the planner decided.
"""

from .assess import (
    FittedAssessor,
    GeometricAssessor,
    ProfileAssessment,
    ProfileAssessor,
    VolumeAssessor,
    _depth_of_closure,
    _select_assessor,
)
from .config import GeometryThresholds, NourishmentConfig
from .execute import (
    FillSpec,
    WorkItem,
    Workset,
    recovery_duration,
    run_campaign,
    run_planned_campaign,
)

__all__ = [
    "FittedAssessor",
    "GeometricAssessor",
    "GeometryThresholds",
    "NourishmentConfig",
    "ProfileAssessment",
    "ProfileAssessor",
    "FillSpec",
    "VolumeAssessor",
    "WorkItem",
    "Workset",
    "_depth_of_closure",
    "_select_assessor",
    "recovery_duration",
    "run_campaign",
    "run_planned_campaign",
]
