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
from .campaign import (
    ProfileNourishmentPlan,
    _Work,
    _Works,
    recovery_duration,
)
from .config import GeometryThresholds, NourishmentConfig
from .execute import run_campaign, run_scheduled_campaign

__all__ = [
    "FittedAssessor",
    "GeometricAssessor",
    "GeometryThresholds",
    "NourishmentConfig",
    "ProfileAssessment",
    "ProfileAssessor",
    "ProfileNourishmentPlan",
    "VolumeAssessor",
    "_Work",
    "_Works",
    "_depth_of_closure",
    "_select_assessor",
    "recovery_duration",
    "run_campaign",
    "run_scheduled_campaign",
]
