"""Nourishment: Tier-1 assessment, Tier-2 reach decision, Tier-3 scheduling.

Split into submodules (config / assess / campaign / decide / schedule); this
re-exports the public API so ``from erosion.nourishment import X`` keeps working.
"""

from .assess import (
    FittedAssessor,
    GeometricAssessor,
    ProfileAssessment,
    ProfileAssessor,
    VolumeAssessor,
    _depth_of_closure,
    _interp_template,
    _select_assessor,
)
from .campaign import ActiveCampaign, ProfileNourishmentPlan, _Work, _Works
from .config import GeometryThresholds, NourishmentConfig
from .decide import ReachDecision, ReachNourishmentDecider
from .schedule import _next_available, run_campaign

__all__ = [
    "ActiveCampaign",
    "FittedAssessor",
    "GeometricAssessor",
    "GeometryThresholds",
    "NourishmentConfig",
    "ProfileAssessment",
    "ProfileAssessor",
    "ProfileNourishmentPlan",
    "ReachDecision",
    "ReachNourishmentDecider",
    "VolumeAssessor",
    "_Work",
    "_Works",
    "_depth_of_closure",
    "_interp_template",
    "_next_available",
    "_select_assessor",
    "run_campaign",
]
