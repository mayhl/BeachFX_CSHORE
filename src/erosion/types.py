from __future__ import annotations

from enum import Enum


class StormResponseType(int, Enum):
    NORMAL = 0
    CAT_DUNE_LOST = 1  # dune eroded below max(UE, BE); LOW_BERM pre-storm
    CAT_PARTIAL = 2  # dune gone, berm survives; LOW_UPLAND pre-storm
    CAT_TOTAL = 3  # dune and berm both gone; LOW_UPLAND pre-storm
    INUNDATION = 4  # CSHORE failed — no morphology output produced


class SnapshotLabel(Enum):
    INIT = "INIT"
    PreStorm = "PreStorm"
    PostStorm = "PostStorm"
    INUNDATION = "INUNDATION"  # CSHORE failure — profile unchanged
    RECS = "RECS"
    REC = "REC"
    Pre_PDI = "Pre-PDI"
    Post_PDI = "Post-PDI"
    SSN = "SSN"
    ESN = "ESN"
    SEN = "SEN"
    EEN = "EEN"
    EndIteration = "EndIteration"
    Periodic = "Periodic"


class DecisionKind(Enum):
    """Reach/SIM-scope orchestrator *decisions* — distinct from the profile-scope
    morphology ``SnapshotLabel``s. Unified only at the event log's ``event_id``.
    """

    NOURISH_TRIGGER = "NOURISH_TRIGGER"  # reach deficit ≥ trigger → launch campaign
    NOURISH_SKIP = "NOURISH_SKIP"  # deficit < trigger → recovery only
    MOBILIZE = "MOBILIZE"  # crew lead-time before it can start
    BLACKOUT_DEFER = "BLACKOUT_DEFER"  # placement start pushed past a blackout window
    INTERRUPT = "INTERRUPT"  # next storm cut the campaign (partial placement)
    WINDOW_EXTENDED = "WINDOW_EXTENDED"  # sim window extended to finish a campaign
