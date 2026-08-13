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
    RECS = "RECS"  # recovery cut short by the next storm
    REC = "REC"  # recovery ran its full period (or to the sim/window end)
    RECN = "RECN"  # recovery cut short by the crew arriving to nourish (Python-model label)
    Pre_PDI = "Pre-PDI"
    Post_PDI = "Post-PDI"
    SSN = "SSN"
    ESN = "ESN"
    SEN = "SEN"
    EEN = "EEN"
    # A campaign a storm cut mid-placement (partial fill).  Same grammar as RECS: the
    # base names the segment, the trailing S names the storm that ended it.  Python-model
    # labels — BeachFX defers a placement a storm would hit, so it has no partial state.
    EENS = "EENS"  # storm-triggered campaign, cut short by the next storm
    ESNS = "ESNS"  # planned cycle, cut short by the next storm
    EndIteration = "EndIteration"
    Periodic = "Periodic"


class CampaignKind(Enum):
    """What launched a nourishment campaign — fixes the morphology label pair.

    BeachFX keeps the two apart: a *planned* cycle is calendar-driven
    (``ProcessStartScheduledNourishmentCycle``) and writes ``SSN``/``ESN``, while a
    post-storm response comes off the damage triggers
    (``CheckEmergencyNourishmentTriggers``) and writes ``SEN``/``EEN``.  Our port
    runs both through one crew/scheduler, so the kind is what selects the labels.
    """

    STORM = "storm"  # post-storm damage response — deficit or emergency trigger
    PLANNED = "planned"  # periodic planned cycle — calendar-driven

    @property
    def start_label(self) -> SnapshotLabel:
        return SnapshotLabel.SEN if self is CampaignKind.STORM else SnapshotLabel.SSN

    @property
    def end_label(self) -> SnapshotLabel:
        return SnapshotLabel.EEN if self is CampaignKind.STORM else SnapshotLabel.ESN

    @property
    def partial_label(self) -> SnapshotLabel:
        """End marker for a campaign a storm cut mid-placement."""
        return SnapshotLabel.EENS if self is CampaignKind.STORM else SnapshotLabel.ESNS


class DecisionKind(Enum):
    """Reach/SIM-scope *decisions* — distinct from the profile-scope
    morphology ``SnapshotLabel``s. Unified only at the event log's ``event_id``.
    """

    NOURISH_TRIGGER = "NOURISH_TRIGGER"  # reach deficit ≥ trigger → launch campaign
    NOURISH_SKIP = "NOURISH_SKIP"  # deficit < trigger → recovery only
    NOURISH_EMERGENCY = "NOURISH_EMERGENCY"  # dune below geometric trigger → forced mobilization
    NOURISH_CYCLE = "NOURISH_CYCLE"  # periodic cycle came due and cleared the volume gate
    CYCLE_DEFER = "CYCLE_DEFER"  # cycle pushed past an in-progress recovery or into a later gap
    BLACKOUT_DEFER = "BLACKOUT_DEFER"  # placement start pushed past a blackout window
    INTERRUPT = "INTERRUPT"  # next storm cut the campaign (partial place) — interrupt policy
    STORM_DEFER = "STORM_DEFER"  # start delayed past a storm-conflicting window — defer policy
