"""The decision vocabulary — what the interval planner says, as plain data.

Scalars and tuples only, picklable by construction; a decision that persists to
``decisions.parquet`` carries its ``DecisionKind`` plus a ``row()`` reproducing
exactly the payload keys the sink already writes, so adopting the vocabulary
changes no output.  ``Placement`` and ``FireCycle`` are executor instructions —
they drive physics but are not themselves audit rows (a placement's audit trail
is the nourishment segment it records, a fired cycle's is its NOURISH_* row).

The planner that produces these lands with the crew-clock fold and the interval
inversion; until then the campaign path emits its decisions inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from ..types import DecisionKind

if TYPE_CHECKING:
    from .calendar import CampaignCarryover


@dataclass(frozen=True)
class ProfileDemand:
    """Scalar view of one profile's Tier-1 assessment — the decider's entire input.

    The decision layer never sees a bed or a template: a profile is a deficit, a
    placement volume, and whether its dune tripped the emergency.  Keeping this
    boundary scalar is what keeps the decider pure (and picklable).
    """

    profile_id: str
    deficit_m3: float  # subaerial deficit (m³) — the trigger volume
    placement_m3: float  # full active-height volume placed (borrow = ×ratio)
    force: bool = False  # emergency geometric trigger fired (Tier-1)


@dataclass(frozen=True)
class Placement:
    """One profile's scheduled fill — an executor instruction, not an audit row.

    ``cut_by_storm`` marks a storm-cut partial (the executor places ``fraction``
    of the template and closes the segment as EENS/ESNS); it is an explicit flag
    rather than ``fraction < 1.0`` because a placement ending exactly at the next
    storm is a partial with fraction 1.0."""

    profile_id: str
    t_start: float
    t_end: float
    placed_m3: float
    borrow_m3: float
    fraction: float = 1.0
    cut_by_storm: bool = False


@dataclass(frozen=True)
class LaunchCampaign:
    """The reach mobilizes: kind, order, and the placement schedule."""

    kind: DecisionKind  # NOURISH_TRIGGER / NOURISH_EMERGENCY / NOURISH_CYCLE
    t: float
    total_deficit: float
    resume: bool
    forced: bool
    order: tuple[str, ...]  # profile IDs in placement order
    placements: tuple = ()  # Placement | DeferBlackout | DeferStorm | Interrupt
    carry_forward: CampaignCarryover | None = None

    def row(self) -> dict:
        return {"deficit": self.total_deficit, "resume": self.resume, "forced": self.forced}


@dataclass(frozen=True)
class SkipCampaign:
    """The gate held: nothing placed, recovery only."""

    kind: ClassVar[DecisionKind] = DecisionKind.NOURISH_SKIP
    t: float
    total_deficit: float
    trigger: float | None
    cycle: bool = False  # the scheduled path marks its skips

    def row(self) -> dict:
        base = {"deficit": self.total_deficit, "trigger": self.trigger}
        return {"cycle": True, **base} if self.cycle else base


@dataclass(frozen=True)
class Interrupt:
    """The next storm cut a placement mid-fill (interrupt policy)."""

    kind: ClassVar[DecisionKind] = DecisionKind.INTERRUPT
    t: float  # the interrupting storm — the time the decision concerns
    profile_id: str
    placed_fraction: float

    def row(self) -> dict:
        return {"placed_fraction": self.placed_fraction}


@dataclass(frozen=True)
class DeferBlackout:
    """A placement start pushed past a blackout window."""

    kind: ClassVar[DecisionKind] = DecisionKind.BLACKOUT_DEFER
    t: float
    profile_id: str
    requested: float
    deferred_to: float

    def row(self) -> dict:
        return {"requested": self.requested, "deferred_to": self.deferred_to}


@dataclass(frozen=True)
class DeferStorm:
    """A storm-conflicting placement not started at all (defer policy)."""

    kind: ClassVar[DecisionKind] = DecisionKind.STORM_DEFER
    t: float
    profile_id: str
    would_start: float
    would_end: float
    storm: float

    def row(self) -> dict:
        return {"would_start": self.would_start, "would_end": self.would_end, "storm": self.storm}


@dataclass(frozen=True)
class DeferCycle:
    """An owed planned cycle pushed out of this gap (crew busy or no room)."""

    kind: ClassVar[DecisionKind] = DecisionKind.CYCLE_DEFER
    t: float  # the owed cycle date — the time the decision concerns
    would_fire: float
    gap_end: float
    crew_busy: bool

    def row(self) -> dict:
        return {"would_fire": self.would_fire, "gap_end": self.gap_end, "crew_busy": self.crew_busy}


@dataclass(frozen=True)
class FireCycle:
    """Run the owed cycle at ``t_fire`` — an executor instruction; the campaign it
    launches emits its own NOURISH_CYCLE / NOURISH_SKIP audit row."""

    t_fire: float
    erode_to: float  # bring the bed to this time before the crew assesses it
