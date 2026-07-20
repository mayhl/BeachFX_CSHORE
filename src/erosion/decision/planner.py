"""The interval planner — pure placement scheduling over the crew clock.

``plan_placements`` is a fold: the crew clock threads through the campaign's
placements as plain state (seeded here — ``t_base`` on a resume, else ``t_base +
mobilization_days``), and every policy branch the old scheduler mixed with bed
mutation lives here as a decision: blackout shift, can't-start-before-the-storm
block, storm-conflict defer-vs-interrupt, partial fraction.  The comparison
operators are copied verbatim from the scheduler they replace — a flipped ``>=``
here is a silent behavior change, so treat them as load-bearing.

Physics-free: metrics in, decisions out.  The executor owns beds, snapshots,
recovery, and the audit emission of what this function returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import CampaignKind, DecisionKind
from .calendar import ActiveCampaign, CalendarState
from .model import (
    DeferBlackout,
    DeferCycle,
    DeferStorm,
    FireCycle,
    Interrupt,
    LaunchCampaign,
    Placement,
    PlanMetrics,
    SkipCampaign,
)

if TYPE_CHECKING:
    from ..nourishment.config import NourishmentConfig

Decision = Placement | DeferBlackout | DeferStorm | Interrupt


@dataclass
class ReachDecision:
    """Reach-scope campaign decision: go/no-go plus the placement order.

    ``kind`` is the ``DecisionKind`` to record (``NOURISH_TRIGGER`` /
    ``NOURISH_SKIP``); ``order`` is profile IDs in placement order (empty on skip).
    """

    mobilize: bool
    kind: DecisionKind
    total_deficit: float
    resume: bool  # a crew is already on site from a prior campaign
    forced: bool = False  # mobilized by an emergency geometric trigger, not the volume gate
    order: list[str] = field(default_factory=list)


class ReachNourishmentDecider:
    """Aggregates per-profile plan metrics into the reach campaign decision.

    Owns the reach-scope gate (deficit ≥ ``volume_trigger``, bypassed while a
    crew is already on site) and the placement order (priority desc; on resume,
    prior-campaign profiles first via a stable sort). The economic ($$) gate and
    the emergency ``force`` override arrive in a later phase; today the gate is
    the volume trigger — behaviour-preserving.
    """

    def decide(
        self,
        metrics: list[PlanMetrics],
        prior: ActiveCampaign | None,
        ncfg: NourishmentConfig,
        forced: bool = False,
    ) -> ReachDecision:
        """Mobilize when a crew is already on site (resume), the reach deficit meets
        the volume trigger, OR any profile raised the emergency geometric ``force``
        (``forced``).  ``forced`` bypasses the volume gate the way ``resume`` does.
        """
        total_deficit = sum(m.volume_m3 for m in metrics)
        resume = prior is not None and prior.crew_on_site
        # None volume_trigger = the regular gate is off (emergency-only reach).
        gate_met = ncfg.volume_trigger is not None and total_deficit >= ncfg.volume_trigger
        if not (resume or forced or gate_met):
            return ReachDecision(False, DecisionKind.NOURISH_SKIP, total_deficit, resume)

        # Equal-spacing: priority = deficit.  A placement-order policy, so it lives
        # here rather than on the assessor that measured the deficit.
        ordered = sorted(metrics, key=lambda m: -m.volume_m3)
        if prior is not None:
            rank = {pid: pos for pos, pid in enumerate(prior.priority_order)}
            # stable sort → priority order preserved within each prior rank
            ordered = sorted(ordered, key=lambda m: rank.get(m.profile_id, 999))
        # Emergency-only when the volume gate wasn't independently met.
        emergency = forced and not gate_met and not resume
        kind = DecisionKind.NOURISH_EMERGENCY if emergency else DecisionKind.NOURISH_TRIGGER
        return ReachDecision(
            True,
            kind,
            total_deficit,
            resume,
            forced=emergency,
            order=[m.profile_id for m in ordered],
        )


_DEFAULT_DECIDER = ReachNourishmentDecider()


def decide_campaign(
    metrics: list[PlanMetrics],
    prior: ActiveCampaign | None,
    ncfg: NourishmentConfig,
    forced: bool,
    origin: CampaignKind,
    t: float,
) -> LaunchCampaign | SkipCampaign:
    """The one campaign gate, for both origins.

    The post-storm response and the planned cycle differ in exactly one decision
    rule: a calendar mobilization records as ``NOURISH_CYCLE`` — unless an
    emergency forced it, which keeps its own kind since it would have mobilized
    the campaign either way.  Everything else (gate, resume, ordering) is shared,
    which is why the remap lives here and not in a second entry point.
    """
    d = _DEFAULT_DECIDER.decide(metrics, prior, ncfg, forced)
    if not d.mobilize:
        trigger = float(ncfg.volume_trigger) if ncfg.volume_trigger is not None else None
        return SkipCampaign(
            t=t,
            total_deficit=d.total_deficit,
            trigger=trigger,
            cycle=origin is CampaignKind.SCHEDULED,
        )
    kind = d.kind
    if origin is CampaignKind.SCHEDULED and kind is DecisionKind.NOURISH_TRIGGER:
        kind = DecisionKind.NOURISH_CYCLE
    return LaunchCampaign(
        kind=kind,
        t=t,
        total_deficit=d.total_deficit,
        resume=d.resume,
        forced=d.forced,
        order=tuple(d.order),
    )


def _next_available(
    t: float,
    duration: float,
    blackouts: list[tuple[float, float]],
) -> float:
    """Return earliest start ≥ t such that [start, start+duration) doesn't overlap any blackout."""
    while True:
        blocking = next(
            (bw_end for bw_start, bw_end in blackouts if t < bw_end and t + duration > bw_start),
            None,
        )
        if blocking is None:
            return t
        t = blocking


def plan_placements(
    order: list[PlanMetrics],
    t_base: float,
    t_next: float,
    ncfg: NourishmentConfig,
    resume: bool,
) -> tuple[list[Decision], ActiveCampaign | None]:
    """Schedule the campaign's placements serially within ``[t_base, t_next]``.

    ``t_base`` anchors the campaign (storm end, or the cycle date for a planned
    cycle); a resume puts the crew straight to work, a cold start pays the
    mobilization lead-time.  The fold stops at the first block or interrupt and
    returns the carry-forward campaign; ``None`` means the crew finished.
    """
    decisions: list[Decision] = []
    t_crew = t_base if resume else t_base + ncfg.mobilization_days
    for i, m in enumerate(order):
        borrow = m.placement_m3 * ncfg.borrow_to_placement_ratio
        duration = borrow / ncfg.production_rate  # days — dredged borrow, not restored geometry
        t_start = _next_available(t_crew, duration, list(ncfg.blackout_windows))
        if t_start > t_crew:
            decisions.append(
                DeferBlackout(
                    t=t_base, profile_id=m.profile_id, requested=t_crew, deferred_to=t_start
                )
            )
        if t_start >= t_next:  # can't start before the next storm
            remaining = [x.profile_id for x in order[i:]]
            return decisions, ActiveCampaign(crew_on_site=False, priority_order=remaining)

        t_end = t_start + duration
        storm_conflict = t_end >= t_next  # next storm would land during placement

        # DEFER policy (BeachFX): a placement the storm would hit is not started at all —
        # delay it to a storm-free window (the campaign carries to the next gap). No start
        # marker, no partial fill; the profile just recovers over this gap.
        if storm_conflict and ncfg.storm_conflict == "defer":
            decisions.append(
                DeferStorm(
                    t=t_base,
                    profile_id=m.profile_id,
                    would_start=t_start,
                    would_end=t_end,
                    storm=t_next,
                )
            )
            remaining = [x.profile_id for x in order[i:]]
            return decisions, ActiveCampaign(crew_on_site=False, priority_order=remaining)

        if storm_conflict:  # INTERRUPT policy: place what fits, resume after the storm
            fraction = (t_next - t_start) / duration
            decisions.append(Interrupt(t=t_next, profile_id=m.profile_id, placed_fraction=fraction))
            decisions.append(
                Placement(
                    profile_id=m.profile_id,
                    t_start=t_start,
                    t_end=t_next,
                    placed_m3=m.placement_m3 * fraction,
                    borrow_m3=borrow * fraction,
                    fraction=fraction,
                    cut_by_storm=True,
                )
            )
            remaining = [m.profile_id] + [x.profile_id for x in order[i + 1 :]]
            return decisions, ActiveCampaign(crew_on_site=True, priority_order=remaining)

        decisions.append(
            Placement(
                profile_id=m.profile_id,
                t_start=t_start,
                t_end=t_end,
                placed_m3=m.placement_m3,
                borrow_m3=borrow,
            )
        )
        t_crew = t_end
    return decisions, None


def erosion_split(cal: CalendarState, t_a: float, t_b: float, recovery_done: float) -> float:
    """Where the gap's one-pass erosion stops.

    A gap is normally eroded in ONE pass, before the campaign — so the campaign
    assesses a bed already eroded to the gap's end.  A planned cycle can't live with
    that: the storm campaign would restore the beach to the template and swallow the
    whole gap's erosion on the way, leaving the cycle nothing to find on its own date.
    So when a cycle is due, erosion holds at the campaign's own span (the last
    recovery completion) and the remainder lands after the cycle has assessed.
    """
    if cal.cycles.peek_due(t_b) is None:
        return t_b
    return max(t_a, min(t_b, recovery_done))


def plan_next_cycle(
    cal: CalendarState, t_b: float, recovery_done: float
) -> FireCycle | DeferCycle | None:
    """The next owed planned cycle in this gap: fire it, defer it, or nothing owed.

    A cycle fires at ``max(owed, recovery_done)``, not its calendar date: one landing
    while the reach is still recovering waits out the recovery (BeachFX's deferral
    code 1) rather than cutting it short.  A crew already carrying unfinished storm
    work owns the window, so the cycle yields to it; and a fire time at or past the
    gap's end has no room to start.  Either way the cycle stays owed and is retried
    in the next gap, blocking those behind it.

    Pure: peeks the tracker without committing — the caller commits (``next_due``,
    then ``clear`` once the fired campaign ran).
    """
    owed = cal.cycles.peek_due(t_b)
    if owed is None:
        return None
    t_fire = max(owed, recovery_done)
    if cal.campaign is not None or t_fire >= t_b:
        return DeferCycle(
            t=owed, would_fire=t_fire, gap_end=t_b, crew_busy=cal.campaign is not None
        )
    return FireCycle(t_fire=t_fire, erode_to=t_fire)
