"""Tier-2 reach decision: the mobilization gate and placement ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..types import DecisionKind

if TYPE_CHECKING:
    from .campaign import ActiveCampaign, _Work
    from .config import NourishmentConfig


@dataclass
class ReachDecision:
    """Reach-scope campaign decision: go/no-go plus the placement order.

    ``kind`` is the ``DecisionKind`` to record (``NOURISH_TRIGGER`` /
    ``NOURISH_SKIP``); ``order`` is the plans in placement order (empty on skip).
    """

    mobilize: bool
    kind: DecisionKind
    total_deficit: float
    resume: bool  # a crew is already on site from a prior campaign
    forced: bool = False  # mobilized by an emergency geometric trigger, not the volume gate
    order: list[_Work] = field(default_factory=list)


class ReachNourishmentDecider:
    """Aggregates per-profile plans into the reach campaign decision.

    Owns the reach-scope gate (deficit ≥ ``volume_trigger``, bypassed while a
    crew is already on site) and the placement order (priority desc; on resume,
    prior-campaign profiles first via a stable sort). The economic ($$) gate and
    the emergency ``force`` override arrive in a later phase; today the gate is
    the volume trigger — behaviour-preserving.
    """

    def decide(
        self,
        plans: list[_Work],
        prior: ActiveCampaign | None,
        ncfg: NourishmentConfig,
        forced: bool = False,
    ) -> ReachDecision:
        """Mobilize when a crew is already on site (resume), the reach deficit meets
        the volume trigger, OR any profile raised the emergency geometric ``force``
        (``forced``).  ``forced`` bypasses the volume gate the way ``resume`` does.
        """
        total_deficit = sum(w.plan.volume_m3 for w in plans)
        resume = prior is not None and prior.crew_on_site
        gate_met = total_deficit >= ncfg.volume_trigger
        if not (resume or forced or gate_met):
            return ReachDecision(False, DecisionKind.NOURISH_SKIP, total_deficit, resume)

        ordered = sorted(plans, key=lambda w: -w.plan.priority_score)
        if prior is not None:
            rank = {pid: pos for pos, pid in enumerate(prior.priority_order)}
            # stable sort → priority order preserved within each prior rank
            ordered = sorted(ordered, key=lambda w: rank.get(w.profile.id, 999))
        # Emergency-only when the volume gate wasn't independently met.
        emergency = forced and not gate_met and not resume
        kind = DecisionKind.NOURISH_EMERGENCY if emergency else DecisionKind.NOURISH_TRIGGER
        return ReachDecision(True, kind, total_deficit, resume, forced=emergency, order=ordered)


_DEFAULT_DECIDER = ReachNourishmentDecider()
