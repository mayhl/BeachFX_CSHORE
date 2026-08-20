from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .profile import ErosionTick
from .units import ufloat

if TYPE_CHECKING:
    from .config import ReachConfig
    from .profile import Profile

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SLCConfig(BaseModel):
    """Sea-level change policy for one reach.

    ``rate`` is in m/day (positive = rising sea level, which lowers the bed equilibrium).
    ``per_profile_rates`` overrides ``rate`` for specific profiles.
    """

    model_config = ConfigDict(extra="forbid")

    rate: float = 0.0
    per_profile_rates: dict[str, float] = {}
    tick_days: ufloat("days") = 30.0
    scenario: str = "mid"  # label carried into output metadata


class UniformErosionConfig(BaseModel):
    """Uniform (or per-profile) background erosion applied as a fixed rate.

    ``rate`` is in m/day.  ``per_profile_rates`` maps profile_id → rate and,
    if non-empty, takes precedence over ``rate`` for those profiles.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["uniform"] = "uniform"
    rate: float = 0.0
    per_profile_rates: dict[str, float] = {}
    tick_days: ufloat("days") = 30.0


class GenCadeErosionConfig(BaseModel):
    """Placeholder for a future GenCade-driven longshore transport erosion model.

    Params TBD — discriminator keeps the schema extensible without changing
    ``ReachConfig`` once GenCade is implemented.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["gencade"] = "gencade"


ErosionConfig = Annotated[
    UniformErosionConfig | GenCadeErosionConfig,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _erosion_rate(profile: Profile, ecfg: UniformErosionConfig | None) -> float:
    if ecfg is None:
        return 0.0
    return ecfg.per_profile_rates.get(profile.id, ecfg.rate)


def _slc_rate(profile: Profile, slc: SLCConfig | None) -> float:
    if slc is None:
        return 0.0
    return slc.per_profile_rates.get(profile.id, slc.rate)


def _apply_tick(
    profile: Profile,
    dt: float,
    t_tick: float,
    ecfg: UniformErosionConfig | None,
    slc: SLCConfig | None,
) -> None:
    ErosionTick(
        t=t_tick,
        dz_erosion=_erosion_rate(profile, ecfg) * dt,
        dz_slc=_slc_rate(profile, slc) * dt,
    ).apply(profile)


# ---------------------------------------------------------------------------
# Phase 1 runner
# ---------------------------------------------------------------------------


def run_interstorm(
    profiles: list[Profile],
    t_start: float,
    t_end: float,
    cfg: ReachConfig,
    catch_up: bool = False,
) -> None:
    """Apply erosion + SLC ticks to all profiles from t_start to t_end.

    Each tick fires at the minimum of the two configured intervals.  Per-tick
    work is GIL-bound numpy + metric fitting with no cross-profile dependency,
    so profiles are processed serially within each tick.

    ``catch_up`` lands the whole window as ONE tick stamped at ``t_end`` — for
    ticks held back during a recovery (they must not precede the blend, which
    would overwrite them below the berm, and their stamps must not fall inside
    the recovery span).
    """
    if t_end <= t_start:
        return

    ecfg: UniformErosionConfig | None = (
        cfg.erosion if isinstance(cfg.erosion, UniformErosionConfig) else None
    )
    slc = cfg.slc

    interval = min(
        ecfg.tick_days if ecfg is not None else math.inf,
        slc.tick_days if slc is not None else math.inf,
    )
    if math.isinf(interval):
        return  # no erosion or SLC configured

    if catch_up:
        for p in profiles:
            _apply_tick(p, t_end - t_start, t_end, ecfg, slc)
        return

    t = t_start
    while t < t_end:
        dt = min(interval, t_end - t)
        t_tick = t + dt
        for p in profiles:
            _apply_tick(p, dt, t_tick, ecfg, slc)
        t = t_tick

    log.debug("run_interstorm: %.1fd → %.1fd (%d profiles)", t_start, t_end, len(profiles))
