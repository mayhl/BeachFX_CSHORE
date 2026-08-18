from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from .interstorm import ErosionConfig, GenCadeErosionConfig, SLCConfig, UniformErosionConfig
from .nourishment import GeometryThresholds, NourishmentConfig
from .profile import Profile, ProfileGeometryConfig, load_raw_profile
from .runner.local import CSHOREParams
from .storm import StormConfig
from .units import parse_ufloat, ufloat

__all__ = [
    "ProfileGeometryConfig",
    "CSHOREParams",
    "StormConfig",
    "SLCConfig",
    "UniformErosionConfig",
    "GenCadeErosionConfig",
    "ErosionConfig",
    "GeometryThresholds",
    "NourishmentConfig",
    "ReachConfig",
]


class ReachConfig(BaseModel):
    """All policy parameters for one reach."""

    model_config = ConfigDict(extra="forbid")

    storm: StormConfig = Field(default_factory=StormConfig)
    cshore: CSHOREParams = Field(default_factory=CSHOREParams)
    erosion: ErosionConfig | None = None
    slc: SLCConfig | None = None
    nourishment: NourishmentConfig | None = None
    # Mean sea level (internal m; bare config floats read as ft like every other
    # elevation): the reference plane for subaerial volume / erosion metrics.
    # Constant in time — relative SLC is applied to the bed (see SLCConfig), so a
    # moving MSL would double-count it.
    msl: ufloat("m", "ft") = 0.0
    # Depth of closure (internal m, positive depth below MSL; bare floats = ft):
    # the seaward limit of the active profile.  Reach-wide default; a profile may
    # override it via ProfileGeometryConfig.depth_of_closure — the two levels now
    # share ONE unit convention (a bare 6.0 used to mean 6.0 m here but 1.83 m
    # there, a silent 3.28x on placement volume).  Used by the nourishment
    # placement model to extend the subaerial deficit down to the full active
    # height (see nourishment.VolumeAssessor / FittedAssessor).  0.0 = no extension.
    depth_of_closure: ufloat("m", "ft") = 0.0
    # Working-grid spacing (m).  Profiles are resampled onto an equipartitioned
    # grid of this dx at load; 1.0 matches CSHORE's internal regrid so the storm
    # response survives the round-trip (see load_raw_profile).  None = native.
    grid_dx: float | None = 1.0


# ---------------------------------------------------------------------------
# Raw-config resolution: layering, plan expansion, run selection, profile loading
# ---------------------------------------------------------------------------


def _require(mapping: dict, key: str, where: str):
    """Fetch a required config key, failing with the key's location instead of a bare KeyError."""
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(f"config is missing the required key {key!r} ({where})") from None


def _resolve_widths(w_raw, priority_order: list[int], context: dict) -> list[float]:
    """Per-profile longshore widths (metres) in priority-reordered list order."""
    n = len(priority_order)
    if isinstance(w_raw, list):
        parsed = [parse_ufloat(v, "m", "ft", context) for v in w_raw]
        return [parsed[i] for i in priority_order]
    w_m = parse_ufloat(w_raw, "m", "ft", context)
    return [w_m / n] * n


# ReachConfig sections that layer default->override across global -> reach -> alt.
# nourishment is deliberately excluded: it stays alt-level here (schema-v2 Phase 2
# feeds it via the plan lowering), leaving the top-level `nourishment` key free for
# the plan table.
_LAYERED_SECTIONS = ("storm", "cshore", "erosion", "slc", "msl", "depth_of_closure")


def _merge_sections(*levels: dict) -> dict:
    """Layer ``_LAYERED_SECTIONS`` default->override across levels (global, reach, alt).

    A dict section present at more than one level is shallow-merged per key (later
    levels win); a scalar or a section present at a single level passes through.
    Non-section keys (profiles, alternatives, ...) are ignored.  Generalises the
    former cshore-only base->reach->alt layering to storm/erosion/slc as well.
    """
    out: dict = {}
    for level in levels:
        for section in _LAYERED_SECTIONS:
            val = level.get(section)
            if val is None:
                continue
            if isinstance(val, dict) and isinstance(out.get(section), dict):
                out[section] = {**out[section], **val}
            else:
                out[section] = val
    return out


def _merge_alt(global_alt: dict, reach_alt: dict) -> dict:
    merged: dict = {}
    for key in set(global_alt) | set(reach_alt):
        g = global_alt.get(key)
        r = reach_alt.get(key)
        if isinstance(g, dict) or isinstance(r, dict):
            merged[key] = {**(g or {}), **(r or {})}
        else:
            merged[key] = r if r is not None else g
    return merged


def _resolve_alternatives(global_alts: dict, reach_alts: dict) -> dict:
    all_ids = set(global_alts) | set(reach_alts)
    return {
        alt_id: _merge_alt(global_alts.get(alt_id, {}), reach_alts.get(alt_id, {}))
        for alt_id in all_ids
    }


def _resolve_reach_plans(reach_id: str, spec, global_plans: dict) -> dict:
    """Resolve one reach's nourishment spec to ``{plan_id: nourishment_config}``.

    ``spec`` is a **list of plan ids** (references into ``global_plans``) or an inline
    ``{plan_id: config}`` **dict** (config shallow-merged over the global entry; ``{}``
    means "use the global entry as-is").  A plan id that resolves to neither a global
    entry nor a non-empty inline config is a dangling reference -> error.
    """
    if isinstance(spec, list):
        resolved = {}
        for key in spec:
            if key not in global_plans:
                raise ValueError(
                    f"reach {reach_id!r}: nourishment plan {key!r} not in the global "
                    f"`nourishment` table (available: {sorted(global_plans)})"
                )
            resolved[key] = dict(global_plans[key])
        return resolved
    if isinstance(spec, dict):
        resolved = {}
        for key, inline in spec.items():
            merged = {**global_plans.get(key, {}), **(inline or {})}
            if not merged:
                raise ValueError(
                    f"reach {reach_id!r}: nourishment plan {key!r} is empty and not in "
                    f"the global `nourishment` table"
                )
            resolved[key] = merged
        return resolved
    raise ValueError(
        f"reach {reach_id!r}: nourishment must be a list of plan ids or a "
        f"{{plan_id: config}} dict, got {type(spec).__name__}"
    )


def _expand_plans(cfg_raw: dict) -> dict:
    """Lower schema-v2 nourishment-plan authoring into the alternatives table.

    Reads the optional global ``nourishment: {plan_id: config}`` table and each
    reach's ``nourishment`` (a list of plan ids, or an inline ``{plan_id: config}``
    dict) and rewrites each reach's ``alternatives`` to
    ``{plan_id: {"nourishment": <resolved>}}`` — the alternative id **is** the plan
    id.  Phase-1 section layering then supplies storm/erosion/slc from global/reach
    level.  A config that uses no plan authoring is returned untouched (v1 path).

    Rules (schema v2): mutually exclusive with a hand-written ``alternatives`` table;
    dangling plan id -> error; full coverage -> every reach must declare the same
    plan-id set (no silent auto-fill).
    """
    reaches = cfg_raw["reaches"]
    global_plans = cfg_raw.get("nourishment", {})
    reach_uses_plans = any("nourishment" in r for r in reaches.values())
    if not global_plans and not reach_uses_plans:
        return cfg_raw  # v1 path — no plan authoring

    if "alternatives" in cfg_raw or any("alternatives" in r for r in reaches.values()):
        raise ValueError(
            "nourishment-plan authoring cannot be combined with an `alternatives` "
            "table; use one or the other (cartesian sweeps stay offline)"
        )

    plan_sets: dict[str, set] = {}
    for reach_id, reach_data in reaches.items():
        spec = reach_data.get("nourishment")
        if spec is None:
            raise ValueError(
                f"reach {reach_id!r}: no nourishment plans declared; every reach must "
                f"define the full plan set (name the current-schedule plan explicitly)"
            )
        resolved = _resolve_reach_plans(reach_id, spec, global_plans)
        reach_data["alternatives"] = {pid: {"nourishment": cfg} for pid, cfg in resolved.items()}
        reach_data.pop("nourishment", None)
        plan_sets[reach_id] = set(resolved)

    universe = set().union(*plan_sets.values())
    for reach_id, ids in plan_sets.items():
        missing = universe - ids
        if missing:
            raise ValueError(
                f"reach {reach_id!r} missing plan id(s) {sorted(missing)}; every reach "
                f"must define the same plan set {sorted(universe)}"
            )

    cfg_raw.pop("nourishment", None)  # global plan table consumed
    return cfg_raw


def _parse_run_spec(spec, available: list[str]) -> list[str]:
    """Resolve a run selector to the subset of alternative ids to run.

    ``spec`` is ``"all"`` (every id), or a comma-separated list of integer ids,
    ``N-M`` inclusive ranges, and/or explicit ids — e.g. ``"1-4,8,11-14,FWP"``.
    Config JSON keys are strings, so an integer token ``3`` matches the key
    ``"3"``.  Unknown ids raise (no silent skip).  Returns the selected ids in
    ``available`` order (parallel run order is irrelevant; this keeps output
    deterministic).
    """
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "all"):
        return list(available)
    tokens = spec.split(",") if isinstance(spec, str) else [str(t) for t in spec]
    selected: set[str] = set()
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"run: descending range {tok!r}")
            selected.update(str(i) for i in range(lo, hi + 1))
        else:
            selected.add(tok)
    unknown = sorted(selected - set(available), key=str)
    if unknown:
        raise ValueError(
            f"run: unknown alternative id(s) {unknown}; available: {sorted(available)}"
        )
    return [a for a in available if a in selected]


def _priority_order(priorities: list[int] | None, n: int) -> list[int]:
    """Return indices 0..n-1 sorted by priority (lower number = first); default is input order."""
    prios = priorities if priorities is not None else list(range(n))
    if len(prios) != n:
        raise ValueError(f"profile_priority has {len(prios)} entries but profiles has {n}")
    return sorted(range(n), key=lambda i: prios[i])


def _load_profiles(
    profile_paths: list[str],
    d50: float,
    reach_id: str,
    priorities: list[int] | None = None,
    geometry: ProfileGeometryConfig | None = None,
    grid_dx: float | None = None,
) -> list[Profile]:
    """Load profiles sorted by priority (lower number = first); IDs use list index."""
    order = _priority_order(priorities, len(profile_paths))
    profiles = []
    for i in order:
        raw = load_raw_profile(profile_paths[i], d50, dx=grid_dx)
        profiles.append(
            Profile(
                id=f"{reach_id}_p{i}",
                x=raw["x"],
                zb=raw["z"].copy(),
                d50=raw["d50"],
                geometry=geometry,
            )
        )
    return profiles


def _fresh_profiles(profiles: list[Profile]) -> list[Profile]:
    return [
        Profile(id=p.id, x=p.x.copy(), zb=p.zb.copy(), d50=p.d50, geometry=p.geometry)
        for p in profiles
    ]
