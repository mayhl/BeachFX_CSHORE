"""Recovery-gallery support: run synthetic post→pre recovery cases, serialize the
recovery parameters, and regenerate the diagnostic fan from those parameters alone.

Mirrors ``fit_gallery``.  A *golden* is a small JSON per case holding everything
needed to reproduce both the pre/post beds and the recovered fan without re-running
the recovery mechanics:

    - ``pre_spec`` / ``post_spec`` : ``make_profile`` kwargs (deterministic w/ ``seed``),
      built on the SAME fixed grid so the blend is node-aligned (no shoreline shift)
    - ``model`` / ``T_recover`` / ``z_berm`` : recovery parameters
    - ``sample_days``           : elapsed days at which recovery is sampled
    - ``fractions``             : recovery fraction at each sample day (the JSON *is*
      the recovery: ``recovered = post + f·(pre − post)`` below ``z_berm``)
    - ``volumes``               : ``volume_above_datum`` of each recovered bed — a
      scalar check that survives independent of the full arrays

``run_case`` recomputes the fractions from ``sample_days`` via the real
``storm._recovery_fraction`` and the recovered beds via the real
``profile.recovered_bed``, so the gallery exercises the shipping math, not a copy.
"""

from __future__ import annotations

import dataclasses
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from erosion.metrics import volume_above_datum
from erosion.profile import recovered_bed
from erosion.storm import _recovery_fraction
from tests.synthetic import DuneSpec, make_profile

ROOT = os.path.dirname(__file__)
GOLDEN_DIR = os.path.join(ROOT, "goldens", "recovery")
ARTIFACT_DIR = os.path.join(ROOT, "_artifacts", "recovery")


@dataclasses.dataclass
class RecoveryCase:
    """A pre/post-storm profile pair plus the recovery parameters to blend them.

    ``pre_spec`` / ``post_spec`` are ``make_profile`` kwargs; both MUST share the
    grid (``dx``, ``x_max``) so ``x`` is identical and the blend is node-aligned.
    """

    name: str
    pre_spec: dict
    post_spec: dict
    model: str = "linear"  # "linear" | "exponential"
    T_recover: float = 21.0  # days (T100 linear / T90 exponential)
    z_berm: float | None = None
    sample_days: tuple[float, ...] = (0.0, 7.0, 14.0, 21.0, 42.0)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
# A storm erodes ``pre`` → ``post``; recovery blends ``post`` back toward ``pre``.
# ``post`` is a lower-berm / dune-stripped version of ``pre`` on the same grid.
# NOTE: seeds are fixed in-spec; APPEND new scenarios (goldens are per-name).

_DUNE = DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0)

_SCENARIOS: list[RecoveryCase] = [
    # Dune lost, full-profile blend (no mask): linear vs exponential contrast.
    RecoveryCase(
        "dune_loss_linear",
        pre_spec=dict(berm_elevation=2.5, berm_width=40.0, upland_elevation=3.0, dune=_DUNE),
        post_spec=dict(berm_elevation=1.0, berm_width=12.0, upland_elevation=3.0, dune=None),
        model="linear",
    ),
    RecoveryCase(
        "dune_loss_exponential",
        pre_spec=dict(berm_elevation=2.5, berm_width=40.0, upland_elevation=3.0, dune=_DUNE),
        post_spec=dict(berm_elevation=1.0, berm_width=12.0, upland_elevation=3.0, dune=None),
        model="exponential",
    ),
    # Below-berm mask: only the eroded lower beach rebuilds by fair-weather
    # recovery; the region at/above z_berm (the lost dune) is frozen at the
    # post-storm bed — regrowing the dune needs nourishment, not this blend.
    RecoveryCase(
        "masked_beach_rebuild",
        pre_spec=dict(berm_elevation=2.5, berm_width=45.0, upland_elevation=3.0, dune=_DUNE),
        post_spec=dict(berm_elevation=1.0, berm_width=15.0, upland_elevation=3.0, dune=None),
        model="linear",
        z_berm=2.0,
    ),
    # Berm-only erosion/recovery (no dune either side); unmasked.
    RecoveryCase(
        "berm_only_linear",
        pre_spec=dict(berm_elevation=2.5, berm_width=50.0, upland_elevation=3.0, dune=None),
        post_spec=dict(berm_elevation=1.2, berm_width=18.0, upland_elevation=3.0, dune=None),
        model="linear",
    ),
    RecoveryCase(
        "berm_only_exponential",
        pre_spec=dict(berm_elevation=2.5, berm_width=50.0, upland_elevation=3.0, dune=None),
        post_spec=dict(berm_elevation=1.2, berm_width=18.0, upland_elevation=3.0, dune=None),
        model="exponential",
    ),
    # Deep erosion: post is stripped near flat; full rebuild toward a tall dune.
    RecoveryCase(
        "deep_erosion_linear",
        pre_spec=dict(
            berm_elevation=3.0,
            berm_width=45.0,
            upland_elevation=3.5,
            dune=DuneSpec("triangular", crest_elevation=6.5, front_width=18.0, back_width=22.0),
        ),
        post_spec=dict(berm_elevation=0.6, berm_width=8.0, upland_elevation=3.5, dune=None),
        model="linear",
    ),
    # Masked deep erosion with an exponential curve — the mask + fast early
    # recovery combination the pipeline actually runs.
    RecoveryCase(
        "masked_deep_exponential",
        pre_spec=dict(
            berm_elevation=3.0,
            berm_width=45.0,
            upland_elevation=3.5,
            dune=DuneSpec("triangular", crest_elevation=6.5, front_width=18.0, back_width=22.0),
        ),
        post_spec=dict(berm_elevation=0.8, berm_width=10.0, upland_elevation=3.5, dune=None),
        model="exponential",
        z_berm=2.5,
    ),
]

CASES: list[RecoveryCase] = list(_SCENARIOS)


# ---------------------------------------------------------------------------
# Run + serialize
# ---------------------------------------------------------------------------


def run_case(case: RecoveryCase):
    """Return ``(x, zb_pre, zb_post, fractions, recovered)`` for a case.

    Fractions come from the real ``_recovery_fraction`` and recovered beds from
    the real ``recovered_bed``, so the gallery tracks the shipping mechanics.
    """
    x, zb_pre, _ = make_profile(**case.pre_spec)
    x_post, zb_post, _ = make_profile(**case.post_spec)
    if not np.array_equal(x, x_post):
        raise ValueError(f"{case.name}: pre/post grids differ (match dx and x_max)")
    fractions = [_recovery_fraction(d, case.T_recover, case.model) for d in case.sample_days]
    recovered = [recovered_bed(zb_post, zb_pre, f, case.z_berm) for f in fractions]
    return x, zb_pre, zb_post, fractions, recovered


def _spec_to_json(spec: dict) -> dict:
    out = dict(spec)
    dune = out.get("dune")
    out["dune"] = dataclasses.asdict(dune) if dune is not None else None
    return out


def _spec_from_json(spec: dict) -> dict:
    out = dict(spec)
    dune = out.get("dune")
    out["dune"] = DuneSpec(**dune) if dune is not None else None
    return out


def case_to_golden(case: RecoveryCase, fractions, recovered, x) -> dict:
    return {
        "name": case.name,
        "pre_spec": _spec_to_json(case.pre_spec),
        "post_spec": _spec_to_json(case.post_spec),
        "model": case.model,
        "T_recover": case.T_recover,
        "z_berm": case.z_berm,
        "sample_days": list(case.sample_days),
        "fractions": [float(f) for f in fractions],
        "volumes": [float(volume_above_datum(x, r)) for r in recovered],
    }


def golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{name}.json")


def save_golden(golden: dict) -> str:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = golden_path(golden["name"])
    with open(path, "w") as fh:
        json.dump(golden, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_golden(name: str) -> dict:
    with open(golden_path(name)) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def save_recovery_png(case: RecoveryCase, out_dir: str = ARTIFACT_DIR) -> str:
    """Render the recovery fan for a case via ``viz.plot_recovery``."""
    from erosion.viz import plot_recovery

    x, zb_pre, zb_post, fractions, _ = run_case(case)
    fig, ax = plt.subplots(figsize=(13, 7))
    plot_recovery(
        ax,
        x,
        zb_pre,
        zb_post,
        fractions,
        z_berm=case.z_berm,
        day_labels=case.sample_days,
        model=case.model,
        title_prefix=f"{case.name} · ",
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{case.name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def regenerate_png(golden: dict, out_dir: str = ARTIFACT_DIR) -> str:
    """Redraw a case from its saved parameters ONLY (no recovery run).

    Beds come from ``pre_spec`` / ``post_spec``; the fan uses the saved
    ``fractions``.  Renders through the same ``viz.plot_recovery`` as the fresh
    path, so the round-trip plot matches — proof the golden captures the recovery.
    """
    from erosion.viz import plot_recovery

    x, zb_pre, _ = make_profile(**_spec_from_json(golden["pre_spec"]))
    _, zb_post, _ = make_profile(**_spec_from_json(golden["post_spec"]))

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_recovery(
        ax,
        x,
        zb_pre,
        zb_post,
        golden["fractions"],
        z_berm=golden["z_berm"],
        day_labels=golden["sample_days"],
        model=golden["model"],
        title_prefix="regen · ",
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{golden['name']}_regen.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def regenerate_all(out_dir: str = ARTIFACT_DIR) -> None:
    """Regenerate every gallery PNG from committed goldens (no recovery run)."""
    for case in CASES:
        regenerate_png(load_golden(case.name), out_dir)


if __name__ == "__main__":
    # Refresh goldens from the current mechanics, then render both galleries.
    for case in CASES:
        x, zb_pre, zb_post, fractions, recovered = run_case(case)
        save_golden(case_to_golden(case, fractions, recovered, x))
        save_recovery_png(case)
        regenerate_png(load_golden(case.name))
    print(f"wrote goldens → {GOLDEN_DIR}")
