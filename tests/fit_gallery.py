"""Fit-gallery support: run synthetic fit cases, serialize their parameters, and
regenerate the diagnostic plot from those saved parameters alone.

A *golden* is a small JSON per case holding everything needed to reproduce both
the raw profile and its fit without re-running ``fit_profile``:

    - ``spec``      : the ``make_profile`` kwargs (deterministic w/ ``seed``)
    - ``design_BE`` / ``datum``
    - ``metrics``   : every ``ProfileMetrics`` scalar field
    - ``knots_x`` / ``knots_z`` : the ``IdealizedProfile`` landmark knots (or null)

``regenerate_png`` redraws raw (from ``spec``) + idealized (from the saved knots)
with no fitting, which is the point: the JSON *is* the fit.  ``test_fit_golden``
compares a fresh fit against the committed golden and, under ``--plot``,
renders the PNG via ``erosion.viz.plot_idealized_fit``.
"""

from __future__ import annotations

import dataclasses
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from erosion.metrics import IdealizedProfile, ProfileMetrics, fit_profile
from tests.synthetic import DuneSpec, make_profile

ROOT = os.path.dirname(__file__)
GOLDEN_DIR = os.path.join(ROOT, "goldens", "fits")
ARTIFACT_DIR = os.path.join(ROOT, "_artifacts", "fits")


@dataclasses.dataclass
class FitCase:
    """One synthetic profile + the design berm elevation used to fit it."""

    name: str
    spec: dict  # make_profile kwargs; ``dune`` given as a DuneSpec
    design_BE: float
    datum: float = 0.0


# Base scenarios (clean); each is emitted below in a clean and a noisy variant.
# (name, make_profile kwargs [dune as a DuneSpec], design_BE)
_SCENARIOS: list[tuple[str, dict, float]] = [
    (
        "triangular_lowupland",
        dict(
            berm_elevation=2.0,
            berm_width=30.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
        ),
        2.0,
    ),
    (
        "trapezoidal_lowupland",
        dict(
            berm_elevation=2.0,
            berm_width=30.0,
            upland_elevation=1.0,
            dune=DuneSpec(
                "trapezoidal",
                crest_elevation=5.0,
                front_width=15.0,
                back_width=20.0,
                top_width=12.0,
            ),
        ),
        2.0,
    ),
    (
        "gaussian_lowupland",
        dict(
            berm_elevation=2.0,
            berm_width=30.0,
            upland_elevation=1.0,
            dune=DuneSpec("gaussian", crest_elevation=5.0, sigma=7.0),
        ),
        2.0,
    ),
    (
        "lowberm",
        dict(
            berm_elevation=2.0,
            upland_elevation=2.5,
            dune=DuneSpec("triangular", crest_elevation=5.0),
        ),
        2.0,
    ),
    # Trapezoidal (flat-topped) dune across the three morphology types
    # (trapezoidal_lowupland above completes the trio).
    (
        "trapezoidal_lowberm",
        dict(
            berm_elevation=2.0,
            upland_elevation=2.5,
            dune=DuneSpec(
                "trapezoidal",
                crest_elevation=6.0,
                front_width=15.0,
                back_width=20.0,
                top_width=12.0,
            ),
        ),
        2.0,
    ),
    (
        "trapezoidal_highupland",
        dict(
            berm_elevation=2.0,
            upland_elevation=4.0,
            dune=DuneSpec(
                "trapezoidal",
                crest_elevation=3.0,
                front_width=15.0,
                back_width=20.0,
                top_width=12.0,
            ),
        ),
        2.0,
    ),
    (
        "scarped",
        dict(
            berm_elevation=2.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=3.0, back_width=20.0),
        ),
        2.0,
    ),
    # Berm scarp (a vertical cut into the berm), LOW_UPLAND; the dune scarp above
    # is `scarped` (steep dune front).  Other-morphology berm scarps are appended
    # at the end (see note there).
    (
        "berm_scarp_lowupland",
        dict(
            berm_elevation=3.0,
            berm_width=45.0,
            upland_elevation=2.0,
            dune=DuneSpec("triangular", crest_elevation=6.0, front_width=15.0, back_width=20.0),
            scarp=(58.0, 1.0),  # 1 m vertical cut into the berm
        ),
        3.0,
    ),
    (
        "no_berm",
        dict(
            berm_elevation=2.0,
            berm_width=0.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0),
        ),
        2.0,
    ),
    (
        "wide_berm",
        dict(
            berm_elevation=2.0,
            berm_width=60.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=15.0, back_width=20.0),
        ),
        2.0,
    ),
    (
        "tall_dune",
        dict(
            berm_elevation=2.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=8.0, front_width=20.0, back_width=25.0),
        ),
        2.0,
    ),
    # HIGH_UPLAND (no dune) at three back slopes — probes berm/upland boundary.
    ("highupland_steep", dict(berm_elevation=2.0, upland_elevation=4.0, dune=None), 2.0),
    ("highupland_gentle", dict(berm_elevation=2.0, upland_elevation=2.5, dune=None), 2.0),
    # NOTE: the noisy seed is the scenario index (below), so APPEND new scenarios
    # here — inserting mid-list reshuffles later cases' seeds and their goldens.
    # Berm scarp in LOW_BERM / HIGH_UPLAND (the LOW_UPLAND one is above).
    (
        "berm_scarp_lowberm",
        dict(
            berm_elevation=3.0,
            upland_elevation=4.0,
            dune=DuneSpec("triangular", crest_elevation=7.0, front_width=15.0, back_width=20.0),
            scarp=(58.0, 0.5),
        ),
        3.0,
    ),
    (
        "berm_scarp_highupland",
        dict(
            berm_elevation=3.0,
            berm_width=45.0,
            upland_elevation=5.5,
            dune=None,
            scarp=(58.0, 1.0),
        ),
        3.0,
    ),
    # Dune scarp (steep front) in LOW_BERM, and combined dune+berm scarps —
    # rounding out ~3 scarp cases per dune-bearing morphology.
    (
        "dune_scarp_lowberm",
        dict(
            berm_elevation=2.0,
            upland_elevation=2.5,
            dune=DuneSpec("triangular", crest_elevation=5.0, front_width=3.0, back_width=20.0),
        ),
        2.0,
    ),
    (
        "duneberm_scarp_lowupland",
        dict(
            berm_elevation=3.0,
            berm_width=45.0,
            upland_elevation=1.0,
            dune=DuneSpec("triangular", crest_elevation=6.0, front_width=3.0, back_width=20.0),
            scarp=(58.0, 1.0),
        ),
        3.0,
    ),
    (
        "duneberm_scarp_lowberm",
        dict(
            berm_elevation=3.0,
            upland_elevation=4.0,
            dune=DuneSpec("triangular", crest_elevation=7.0, front_width=3.0, back_width=20.0),
            scarp=(58.0, 0.5),
        ),
        3.0,
    ),
]

CASES: list[FitCase] = []
for _i, (_name, _spec, _be) in enumerate(_SCENARIOS):
    CASES.append(FitCase(_name, _spec, _be))
    CASES.append(FitCase(f"{_name}_noisy", {**_spec, "noise": 0.05, "seed": _i + 1}, _be))


# ---------------------------------------------------------------------------
# Run + serialize
# ---------------------------------------------------------------------------


def run_case(case: FitCase):
    """Return ``(x, zb, metrics, ideal, truth)`` for a case."""
    x, zb, truth = make_profile(**case.spec)
    m, ideal = fit_profile(x, zb, case.design_BE, case.datum)
    return x, zb, m, ideal, truth


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


def _metrics_to_json(m: ProfileMetrics) -> dict:
    out: dict = {}
    for f in dataclasses.fields(m):
        v = getattr(m, f.name)
        if isinstance(v, (bool, np.bool_)):
            out[f.name] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[f.name] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[f.name] = float(v)
        else:
            out[f.name] = v
    return out


def case_to_golden(case: FitCase, m: ProfileMetrics, ideal: IdealizedProfile | None) -> dict:
    return {
        "name": case.name,
        "spec": _spec_to_json(case.spec),
        "design_BE": case.design_BE,
        "datum": case.datum,
        "metrics": _metrics_to_json(m),
        "knots_x": ideal.knots_x.tolist() if ideal is not None else None,
        "knots_z": ideal.knots_z.tolist() if ideal is not None else None,
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


def save_fit_png(case: FitCase, out_dir: str = ARTIFACT_DIR) -> str:
    """Fit the case fresh and render the full diagnostic overlay.

    For a scarp case, also overlays the intact **pre-scarp** profile (the same
    spec without the ``scarp`` cut) so the scarp shows as the gap to the raw."""
    from erosion.viz import plot_idealized_fit

    x, zb, _ = make_profile(**case.spec)
    ref_zb = None
    if case.spec.get("scarp") is not None:
        ref_zb = make_profile(**{k: v for k, v in case.spec.items() if k != "scarp"})[1]
    fig = plot_idealized_fit(x, zb, case.design_BE, case.datum, ref_zb=ref_zb)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{case.name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def regenerate_png(golden: dict, out_dir: str = ARTIFACT_DIR) -> str:
    """Redraw a case from its saved parameters ONLY (no fitting).

    Raw comes from ``spec``; the idealized curve is the saved ``knots``.  Renders
    through the same ``viz.plot_fit_overlay`` as the fresh-fit path, so the
    round-trip plot matches the original — proof the golden captures the fit."""
    from types import SimpleNamespace

    from erosion.viz import plot_fit_overlay

    x, zb, _ = make_profile(**_spec_from_json(golden["spec"]))
    m = SimpleNamespace(**golden["metrics"])
    ideal = (
        IdealizedProfile(np.asarray(golden["knots_x"]), np.asarray(golden["knots_z"]))
        if golden["knots_x"] is not None
        else None
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_fit_overlay(ax, x, zb, ideal, m, golden["datum"], title_prefix="regen · ")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{golden['name']}_regen.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def regenerate_all(out_dir: str = ARTIFACT_DIR) -> None:
    """Regenerate every gallery PNG from committed goldens (no fitting)."""
    for case in CASES:
        regenerate_png(load_golden(case.name), out_dir)


if __name__ == "__main__":
    # Refresh goldens from the current fitter, then render both galleries.
    for case in CASES:
        x, zb, m, ideal, _ = run_case(case)
        save_golden(case_to_golden(case, m, ideal))
        save_fit_png(case)
        regenerate_png(load_golden(case.name))
    print(f"wrote goldens → {GOLDEN_DIR}")
    print(f"wrote gallery → {ARTIFACT_DIR}")
