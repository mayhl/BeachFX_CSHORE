"""Run the profile fitter on the real input profiles in ``data/profiles/`` — not
just synthetic ones — so the fit is exercised against actual CSHORE-convention
field data (full cross-shore, feet→meters, deep offshore).

Each profile is fit, checked for basic validity, and regression-locked against a
committed golden (``tests/goldens/real/*.json``).  Regenerate after an intended
fitter change:

    REGEN_FIT_GOLDENS=1 uv run pytest tests/erosion/test_fit_real.py

Render the fits (zoomed to the subaerial beach) with ``--plot``.
"""

import glob
import json
import math
import os

import numpy as np
import pytest

from erosion.metrics import MorphType, fit_profile
from erosion.profile import load_raw_profile
from tests.fit_gallery import _metrics_to_json

_HERE = os.path.dirname(__file__)
_ROOT = os.path.dirname(os.path.dirname(_HERE))
PROFILE_DIR = os.path.join(_ROOT, "data", "profiles")
GOLDEN_DIR = os.path.join(_HERE, "..", "goldens", "real")
ARTIFACT_DIR = os.path.join(_HERE, "..", "_artifacts", "real")

DESIGN_BE, DATUM, D50 = 1.8, 0.0, 0.3  # design berm (m) used only for classification

_ABS, _REL = 1e-6, 1e-6


def _profiles() -> list[str]:
    return sorted(glob.glob(os.path.join(PROFILE_DIR, "*.csv")))


def _close(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= _ABS + _REL * abs(b)
    return a == b


@pytest.mark.parametrize("path", _profiles(), ids=lambda p: os.path.basename(p))
def test_fit_real_profile(path, plot):
    raw = load_raw_profile(path, D50)
    x, z = raw["x"], raw["z"]
    m, ideal = fit_profile(x, z, DESIGN_BE, DATUM)

    # --- basic validity (no ground truth for field data) ---
    assert m.morph_type in {t.value for t in MorphType}
    if not np.isnan(m.dune_crest_elevation) and not np.isnan(m.berm_elevation):
        assert m.dune_crest_elevation >= m.berm_elevation

    # --- golden regression on the metrics ---
    name = os.path.splitext(os.path.basename(path))[0]
    gpath = os.path.join(GOLDEN_DIR, f"{name}.json")
    fresh = _metrics_to_json(m)
    if os.environ.get("REGEN_FIT_GOLDENS"):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(gpath, "w") as fh:
            json.dump(fresh, fh, indent=2, sort_keys=True)
            fh.write("\n")
        pytest.skip(f"regenerated real golden for {name}")

    with open(gpath) as fh:
        golden = json.load(fh)
    for field, want in golden.items():
        assert _close(fresh[field], want), f"{name}.{field}: {fresh[field]!r} != {want!r}"

    if plot:
        _save_real_png(name, x, z)


def _save_real_png(name: str, x, z, out_dir: str = ARTIFACT_DIR) -> str:
    """Render the fit zoomed to the subaerial beach (the profile spans ~1.6 km)."""
    import matplotlib.pyplot as plt

    from erosion.viz import plot_idealized_fit

    fig = plot_idealized_fit(x, z, DESIGN_BE, DATUM)
    ax = fig.axes[0]
    dry = np.where(z > DATUM - 1.0)[0]  # zoom to the beach: ~1 m below datum to the top
    if len(dry):
        ax.set_xlim(max(0.0, x[dry[0]] - 30), x[-1] + 10)
        ax.set_ylim(-2.0, float(z.max()) + 0.5)
    ax.set_title(f"{name}   " + ax.get_title(), fontsize=17)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
