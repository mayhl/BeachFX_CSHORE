"""Mass balance: the accounting stream (placements.csv) against the bed mutation.

Nothing else in the suite ties ``placed_m3`` to what the placement actually did
to the profile, so the two could diverge indefinitely with everything green.
The design relation: the parametric restore adds SUBAERIAL sand only, while the
accounting extends the wedge down to the depth of closure (cReach.cpp:1649) --
so bed volume == placed x be/(be+dclose), collapsing to equality at dclose=0.
"""

import tempfile

import pytest

from erosion.config import ReachConfig
from erosion.metrics import volume_above_datum
from erosion.results import ParquetResultsSink
from tests.builders import ncfg, run, storms_at, template_profile
from tests.doubles import SEVERE, ScriptedRunner

_BE = 2.0  # template_profile's berm elevation (msl 0, so also the wedge height)


def _campaign(dclose: float):
    """One SEVERE storm, one full restore campaign; return (placed_m3, dV_bed)."""
    p = template_profile()
    cfg = ReachConfig(
        storm={"z_berm": _BE},
        nourishment=ncfg(volume_trigger=30.0, production_rate=100.0, assessor="volume"),
        depth_of_closure={"value": dclose, "units": "m"},
    )
    sink = ParquetResultsSink(tempfile.mkdtemp(), "test", "FWOP", lifecycle=0)
    profiles, sink = run(
        [p],
        storms_df=storms_at([20]),
        sim_end=200.0,
        cfg=cfg,
        runner=ScriptedRunner(SEVERE),
        sink=sink,
    )
    pr = profiles[0]
    (row,) = sink._nourishment_rows
    snaps = {s.label.value: s for s in pr.snapshots}
    dv = volume_above_datum(pr.x, snaps["EEN"].zb) - volume_above_datum(pr.x, snaps["SEN"].zb)
    return row["placed_m3"], dv


class TestPlacementMassBalance:
    def test_no_closure_extension_is_exact(self):
        placed, dv = _campaign(dclose=0.0)
        assert placed > 0.0
        assert dv == pytest.approx(placed, rel=1e-6)

    def test_closure_extension_bills_the_full_wedge(self):
        dclose = 6.0
        placed, dv = _campaign(dclose=dclose)
        # subaerial share of the billed wedge lands on the bed; the rest is the
        # subaqueous extension the template cannot place
        assert dv == pytest.approx(placed * _BE / (_BE + dclose), rel=1e-6)
        assert dv < placed
