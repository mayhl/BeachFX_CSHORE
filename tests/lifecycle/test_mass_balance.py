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


class TestPartialFillMassBalance:
    """An interrupted campaign, segment by segment.

    The resume-to-completion segment balances exactly.  The storm-cut partial
    segment does NOT: billing is crew-rate x working time while the bed blend is
    fraction x the parametric template's gap, and the template gap need not equal
    the planner's assessed volume.  Measured on this scenario: billed 53.996 m3,
    bed received 51.392 m3 (~5% overbill).  Documents current behavior -- the
    reconciliation (bill what the bed got, or blend what was billed) is an open
    domain decision; these pins keep the divergence VISIBLE and bounded.
    """

    def _interrupted(self):
        from tests.doubles import MASSIVE, NONE

        p = template_profile()
        cfg = ReachConfig(
            storm={"z_berm": _BE},
            nourishment=ncfg(volume_trigger=10.0, production_rate=4.0, assessor="volume"),
        )
        sink = ParquetResultsSink(tempfile.mkdtemp(), "test", "FWOP", lifecycle=0)
        profiles, sink = run(
            [p],
            storms_df=storms_at([20, 34]),
            sim_end=100.0,
            cfg=cfg,
            runner=ScriptedRunner({"p0": [MASSIVE, NONE]}),
            sink=sink,
        )
        pr = profiles[0]
        va = {i: volume_above_datum(pr.x, s.zb) for i, s in enumerate(pr.snapshots)}
        labels = [s.label.value for s in pr.snapshots]
        partial, full = sink._nourishment_rows
        dv_partial = va[labels.index("EENS")] - va[labels.index("SEN")]
        i_resume = len(labels) - 1 - labels[::-1].index("SEN")
        dv_full = va[labels.index("EEN")] - va[i_resume]
        return partial, full, dv_partial, dv_full

    def test_resume_segment_is_exact(self):
        _, full, _, dv_full = self._interrupted()
        assert dv_full == pytest.approx(full["placed_m3"], rel=1e-6)

    def test_partial_segment_bills_the_crew_time(self):
        partial, _, _, _ = self._interrupted()
        worked = partial["t_end"] - partial["t_start"]
        assert partial["placed_m3"] == pytest.approx(4.0 * worked, rel=1e-3)

    def test_partial_segment_never_credits_more_than_billed(self):
        # the divergence stays one-sided (accounting >= beach) and bounded
        partial, _, dv_partial, _ = self._interrupted()
        assert 0.0 < dv_partial <= partial["placed_m3"]
        assert dv_partial == pytest.approx(partial["placed_m3"], rel=0.10)
