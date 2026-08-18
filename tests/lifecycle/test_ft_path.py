"""The ft input path, end to end: one physical scenario authored twice.

Every other lifecycle test opts out of unit conversion with an
``input_units="m"`` context, so the ft default -- production's default
(``pipeline.py``) -- ran uncovered.  Here the SAME storm/nourishment scenario is
authored in metres (metric context) and in feet (no context, the ft default);
the two configs must produce byte-identical physics and accounting.
"""

import tempfile

import numpy as np
import pytest

from erosion.config import ReachConfig
from erosion.results import ParquetResultsSink
from tests.builders import run, storms_at, template_profile
from tests.doubles import SEVERE, ScriptedRunner

_M_TO_FT = 1.0 / 0.3048
_M3_TO_CY = 1.0 / (27.0 * 0.3048**3)


def _run(cfg):
    sink = ParquetResultsSink(tempfile.mkdtemp(), "test", "FWOP", lifecycle=0)
    profiles, sink = run(
        [template_profile()],
        storms_df=storms_at([20]),
        sim_end=200.0,
        cfg=cfg,
        runner=ScriptedRunner(SEVERE),
        sink=sink,
    )
    return profiles[0], sink._nourishment_rows


class TestFtAuthoredTwin:
    def test_ft_and_metric_authoring_agree_exactly(self):
        dclose_m, msl_m, trigger_m3, rate_m3day = 6.0, 0.2, 30.0, 100.0
        metric = ReachConfig.model_validate(
            {
                "storm": {"z_berm": 2.0},
                "depth_of_closure": dclose_m,
                "msl": msl_m,
                "nourishment": {
                    "volume_trigger": trigger_m3,
                    # NOTE bare metric rates are per YEAR (the m3/yr equivalent of
                    # the cy/yr ft default) -- state the unit to mean per day
                    "production_rate": {"value": rate_m3day, "units": "m3/day"},
                    "assessor": "volume",
                },
            },
            context={"input_units": "m"},
        )
        ft = ReachConfig.model_validate(
            {
                "storm": {"z_berm": 2.0 * _M_TO_FT},
                "depth_of_closure": dclose_m * _M_TO_FT,
                "msl": msl_m * _M_TO_FT,
                "nourishment": {
                    "volume_trigger": trigger_m3 * _M3_TO_CY,
                    # cy/yr is the ft-default rate unit
                    "production_rate": rate_m3day * _M3_TO_CY * 365.25,
                    "assessor": "volume",
                },
            }
            # no context: the ft default, as production runs it
        )
        # the configs resolve to the same internal SI values...
        assert ft.depth_of_closure == pytest.approx(metric.depth_of_closure, rel=1e-12)
        assert ft.msl == pytest.approx(metric.msl, rel=1e-12)
        nm, nf = metric.nourishment, ft.nourishment
        assert nf.volume_trigger == pytest.approx(nm.volume_trigger, rel=1e-12)
        assert nf.production_rate == pytest.approx(nm.production_rate, rel=1e-12)

        # ...and the lifecycles they drive are indistinguishable
        p_m, rows_m = _run(metric)
        p_f, rows_f = _run(ft)
        np.testing.assert_allclose(p_f.zb, p_m.zb, atol=1e-9)
        assert [s.label for s in p_f.snapshots] == [s.label for s in p_m.snapshots]
        assert len(rows_f) == len(rows_m)
        for rm, rf in zip(rows_m, rows_f):
            assert rf["placed_m3"] == pytest.approx(rm["placed_m3"], rel=1e-9)
            assert rf["t_end"] == pytest.approx(rm["t_end"], rel=1e-9)
