"""Unit tests for the post-hoc derived-metric helpers in ``erosion.summary``."""

import numpy as np
import pandas as pd
import pytest

from erosion.summary import profile_extremes, storm_deltas


def _row(profile_id, label, t, **kw):
    base = dict(
        profile_id=profile_id,
        label=label,
        t=t,
        dune_crest_elevation=np.nan,
        berm_elevation=np.nan,
        berm_width=0.0,
        dune_width=0.0,
        dune_front_relief=np.nan,
        dune_back_relief=np.nan,
        volume_above_datum=0.0,
        dune_scarp=False,
        scarp_height=np.nan,
    )
    base.update(kw)
    return base


@pytest.fixture
def metrics():
    # One profile over a storm: INIT → PreStorm → PostStorm → EndIteration.
    return pd.DataFrame(
        [
            _row("P1", "INIT", 0.0, dune_crest_elevation=6.0, berm_elevation=2.0,
                 berm_width=40.0, volume_above_datum=500.0, dune_front_relief=4.0),
            _row("P1", "PreStorm", 10.0, dune_crest_elevation=6.0, berm_elevation=2.0,
                 berm_width=40.0, volume_above_datum=500.0, dune_front_relief=4.0),
            _row("P1", "PostStorm", 10.0, dune_crest_elevation=5.0, berm_elevation=1.5,
                 berm_width=25.0, volume_above_datum=430.0, dune_front_relief=3.5,
                 dune_scarp=True, scarp_height=0.8),
            _row("P1", "EndIteration", 20.0, dune_crest_elevation=5.2, berm_elevation=1.7,
                 berm_width=30.0, volume_above_datum=455.0, dune_front_relief=3.5),
        ]
    )


def test_profile_extremes_envelope_and_net(metrics):
    ex = profile_extremes(metrics).set_index("profile_id").loc["P1"]
    assert ex["dune_crest_elevation_max"] == 6.0
    assert ex["dune_crest_elevation_min"] == 5.0
    assert ex["dune_crest_elevation_range"] == pytest.approx(1.0)
    assert ex["berm_width_max"] == 40.0 and ex["berm_width_min"] == 25.0
    # Net = last (EndIteration, t=20) minus first (INIT, t=0), ordered by t.
    assert ex["dune_crest_elevation_net"] == pytest.approx(5.2 - 6.0)
    assert ex["volume_above_datum_net"] == pytest.approx(455.0 - 500.0)
    assert ex["scarp_height_max"] == pytest.approx(0.8)


def test_profile_extremes_skips_nan_dune():
    # A profile that is HIGH_UPLAND (no dune) at one snapshot still reports its
    # dune envelope from the snapshots where a dune exists.
    df = pd.DataFrame(
        [
            _row("P2", "INIT", 0.0, dune_crest_elevation=np.nan),  # no dune
            _row("P2", "PostStorm", 5.0, dune_crest_elevation=4.0),
        ]
    )
    ex = profile_extremes(df).set_index("profile_id").loc["P2"]
    assert ex["dune_crest_elevation_max"] == 4.0
    assert ex["dune_crest_elevation_min"] == 4.0


def test_storm_deltas_drop_convention(metrics):
    d = storm_deltas(metrics)
    assert len(d) == 1
    row = d.iloc[0]
    assert row["profile_id"] == "P1" and row["t"] == 10.0
    assert row["crest_drop"] == pytest.approx(1.0)  # 6.0 - 5.0
    assert row["berm_elev_drop"] == pytest.approx(0.5)  # 2.0 - 1.5
    assert row["berm_width_loss"] == pytest.approx(15.0)  # 40 - 25
    assert row["eroded_volume"] == pytest.approx(70.0)  # 500 - 430, positive = erosion
    assert bool(row["dune_scarp"]) is True
    assert row["scarp_height"] == pytest.approx(0.8)
    assert bool(row["inundation"]) is False


def test_storm_deltas_inundation_is_unchanged():
    # CSHORE failure: INUNDATION carries the pre-storm profile through unchanged,
    # so every delta is zero and the row is flagged.
    df = pd.DataFrame(
        [
            _row("P3", "PreStorm", 3.0, dune_crest_elevation=5.0, volume_above_datum=400.0),
            _row("P3", "INUNDATION", 3.0, dune_crest_elevation=5.0, volume_above_datum=400.0),
        ]
    )
    d = storm_deltas(df)
    assert len(d) == 1
    assert d.iloc[0]["eroded_volume"] == pytest.approx(0.0)
    assert bool(d.iloc[0]["inundation"]) is True


def test_empty_inputs_return_empty():
    empty = pd.DataFrame()
    assert profile_extremes(empty).empty
    assert storm_deltas(pd.DataFrame({"label": [], "profile_id": [], "t": []})).empty
