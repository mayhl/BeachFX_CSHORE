"""Offline alternatives generator — cartesian / zip expansion."""

import pytest

from erosion.sweep import _expand_range, _set_dotted, expand_alternatives


class TestSetDotted:
    def test_creates_nested(self):
        d: dict = {}
        _set_dotted(d, "a.b.c", 5)
        assert d == {"a": {"b": {"c": 5}}}

    def test_single_key(self):
        d: dict = {}
        _set_dotted(d, "x", 1)
        assert d == {"x": 1}

    def test_collision_raises(self):
        with pytest.raises(ValueError, match="collides"):
            _set_dotted({"a": 3}, "a.b", 1)


class TestExpandAlternatives:
    AXES = {
        "nourishment.borrow_to_placement_ratio": [1.0, 1.5],
        "nourishment.storm_conflict": ["interrupt", "defer"],
    }

    def test_cartesian_count(self):
        out = expand_alternatives({}, self.AXES, mode="cartesian")
        assert len(out) == 4  # 2 x 2

    def test_zip_count(self):
        out = expand_alternatives({}, self.AXES, mode="zip")
        assert len(out) == 2  # paired

    def test_zip_unequal_lengths_raises(self):
        axes = {"a.x": [1, 2, 3], "a.y": [1, 2]}
        with pytest.raises(ValueError, match="equal-length"):
            expand_alternatives({}, axes, mode="zip")

    def test_ids_numbered_from_start(self):
        out = expand_alternatives({}, self.AXES, start_id=11)
        assert sorted(out, key=int) == ["11", "12", "13", "14"]

    def test_dotted_paths_applied(self):
        out = expand_alternatives({}, self.AXES, mode="zip")
        first = out["1"]
        assert first["nourishment"]["borrow_to_placement_ratio"] == 1.0
        assert first["nourishment"]["storm_conflict"] == "interrupt"

    def test_name_summarizes_point(self):
        out = expand_alternatives({}, self.AXES, mode="zip")
        assert out["1"]["name"] == "borrow_to_placement_ratio=1.0, storm_conflict=interrupt"

    def test_base_not_mutated_and_merged(self):
        base = {"storm": {"T_recover": 21.0}}
        out = expand_alternatives(base, self.AXES, mode="zip")
        assert base == {"storm": {"T_recover": 21.0}}  # untouched
        assert out["1"]["storm"]["T_recover"] == 21.0  # carried into each point

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="cartesian.*zip"):
            expand_alternatives({}, self.AXES, mode="grid")

    def test_no_axes_raises(self):
        with pytest.raises(ValueError, match="no axes"):
            expand_alternatives({}, {}, mode="cartesian")


class TestExpandRange:
    def test_list_passthrough(self):
        assert _expand_range([0, 1, 2]) == [0, 1, 2]

    def test_range_string(self):
        assert _expand_range("0-3") == [0, 1, 2, 3]

    def test_mixed(self):
        assert _expand_range("0-2,7,10-11") == [0, 1, 2, 7, 10, 11]

    def test_descending_raises(self):
        with pytest.raises(ValueError, match="descending"):
            _expand_range("3-1")


class TestLifecycleAxis:
    AXES = {
        "nourishment.borrow_to_placement_ratio": [1.0, 1.5],
        "lifecycle": "0-2",  # range string
    }

    def test_crosses_params_by_lifecycle(self):
        out = expand_alternatives({}, self.AXES, mode="cartesian")
        assert len(out) == 6  # 2 params x 3 lifecycles

    def test_lifecycle_is_top_level_not_nested(self):
        out = expand_alternatives({}, self.AXES, mode="cartesian")
        run = out["1"]
        assert run["lifecycle"] == 0  # top-level int
        assert "lifecycle" not in run.get("nourishment", {})

    def test_lifecycle_list_form(self):
        axes = {"nourishment.storm_conflict": ["interrupt"], "lifecycle": [5, 6]}
        out = expand_alternatives({}, axes, mode="cartesian")
        assert sorted(r["lifecycle"] for r in out.values()) == [5, 6]

    def test_name_includes_lifecycle(self):
        out = expand_alternatives({}, self.AXES, mode="cartesian")
        assert "lifecycle=0" in out["1"]["name"]
