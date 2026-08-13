"""Pipeline orchestration helpers — the alternative run-selector."""

import pytest

from erosion.config import _expand_plans, _merge_sections, _parse_run_spec


class TestParseRunSpec:
    AVAIL = ["1", "2", "3", "4", "8", "11", "12", "13", "14", "55", "FWOP", "FWP"]

    def test_all_returns_everything(self):
        assert _parse_run_spec("all", self.AVAIL) == self.AVAIL

    def test_all_case_insensitive_and_whitespace(self):
        assert _parse_run_spec("  ALL ", self.AVAIL) == self.AVAIL

    def test_none_returns_everything(self):
        assert _parse_run_spec(None, self.AVAIL) == self.AVAIL

    def test_single_id(self):
        assert _parse_run_spec("8", self.AVAIL) == ["8"]

    def test_range(self):
        assert _parse_run_spec("1-4", self.AVAIL) == ["1", "2", "3", "4"]

    def test_mixed_ranges_and_singletons(self):
        # the headline example
        assert _parse_run_spec("1-4,8,11-14,55", self.AVAIL) == [
            "1",
            "2",
            "3",
            "4",
            "8",
            "11",
            "12",
            "13",
            "14",
            "55",
        ]

    def test_explicit_string_ids(self):
        assert _parse_run_spec("FWOP,FWP", self.AVAIL) == ["FWOP", "FWP"]

    def test_mixed_int_and_string(self):
        assert _parse_run_spec("1-2,FWP", self.AVAIL) == ["1", "2", "FWP"]

    def test_order_follows_available_not_spec(self):
        # deterministic output order regardless of how the spec is written
        assert _parse_run_spec("8,1", self.AVAIL) == ["1", "8"]

    def test_list_spec(self):
        assert _parse_run_spec([1, 2, "FWP"], self.AVAIL) == ["1", "2", "FWP"]

    def test_whitespace_and_empty_tokens_tolerated(self):
        assert _parse_run_spec(" 1 , , 2 ", self.AVAIL) == ["1", "2"]

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="unknown alternative id"):
            _parse_run_spec("1-3,99", self.AVAIL)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown alternative id"):
            _parse_run_spec("FWXX", self.AVAIL)

    def test_descending_range_raises(self):
        with pytest.raises(ValueError, match="descending range"):
            _parse_run_spec("4-1", self.AVAIL)


class TestMergeSections:
    """Global -> reach -> alt layering of storm/cshore/erosion/slc (schema-v2 Phase 1)."""

    def test_single_level_passes_through(self):
        assert _merge_sections({"storm": {"T_recover": 21.0}}) == {"storm": {"T_recover": 21.0}}

    def test_later_level_wins_per_key(self):
        out = _merge_sections(
            {"storm": {"T_recover": 21.0, "recovery_model": "linear"}},  # global
            {"storm": {"T_recover": 30.0}},  # reach overrides one key
        )
        assert out["storm"] == {"T_recover": 30.0, "recovery_model": "linear"}

    def test_alt_beats_reach_beats_global(self):
        out = _merge_sections(
            {"cshore": {"d50": 0.1, "dx": 1.0}},
            {"cshore": {"d50": 0.2}},
            {"cshore": {"d50": 0.3}},
        )
        assert out["cshore"] == {"d50": 0.3, "dx": 1.0}

    def test_scalar_section_replaces(self):
        assert _merge_sections({"msl": 0.0}, {"msl": 1.5})["msl"] == 1.5

    def test_alt_only_config_is_unchanged(self):
        # Existing configs author storm/erosion/slc only in the alt -> global/reach empty.
        out = _merge_sections({}, {}, {"storm": {"T_recover": 21.0}, "slc": {"rate": 1e-5}})
        assert out == {"storm": {"T_recover": 21.0}, "slc": {"rate": 1e-5}}

    def test_non_section_keys_ignored(self):
        out = _merge_sections({"profiles": ["a.csv"], "alternatives": {"A": {}}, "storm": {"x": 1}})
        assert out == {"storm": {"x": 1}}

    def test_nourishment_not_layered(self):
        # nourishment is reserved for the Phase-2 plan table, not merged here.
        assert "nourishment" not in _merge_sections({"nourishment": {"volume_trigger": 1}})


class TestExpandPlans:
    """schema-v2 nourishment-plan lowering into the alternatives table."""

    def test_v1_config_untouched(self):
        cfg = {"alternatives": {"FWOP": {}}, "reaches": {"R": {"profiles": ["p.csv"]}}}
        out = _expand_plans(cfg)
        assert "alternatives" not in out["reaches"]["R"]  # no plan rewrite happened

    def test_list_form_references_global_table(self):
        cfg = {
            "nourishment": {"fwop": {"volume_trigger": 1}, "fwp": {"volume_trigger": 2}},
            "reaches": {"R": {"nourishment": ["fwop", "fwp"]}},
        }
        out = _expand_plans(cfg)
        assert out["reaches"]["R"]["alternatives"] == {
            "fwop": {"nourishment": {"volume_trigger": 1}},
            "fwp": {"nourishment": {"volume_trigger": 2}},
        }
        assert "nourishment" not in out  # global table consumed
        assert "nourishment" not in out["reaches"]["R"]  # reach spec consumed

    def test_dict_form_inline_overrides_global(self):
        cfg = {
            "nourishment": {"fwp": {"volume_trigger": 1, "production_rate": 5}},
            "reaches": {"R": {"nourishment": {"fwp": {"volume_trigger": 9}}}},
        }
        out = _expand_plans(cfg)
        # inline volume_trigger overrides global; production_rate inherited
        assert out["reaches"]["R"]["alternatives"]["fwp"]["nourishment"] == {
            "volume_trigger": 9,
            "production_rate": 5,
        }

    def test_dict_form_empty_uses_global(self):
        cfg = {
            "nourishment": {"fwop": {"volume_trigger": 1}},
            "reaches": {"R": {"nourishment": {"fwop": {}}}},
        }
        out = _expand_plans(cfg)
        assert out["reaches"]["R"]["alternatives"]["fwop"]["nourishment"] == {"volume_trigger": 1}

    def test_dangling_key_in_list_raises(self):
        cfg = {"nourishment": {"fwop": {}}, "reaches": {"R": {"nourishment": ["nope"]}}}
        with pytest.raises(ValueError, match="not in the global"):
            _expand_plans(cfg)

    def test_dangling_empty_dict_key_raises(self):
        cfg = {"nourishment": {}, "reaches": {"R": {"nourishment": {"ghost": {}}}}}
        with pytest.raises(ValueError, match="empty and not in the global"):
            _expand_plans(cfg)

    def test_coverage_mismatch_raises(self):
        cfg = {
            "nourishment": {"fwop": {"v": 1}, "fwp": {"v": 2}},
            "reaches": {
                "R1": {"nourishment": ["fwop", "fwp"]},
                "R2": {"nourishment": ["fwop"]},  # missing fwp
            },
        }
        with pytest.raises(ValueError, match="missing plan id"):
            _expand_plans(cfg)

    def test_mutual_exclusion_with_alternatives_raises(self):
        cfg = {
            "alternatives": {"A": {}},
            "nourishment": {"fwop": {"v": 1}},
            "reaches": {"R": {"nourishment": ["fwop"]}},
        }
        with pytest.raises(ValueError, match="cannot be combined with an `alternatives`"):
            _expand_plans(cfg)

    def test_reach_without_plans_raises(self):
        cfg = {
            "nourishment": {"fwop": {"v": 1}},
            "reaches": {"R1": {"nourishment": ["fwop"]}, "R2": {"profiles": ["p.csv"]}},
        }
        with pytest.raises(ValueError, match="no nourishment plans declared"):
            _expand_plans(cfg)
