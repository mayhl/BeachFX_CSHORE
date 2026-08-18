"""LocalCSHORERunner preflight: a broken install fails at construction, not per storm."""

import os

import pytest

from erosion.runner import CSHOREParams, LocalCSHORERunner
from erosion.runner import local as local_mod


def _make(tmp_path):
    return LocalCSHORERunner(CSHOREParams(), work_dir=str(tmp_path))


class TestPreflight:
    def test_missing_binary_raises_at_init(self, tmp_path, monkeypatch):
        missing = str(tmp_path / "no_such_cshore.out")
        monkeypatch.setattr(local_mod, "_exe_path", lambda: missing)
        with pytest.raises(FileNotFoundError, match="CSHORE binary not found"):
            _make(tmp_path)

    def test_non_executable_binary_raises_at_init(self, tmp_path, monkeypatch):
        exe = tmp_path / "cshore.out"
        exe.write_bytes(b"\x00")
        os.chmod(exe, 0o644)
        monkeypatch.setattr(local_mod, "_exe_path", lambda: str(exe))
        with pytest.raises(PermissionError, match="not executable"):
            _make(tmp_path)

    def test_shipped_binary_passes_preflight(self, tmp_path):
        _make(tmp_path)


class TestSolverWarningScan:
    """OMESSG is the solver's only trace of non-convergence; the scan must read it."""

    def test_scan_solver_warnings_counts_lines(self, tmp_path):
        (tmp_path / "OMESSG").write_text(
            "END OF LANDWARD MARCHING\n"
            "NO CONVERGENCE OF QO ITERATION\n"
            "Square of sigma SIGTIE is negative\n"
            "NO CONVERGENCE OF QO ITERATION\n"
            "ordinary line\n"
        )
        assert local_mod._scan_solver_warnings(str(tmp_path)) == (2, 1)

    def test_scan_missing_omessg_is_clean(self, tmp_path):
        assert local_mod._scan_solver_warnings(str(tmp_path)) == (0, 0)
