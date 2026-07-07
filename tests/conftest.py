"""Pytest configuration.

Shared test data builders (profiles, storms, forcing, configs, lifecycle runner)
live in ``tests/builders.py`` and are imported directly by the test modules —
they are not exposed as fixtures here.

``--plot`` opts in to writing diagnostic plot artifacts (e.g. the fit gallery in
``tests/fit_gallery.py``); it is off by default so the suite stays fast and
headless.  Any test may depend on the ``plot`` fixture and skip its plotting when
it is False.  The matplotlib backend is forced to Agg before any test module
imports ``erosion.viz`` (which imports pyplot at module load).
"""

from datetime import timedelta

import matplotlib
import pytest

matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Opt in to writing diagnostic plot artifacts to tests/_artifacts/; "
        "off by default so the suite stays fast and headless.",
    )
    parser.addoption(
        "--events-table",
        action="store_true",
        default=False,
        help="Print an expected-vs-generated orchestrator event table (with OK/FAIL) "
        "in the terminal summary.",
    )


def pytest_configure(config):
    config._event_rows = []


@pytest.fixture
def plot(request) -> bool:
    """True when the suite was run with ``--plot`` (opt-in diagnostic plots)."""
    return request.config.getoption("--plot")


@pytest.fixture
def record_event(request):
    """Record an (expected, generated) event-sequence row for the summary table.

    Always collects; the table is only printed when ``--events-table`` is passed.
    ``expected`` / ``generated`` are label sequences (enums or their values).
    """

    def _rec(
        scenario,
        desc,
        profile_id,
        expected,
        generated_labels,
        generated_times=None,
        sim_start=None,
        durations=None,
        decisions=None,
    ):
        # ``expected`` entries are a label ``L`` or a time-pinned ``(L, t)`` tuple.
        exp_labels = [e[0] if isinstance(e, tuple) else e for e in expected]
        exp_times = [e[1] if isinstance(e, tuple) else None for e in expected]
        exp = [getattr(x, "value", x) for x in exp_labels]
        gen = [getattr(x, "value", x) for x in generated_labels]
        ok = exp == gen
        if ok and generated_times is not None:
            for i, want in enumerate(exp_times):
                if want is not None and (
                    i >= len(generated_times) or abs(generated_times[i] - want) > 1e-6
                ):
                    ok = False
                    break

        def _annot(i, g):
            # pinned events show day + calendar date, e.g. SSN@20d(2030-01-21)
            if generated_times is None or i >= len(exp_times) or exp_times[i] is None:
                return g
            t = generated_times[i]
            if sim_start is not None:
                return f"{g}@{t:g}d({(sim_start + timedelta(days=t)).date()})"
            return f"{g}@{t:g}d"

        disp = [_annot(i, g) for i, g in enumerate(gen)]
        # interval-in-days notes, e.g. "SSN→ESN 3d"
        dur_notes = (
            [f"{gen[a]}→{gen[b]} {days:g}d" for a, b, days in durations] if durations else []
        )
        request.config._event_rows.append(
            {
                "scenario": scenario,
                "desc": desc,
                "profile": profile_id,
                "expected": exp,
                "generated": disp,
                "durations": dur_notes,
                "decisions": [getattr(d, "value", d) for d in decisions] if decisions else [],
                "status": "OK" if ok else "FAIL",
            }
        )

    return _rec


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--events-table"):
        return
    rows = getattr(config, "_event_rows", [])
    if not rows:
        return

    tw = terminalreporter
    tw.write_line("")
    tw.section("Orchestrator event sequences (expected vs generated)", sep="=")

    w_scn = max(len(r["scenario"]) for r in rows)
    tw.write_line(f"{'SCENARIO':<{w_scn}}  PROF  STATUS  EVENTS")
    for r in rows:
        ok = r["status"] == "OK"
        markup = {"green": True} if ok else {"red": True, "bold": True}
        seq = " → ".join(r["generated"])
        tw.write(f"{r['scenario']:<{w_scn}}  {r['profile']:<4}  ")
        tw.write(f"{r['status']:<6}", **markup)
        tw.write_line(f"  {seq}")
        pad = " " * (w_scn + 2 + 6 + 8)
        if r.get("decisions"):
            tw.write_line(f"{pad}decisions (reach): {' → '.join(r['decisions'])}")
        if r.get("durations"):
            tw.write_line(f"{pad}intervals: {', '.join(r['durations'])}")
        if not ok:
            tw.write_line(f"{pad}expected: {' → '.join(r['expected'])}")
