"""Pytest configuration — and the map of the suite.

The layout (what lives where):

    builders.py         shared builders: profiles, storms, ``ncfg`` (the one
                        nourishment-policy doorway), and the lifecycle ``run()``
                        (its ``runner`` is always explicit)
    doubles.py          morphology-stated CSHORE stand-ins (``BermCut``,
                        ``DuneScarp``, ``Overwash``, ...) + ``ScriptedRunner``
    synthetic.py        parametric synthetic profile generator (``make_profile``)
    erosion/            unit tests: metrics fitting (+ goldens), results sink,
                        runner preflight, postprocess summaries
    decision/           the pure planner suite (gates, crew clock, blackouts,
                        cycles) and calendar-state pickling
    lifecycle/          the interval loop end-to-end on doubles: profile events,
                        storms, campaigns, planned cycles, event sequences, and
                        the golden output timelines (``test_golden_timelines``)
    integration/        only tests needing the real CSHORE binary — opt in with
                        ``pytest -m integration``
    goldens/            JSON regression fixtures; regenerate fits + real via
                        ``REGEN_FIT_GOLDENS=1``, recovery via ``REGEN_RECOVERY_GOLDENS=1``
    fit_gallery.py, recovery_gallery.py
                        golden-case galleries (``--plot`` renders them)

Shared test data builders are imported directly by the test modules — they are
not exposed as fixtures here.

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
        help="Print an expected-vs-generated lifecycle event table (with OK/FAIL) "
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
        reach_id=None,
    ):
        # ``expected`` entries are a label ``L`` or a time-pinned ``(L, t)`` tuple.
        exp_labels = [e[0] if isinstance(e, tuple) else e for e in expected]
        exp_times = [e[1] if isinstance(e, tuple) else None for e in expected]
        exp = [getattr(x, "value", x) for x in exp_labels]
        gen = [getattr(x, "value", x) for x in generated_labels]

        # One row per event step: index, event, elapsed days, calendar date, and a
        # per-event verdict (``ok`` for a matched time-pin, ``·`` unpinned, ``FAIL``
        # on a label or time mismatch).  ``expected`` shows through only when it diverges.
        n = max(len(gen), len(exp))
        events = []
        row_ok = True
        for i in range(n):
            g = gen[i] if i < len(gen) else None
            e = exp[i] if i < len(exp) else None
            has_t = generated_times is not None and i < len(generated_times)
            t = generated_times[i] if has_t else None
            want_t = exp_times[i] if i < len(exp_times) else None
            label_ok = g == e
            time_ok = want_t is None or (t is not None and abs(t - want_t) <= 1e-6)
            match = label_ok and time_ok
            if not match:
                row_ok = False
            verdict = "FAIL" if not match else ("ok" if want_t is not None else "·")
            events.append(
                {
                    "idx": i,
                    "event": g if g is not None else "—",
                    "elapsed": t,
                    "date": str((sim_start + timedelta(days=t)).date())
                    if (sim_start is not None and t is not None)
                    else "—",
                    "verdict": verdict,
                    "expected": e if not label_ok else None,
                }
            )

        # interval-in-days notes, e.g. "SEN#0→SEN#1 14d" — endpoints are (label, nth)
        dur_notes = (
            [f"{a[0].value}#{a[1]}→{b[0].value}#{b[1]} {days:g}d" for a, b, days in durations]
            if durations
            else []
        )
        request.config._event_rows.append(
            {
                "scenario": scenario,
                "desc": desc,
                "reach": reach_id,
                "profile": profile_id,
                "events": events,
                "durations": dur_notes,
                "decisions": [getattr(d, "value", d) for d in decisions] if decisions else [],
                "status": "OK" if row_ok else "FAIL",
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

    # Group rows (one per profile) under their scenario, preserving first-seen order.
    by_scenario: dict[str, list] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario"], []).append(r)

    glyph = {"ok": "✓", "·": "·", "FAIL": "✗"}

    def _seq(events):
        # arrow-joined sequence; pinned events (verdict != "·") show day + date
        out = []
        for e in events:
            if e["verdict"] != "·" and e["elapsed"] is not None:
                out.append(f"{e['event']}@{e['elapsed']:g}d({e['date']})")
            else:
                out.append(str(e["event"]))
        return " → ".join(out)

    def _rule(widths, left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def _row(cells, widths, aligns, indent=""):
        parts = []
        for c, w, a in zip(cells, widths, aligns):
            parts.append(f" {c:>{w}} " if a == ">" else f" {c:<{w}} ")
        return indent + "│" + "│".join(parts) + "│"

    # --- overview: one line per profile ------------------------------------
    tw.write_line("")
    tw.section("Reach-loop event sequences — overview", sep="═")
    w_scn = max([len("SCENARIO")] + [len(r["scenario"]) for r in rows])
    tw.write_line(f"  {'SCENARIO':<{w_scn}}  {'PROF':<4}  {'STATUS':<6}  EVENTS")
    tw.write_line(f"  {'─' * w_scn}  {'─' * 4}  {'─' * 6}  {'─' * 6}")
    for r in rows:
        ok = r["status"] == "OK"
        markup = {"green": True} if ok else {"red": True, "bold": True}
        status = ("✓ " if ok else "✗ ") + r["status"]
        tw.write(f"  {r['scenario']:<{w_scn}}  {r['profile']:<4}  ")
        tw.write(f"{status:<8}", **markup)
        tw.write_line(f"  {_seq(r['events'])}")

    # --- per-scenario detail: one reach timeline, all profiles merged -------
    # Events from every profile are combined and ordered by time, so the table
    # reads as the reach's actual event sequence (crew serialization, interrupts,
    # simultaneous storms).  Δt is recomputed across the merged order, not per
    # profile.  Ties (same t) break by profile id, then each profile's causal idx.
    tw.write_line("")
    tw.section("Per-scenario detail (reach timeline)", sep="═")
    headers = ["#", "REACH", "PROFILE", "EVENT", "Δt", "DATE", "OK"]
    aligns = [">", "<", "<", "<", ">", "<", "<"]
    for scenario, profiles in by_scenario.items():
        desc = profiles[0]["desc"]
        tw.write_line("")
        tw.write_line(f"▌ {scenario} — {desc}", bold=True)

        merged = []  # (t, profile_id, idx, reach, event)
        for r in profiles:
            reach = r.get("reach") or "—"
            for e in r["events"]:
                merged.append((e["elapsed"], r["profile"], e["idx"], reach, e))
        # reach timeline: order by date, then profile id, then each profile's causal idx
        merged.sort(key=lambda m: (m[0] if m[0] is not None else float("-inf"), m[1], m[2]))

        notes = []
        for r in profiles:
            if r.get("decisions"):
                notes.append(f"decisions: {' → '.join(r['decisions'])}")
            if r.get("durations"):
                notes.append(f"intervals: {', '.join(r['durations'])}")
        if notes:
            tw.write_line("  " + "   ·   ".join(notes))

        body, prev_t = [], None
        for seq, (t, prof, _idx, reach, e) in enumerate(merged):
            dt = "—" if (prev_t is None or t is None) else f"{t - prev_t:g}d"
            body.append(
                [
                    str(seq),
                    reach,
                    str(prof),
                    str(e["event"]),
                    dt,
                    e["date"],
                    glyph.get(e["verdict"], e["verdict"]),
                ]
            )
            if t is not None:
                prev_t = t
        widths = [max(len(headers[c]), *(len(row[c]) for row in body)) for c in range(len(headers))]

        tw.write_line(_rule(widths, "  ┌", "┬", "┐"))
        tw.write_line(_row(headers, widths, aligns, indent="  "))
        tw.write_line(_rule(widths, "  ├", "┼", "┤"))
        for (_t, _prof, _idx, _reach, e), cells in zip(merged, body):
            line = _row(cells, widths, aligns, indent="  ")
            if e["verdict"] == "FAIL":
                exp = f"   ← expected {e['expected']}" if e["expected"] is not None else ""
                tw.write_line(line + exp, red=True, bold=True)
            else:
                tw.write_line(line)
        tw.write_line(_rule(widths, "  └", "┴", "┘"))
