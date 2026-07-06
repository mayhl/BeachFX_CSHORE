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


@pytest.fixture
def plot(request) -> bool:
    """True when the suite was run with ``--plot`` (opt-in diagnostic plots)."""
    return request.config.getoption("--plot")
