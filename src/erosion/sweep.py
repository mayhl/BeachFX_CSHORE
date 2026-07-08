#!/usr/bin/env python3
"""Offline generator: expand a parameter product into numbered ``alternatives``.

The pipeline itself has no product/sweep machinery — it just runs an id-keyed
table of alternatives, selected via the ``run`` spec (see ``pipeline._parse_run_spec``).
This helper *authors* that table: give it a base config, a set of axes (dotted
config paths -> value lists), and a mode, and it emits ``{id: config}`` numbered
entries you paste into a config's ``alternatives`` block (then ``run: all`` or a
range like ``1-4,8``).

Cartesian = cross every axis; zip = pair them positionally (all axes same length).

    uv run gen-alternatives spec.json            # print the alternatives block
    uv run gen-alternatives spec.json > alts.json

``spec.json``::

    {
      "mode": "cartesian",
      "base": { "storm": {"T_recover": 21.0}, "nourishment": {...} },
      "axes": {
        "nourishment.borrow_to_placement_ratio": [1.0, 1.5],
        "nourishment.storm_conflict": ["interrupt", "defer"]
      }
    }
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import sys

# Reserved axis name: sets the run's top-level lifecycle (a Monte-Carlo storm
# realization in the storm file), not a nested config field.  Crossed like any
# other axis, so `(param axes) × lifecycle` fans out one run per lifecycle.
LIFECYCLE_AXIS = "lifecycle"


def _expand_range(spec) -> list[int]:
    """Expand a lifecycle spec into a list of ints: ``"0-3,7"`` or ``[0,1,2]``."""
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    ids: list[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"lifecycle: descending range {tok!r}")
            ids.extend(range(lo, hi + 1))
        else:
            ids.append(int(tok))
    return ids


def _set_dotted(d: dict, path: str, value) -> None:
    """Set ``d[a][b][c] = value`` for a dotted ``path`` ("a.b.c"), creating dicts."""
    keys = path.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"axis path {path!r} collides with a non-dict at {k!r}")
    node[keys[-1]] = value


def _points(axes: dict[str, list], mode: str) -> list[tuple]:
    """Value tuples (aligned to ``axes`` keys) for the product or the zip."""
    values = list(axes.values())
    if mode == "cartesian":
        return list(itertools.product(*values))
    if mode == "zip":
        lengths = {len(v) for v in values}
        if len(lengths) > 1:
            raise ValueError(
                f"zip mode needs equal-length axes; got { {k: len(v) for k, v in axes.items()} }"
            )
        return list(zip(*values))
    raise ValueError(f"mode must be 'cartesian' or 'zip', got {mode!r}")


def expand_alternatives(
    base: dict,
    axes: dict[str, list],
    mode: str = "cartesian",
    start_id: int = 1,
) -> dict[str, dict]:
    """Expand a parameter product into numbered alternative configs.

    ``base`` is the template each point starts from (deep-copied, never mutated).
    ``axes`` maps a dotted config path -> list of values.  The reserved axis
    ``lifecycle`` sets the run's top-level lifecycle (its value may be a list or a
    range string like ``"0-49"``); it is crossed like any other axis, so
    ``(param axes) × lifecycle`` yields one run per lifecycle.  Returns ``{str(id):
    config}`` counting from ``start_id``; each config carries a ``name`` summarizing
    its axis values (leaf=value pairs).  ``mode`` is ``"cartesian"`` or ``"zip"``.
    """
    if not axes:
        raise ValueError("expand_alternatives: no axes given")
    axes = dict(axes)
    if LIFECYCLE_AXIS in axes:
        axes[LIFECYCLE_AXIS] = _expand_range(axes[LIFECYCLE_AXIS])
    keys = list(axes)
    out: dict[str, dict] = {}
    for i, point in enumerate(_points(axes, mode), start=start_id):
        cfg = copy.deepcopy(base)
        for path, value in zip(keys, point):
            if path == LIFECYCLE_AXIS:
                cfg[LIFECYCLE_AXIS] = int(value)
            else:
                _set_dotted(cfg, path, value)
        cfg["name"] = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, point))
        out[str(i)] = cfg
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gen-alternatives",
        description="Expand a {base, axes, mode} spec into a numbered alternatives block.",
    )
    parser.add_argument("spec", help="path to a JSON spec ({mode, base, axes})")
    parser.add_argument("--start-id", type=int, default=1, help="first id (default: 1)")
    args = parser.parse_args(argv)

    with open(args.spec) as f:
        spec = json.load(f)
    alts = expand_alternatives(
        spec.get("base", {}),
        spec["axes"],
        mode=spec.get("mode", "cartesian"),
        start_id=args.start_id,
    )
    json.dump({"alternatives": alts}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    print(
        f"# {len(alts)} alternative(s) generated ({spec.get('mode', 'cartesian')})", file=sys.stderr
    )


if __name__ == "__main__":
    main()
