"""Unit-aware float annotation for pydantic config models.

``ufloat(internal, default_input)`` returns an ``Annotated[float, ...]`` type that:

- Bare ``float`` / ``int``: treated as ``default_input`` unit (or ``internal``
  when ``default_input`` is None), stored as ``internal`` unit internally.
- ``{"value": float, "units": str}``: converts from the named unit to ``internal``.
- Pydantic validation context ``{"input_units": "m" | "ft"}`` overrides the
  bare-float default unit when ``default_input`` is set.

Fields without ``default_input`` (all ``CSHOREConfig`` fields) are context-immune:
a bare float always means the internal unit regardless of context.

Conversion factors
------------------
All factors convert to a canonical base for each dimension:
  length       → m       (ft, mm, cm)
  time         → days    (hours)
  3-D volume   → m³      (cy = 0.7646 m³)
  3-D vol rate → m³/day  (cy/day, cy/yr, m³/yr)
  2-D volume   → m²/m    (cy/ft  = 0.7646 m³ / 0.3048 m = 2.5083 m²/m)
  2-D vol rate → m²/m/d  (cy/ft/day, same linear factor as cy/ft)
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator

_CY_TO_M3: float = 27.0 * 0.3048**3  # 1 yd³ in m³ = 0.764554857984 (exact)
M3_TO_CY: float = 1.0 / _CY_TO_M3  # reporting direction; keep the pair exact reciprocals

_FACTORS: dict[str, float] = {
    # length
    "m": 1.0,
    "ft": 0.3048,
    "mm": 0.001,
    "cm": 0.01,
    # time
    "days": 1.0,
    "hours": 1.0 / 24.0,
    # 3-D volume
    "m3": 1.0,
    "cy": _CY_TO_M3,  # cubic yards → m³
    # 3-D volume rate (canonical base: m³/day)
    "m3/day": 1.0,
    "cy/day": _CY_TO_M3,
    "m3/yr": 1.0 / 365.25,
    "cy/yr": _CY_TO_M3 / 365.25,
    # 2-D volume (area per longshore length)
    "m2/m": 1.0,
    "cy/ft": _CY_TO_M3 / 0.3048,  # ≈ 2.5083
    # 2-D volume rate
    "m2/m/day": 1.0,
    "cy/ft/day": _CY_TO_M3 / 0.3048,
}

# When context "input_units" == "m", map ft-default units to their SI equivalents.
_CONTEXT_EQUIV: dict[str, str] = {
    "ft": "m",
    "cy": "m3",
    "cy/day": "m3/day",
    "cy/yr": "m3/yr",
    "cy/ft": "m2/m",
    "cy/ft/day": "m2/m/day",
}

M_TO_FT: float = 1.0 / 0.3048  # 3.28084 ft/m


def _convert(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return value
    return value * _FACTORS[from_unit] / _FACTORS[to_unit]


def _coerce(
    v: Any,
    internal: str,
    default_input: str | None,
    ctx_units: str | None,
) -> float:
    """Core unit coercion shared by the pydantic validator and ``parse_ufloat``.

    Accepts a ``{"value", "units"}`` dict or a bare number.  Bare numbers use
    ``default_input`` (or ``internal`` when that is ``None``), with ``ctx_units
    == "m"`` remapping ft-default units to their SI equivalents via
    ``_CONTEXT_EQUIV``.
    """
    if isinstance(v, dict):
        raw, unit = float(v["value"]), str(v["units"])
    elif isinstance(v, (int, float)):
        raw = float(v)
        if default_input is None:
            return raw  # context-immune
        if ctx_units == "m" and default_input in _CONTEXT_EQUIV:
            unit = _CONTEXT_EQUIV[default_input]
        else:
            unit = default_input
    else:
        raise ValueError(
            f'expected float or {{"value": ..., "units": ...}} dict, got {type(v).__name__}'
        )
    return _convert(raw, unit, internal)


def ufloat(internal: str, default_input: str | None = None) -> Any:
    """Return an ``Annotated[float, ...]`` pydantic field type.

    Parameters
    ----------
    internal:
        Unit stored internally by the framework after validation.
    default_input:
        Unit assumed for bare floats in the config JSON.  When ``None`` the
        field is context-immune (bare float = internal unit always).  When set,
        a pydantic validation context ``{"input_units": "m"}`` can override this
        default at model-validate time.
    """

    def _parse(v: Any, info: Any) -> float:
        ctx = getattr(info, "context", None) or {}
        return _coerce(v, internal, default_input, ctx.get("input_units"))

    return Annotated[float, BeforeValidator(_parse)]


def parse_ufloat(
    v: Any,
    internal: str,
    default_input: str | None = None,
    context: dict[str, str] | None = None,
) -> float:
    """Parse a unit-aware value outside of a pydantic model.

    Useful in ``run_pipeline.py`` for reach-level geometry fields
    (e.g. ``longshore_width``) that live outside ``ReachConfig``.
    """
    return _coerce(v, internal, default_input, (context or {}).get("input_units"))


def to_output_length(value: float, output_unit: str) -> float:
    """Convert an internal meter (or m²/m) value to the requested output unit."""
    return value * M_TO_FT if output_unit == "ft" else float(value)
