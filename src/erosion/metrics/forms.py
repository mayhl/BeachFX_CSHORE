"""Idealized-profile assembly, dune-form selection (triangle / trapezoid /
skew-gaussian by BIC), and scarp/quality diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .detect import _linear_slope, _longest_flat_run
from .types import (
    _BENCH_MIN_STEP,
    _CREST_EPS,
    _FLAT_SLOPE,
    _GAUSS_MIN_NODES,
    _LEVEL_TOL,
    _MIN_PLATEAU_M,
    _NAN,
    _SCARP_MIN_H,
    _SCARP_RESID,
    _SCARP_SLOPE,
    _TOP_FRAC,
    IdealizedProfile,
    ProfileMetrics,
    _bic,
    _FormFit,
)


def _select_dune_form(
    x,
    zb,
    dx,
    datum,
    BE,
    UE,
    x_shore,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev_meas,
    seaward_base,
    seaward_toe_idx,
    crest_x,
    DE,
    upland_start,
    landward_toe_idx,
):
    """Fit candidate idealized dune forms and select by BIC.

    A triangular apex, a trapezoidal flat top (broad/multi-crest massif), and a
    skew-gaussian bump (rounded crest, independent front/back slopes).  Each is
    least-squares fit against the raw; the winner minimizes the BIC, which trades
    misfit against the crest-shape DOF (triangle 2 < trapezoid 3 < gaussian 4) so
    a genuinely peaked dune is not given a spurious flat top, and a rounded dune
    is not forced into a sharp apex, without a hand-tuned RMS margin.

    Returns ``(ideal, geom, fit_quality)`` where ``geom`` is the 10-tuple from
    ``_dune_geom_from_knots``.
    """

    def _knot_candidate(cx, cz, cx_end, k):
        idl = _ideal_dune(
            x,
            datum,
            x_shore,
            berm_present,
            berm_start_idx,
            berm_end_idx,
            berm_elev_meas,
            cx,
            cz,
            UE,
            upland_start,
            landward_toe_idx,
            seaward_toe_idx,
            crest_x_end=cx_end,
            seaward_toe_z=seaward_base,
        )
        # Keep a trapezoid's two top knots at a shared elevation through the fit.
        flat_top = None
        if cx_end is not None:
            tops = np.where(np.abs(idl.knots_z - cz) < 1e-9)[0]
            if len(tops) == 2:
                flat_top = (int(tops[0]), int(tops[1]))
        idl = _refine_dune(x, zb, idl.knots_x, idl.knots_z, datum, flat_top=flat_top)
        geom = _dune_geom_from_knots(idl.knots_x, idl.knots_z, berm_elev_meas, BE, berm_present, UE)
        return _FormFit(idl, _fit_quality(x, zb, idl, datum), k, geom)

    nwet = int(np.sum(zb > datum))
    cands = [_knot_candidate(crest_x, DE, None, 2)]  # triangle

    # Trapezoidal candidate: the broad near-crest band (a relief fraction below the
    # apex), leveled at its median so a bumpy multi-crest top idealizes flat.
    top_tol = max(_CREST_EPS, (1.0 - _TOP_FRAC) * (DE - seaward_base))
    band = np.arange(seaward_toe_idx, landward_toe_idx + 1)
    band = band[zb[band] >= DE - top_tol]
    if len(band) >= 2 and (x[band[-1]] - x[band[0]]) >= _MIN_PLATEAU_M:
        cands.append(
            _knot_candidate(float(x[band[0]]), float(np.median(zb[band])), float(x[band[-1]]), 3)
        )

    gauss = _gaussian_candidate(
        x,
        zb,
        datum,
        dx,
        seaward_toe_idx,
        landward_toe_idx,
        crest_x,
        DE,
        seaward_base,
        UE,
        berm_present,
        berm_start_idx,
        berm_end_idx,
        berm_elev_meas,
        x_shore,
    )
    if gauss is not None:
        cands.append(gauss)

    best = min(cands, key=lambda c: _bic(c.rms, nwet, c.k))
    return best.ideal, best.geom, best.rms


def _fit_no_dune(
    x,
    zb,
    zs,
    dx,
    datum,
    x_shore,
    morph_type,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev_meas,
    berm_width,
    shore_idx,
    seaward_toe_idx,
    upland_start,
    UE,
    vol,
    K,
    foreshore_slope,
):
    """Assemble metrics + idealized profile for a dune-less section (HIGH_UPLAND,
    or a crest that never clears the seaward toe).  Returns ``(metrics, ideal)``."""
    bench = (
        _mid_bench(x, zs, zb, dx, berm_end_idx + 1, upland_start, berm_elev_meas, UE)
        if berm_present
        else None
    )
    ideal = _ideal_no_dune(
        x,
        datum,
        x_shore,
        berm_present,
        berm_start_idx,
        berm_end_idx,
        berm_elev_meas,
        UE,
        upland_start,
        bench=bench,
    )
    # Berm-zone scarp detection — steep-run only, since a HIGH_UPLAND berm/
    # upland boundary is fuzzy and the residual path would false-flag a rising
    # back.  A berm-less beach still reads as scarped.
    berm_scarp, berm_sc_h = _scarp(x, zb, ideal, shore_idx, seaward_toe_idx, datum, residual=False)
    berm_scarp = berm_scarp or (not berm_present)
    m = ProfileMetrics(
        morph_type=morph_type,
        shoreline_x=x_shore,
        foreshore_slope=foreshore_slope,
        berm_elevation=berm_elev_meas,
        berm_width=berm_width,
        upland_elevation=UE,
        volume_above_datum=vol,
        berm_scarp=berm_scarp,
        scarp_height=berm_sc_h,
        n_upland_nodes=K,
    )
    m.fit_quality = _fit_quality(x, zb, ideal, datum)
    return m, ideal


def _knots_to_ideal(kx: list[float], kz: list[float]) -> IdealizedProfile:
    order = np.argsort(kx)
    return IdealizedProfile(np.asarray(kx)[order], np.asarray(kz)[order])


# The morphology class selects the idealized *function form*.  LOW_UPLAND and
# LOW_BERM share the dune form (they differ only in the BE-vs-UE classification);
# HIGH_UPLAND (and any degenerate crest-below-toe) uses the no-dune form.


def _mid_bench(x, zs, zb, dx, lo, hi, berm_elev, UE):
    """A flat terrace between the berm and the upland (a stepped HIGH_UPLAND back).
    Returns ``(start, end, elev)`` or ``None``.  Requires a genuinely flat shelf
    set off from both the berm and the upland by a real step, so a monotonic ramp
    — whose drift-bounded segments look flat — is not split into a phantom bench."""
    if hi - lo < 3:
        return None
    K = max(3, int(round(_MIN_PLATEAU_M / dx)))
    b0, b1 = _longest_flat_run(zs, dx, lo, hi, z_ceiling=UE - _LEVEL_TOL)
    if b1 - b0 < K:
        return None
    bench_z = float(np.median(zb[b0 : b1 + 1]))
    slope = abs(_linear_slope(x[b0 : b1 + 1], zs[b0 : b1 + 1]))
    if (
        slope < _FLAT_SLOPE / 3
        and bench_z - berm_elev >= _BENCH_MIN_STEP
        and UE - bench_z >= _BENCH_MIN_STEP
    ):
        return b0, b1, bench_z
    return None


def _ideal_no_dune(
    x,
    datum,
    x_shore,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev,
    UE,
    upland_start,
    bench=None,
) -> IdealizedProfile:
    """HIGH_UPLAND form: shore → berm → [optional mid-bench] → rise to the upland
    level at the crop point → flat.  Never ramps across the whole back to the far
    end.  ``bench`` is an optional ``(start, end, elev)`` intermediate terrace."""
    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        kx += [float(x[berm_start_idx]), float(x[berm_end_idx])]
        kz += [float(berm_elev), float(berm_elev)]
        if bench is not None:
            b0, b1, bz = bench
            kx += [float(x[b0]), float(x[b1])]
            kz += [float(bz), float(bz)]
    # Rise to the upland level at the crop point, then flat — for berm AND
    # berm-less backs (else a no-berm HIGH_UPLAND ramps straight to the far end
    # and misses the plateau).
    kx.append(float(x[upland_start]))
    kz.append(float(UE))
    kx.append(float(x[-1]))
    kz.append(float(UE))
    return _knots_to_ideal(kx, kz)


def _ideal_dune(
    x,
    datum,
    x_shore,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev,
    crest_x,
    crest_z,
    UE,
    upland_start,
    landward_toe_idx,
    seaward_toe_idx,
    crest_x_end=None,
    seaward_toe_z=None,
) -> IdealizedProfile:
    """LOW_UPLAND / LOW_BERM form: shore → berm → dune(front, crest[, plateau],
    back) → upland.  Two-sided base: seaward toe on the berm, landward toe on the
    upland (generally different elevations)."""
    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        # Extend the flat berm to the seaward toe (the true ascent start) so the
        # dune front ramps from the toe, not across leftover berm.
        berm_land = (
            seaward_toe_idx
            if seaward_toe_idx is not None and seaward_toe_idx > berm_end_idx
            else berm_end_idx
        )
        kx += [float(x[berm_start_idx]), float(x[berm_land])]
        kz += [float(berm_elev), float(berm_elev)]
    elif (
        seaward_toe_idx is not None
        and seaward_toe_z is not None
        and x_shore < float(x[seaward_toe_idx]) < float(crest_x)
    ):
        # No berm: knot at the dune's seaward toe so the front bends at the
        # foreshore→dune-front knee rather than one straight line to the crest.
        kx.append(float(x[seaward_toe_idx]))
        kz.append(float(seaward_toe_z))
    kx.append(float(crest_x))
    kz.append(float(crest_z))
    # Flat-topped (trapezoidal) dune: second crest knot at the plateau's landward
    # edge so the top idealizes flat, not as a single apex.
    if crest_x_end is not None and crest_x_end > crest_x:
        kx.append(float(crest_x_end))
        kz.append(float(crest_z))
    toe = landward_toe_idx if landward_toe_idx is not None else upland_start
    kx.append(float(x[toe]))
    kz.append(float(UE))
    kx.append(float(x[-1]))
    kz.append(float(UE))
    return _knots_to_ideal(kx, kz)


def _refine_dune(x, zb, kx, kz, datum, win: float = 8.0, flat_top=None) -> IdealizedProfile:
    """Least-squares refine of the interior dune knots (seaward toe → landward
    toe) against the raw profile, starting from the detected idealization.

    Knot x stays within ±``win`` of detection; knot z is bounded to
    ``[datum, max(zb)]`` so the crest can only be pulled DOWN toward the data,
    never pushed above it — which corrects a noise-inflated sharp apex while
    leaving a rounded (gaussian) crest at its detected height.  Shoreline, berm,
    and upland knots stay fixed.  ``flat_top=(i, j)`` ties knot ``j``'s elevation
    to knot ``i``'s so a trapezoidal top stays flat (a genuine plateau, with a
    well-defined crest and width) instead of tilting into a general quadrilateral
    under the fit."""
    kx = np.asarray(kx, dtype=float).copy()
    kz = np.asarray(kz, dtype=float).copy()
    free = list(range(2, len(kx) - 1))  # interior: seaward toe → landward toe
    wet = zb > datum
    if not free or not np.any(wet) or float(np.max(zb)) <= datum:
        return IdealizedProfile(kx, kz)
    nf = len(free)
    z_ceiling = float(np.max(zb))

    def resid(p):
        kx2, kz2 = kx.copy(), kz.copy()
        kx2[free] = np.sort(p[:nf])
        kz2[free] = p[nf:]
        if flat_top is not None:
            kz2[flat_top[1]] = kz2[flat_top[0]]
        return (zb - np.interp(x, kx2, kz2, left=kz2[0], right=kz2[-1]))[wet]

    lo = np.concatenate([kx[free] - win, np.full(nf, datum)])
    hi = np.concatenate([kx[free] + win, np.full(nf, z_ceiling)])
    p0 = np.clip(np.concatenate([kx[free], kz[free]]), lo, hi)
    try:
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=2000)
        kx[free] = np.sort(sol.x[:nf])
        kz[free] = sol.x[nf:]
        if flat_top is not None:
            kz[flat_top[1]] = kz[flat_top[0]]
    except Exception:
        pass
    return IdealizedProfile(kx, kz)


def _gaussian_candidate(
    x,
    zb,
    datum,
    dx,
    seaward_toe_idx,
    landward_toe_idx,
    crest_x,
    DE,
    seaward_base,
    UE,
    berm_present,
    berm_start_idx,
    berm_end_idx,
    berm_elev_meas,
    x_shore,
) -> _FormFit | None:
    """Skew-gaussian dune candidate: a bump ``A·exp(−½((x−x₀)/σ)²)`` with
    *independent* front/back scales ``σ_f, σ_b`` riding on the linear two-sided toe
    baseline (seaward toe on the berm, landward toe on the upland).  Bounded
    least-squares against the raw profile over the dune region.  The fitted curve
    is sampled at the profile nodes into a piecewise-linear ``IdealizedProfile`` (so
    ``evaluate``/scarp/viz/golden are unchanged), with the outer shore/berm/upland
    knots stitched on.  Returns ``None`` if the region is too short or the fit
    fails.  Unlike the knot forms, the smooth flanks do not trip the residual scarp,
    so a rounded dune stops false-flagging a front scarp."""
    lo, hi = int(seaward_toe_idx), int(landward_toe_idx)
    if hi - lo + 1 < _GAUSS_MIN_NODES:
        return None
    xs, xl = float(x[lo]), float(x[hi])
    if xl <= xs:
        return None
    reg = slice(lo, hi + 1)
    xr, zr = x[reg], zb[reg]
    span = xl - xs

    def baseline(xx):  # linear two-sided base connecting the toes
        return seaward_base + (UE - seaward_base) * (xx - xs) / span

    base_r = baseline(xr)

    def curve(p):
        A, x0, sf, sb = p
        sig = np.where(xr <= x0, sf, sb)
        return base_r + A * np.exp(-0.5 * ((xr - x0) / np.maximum(sig, 1e-6)) ** 2)

    zceil = float(np.max(zb))
    A0 = max(DE - float(baseline(crest_x)), 0.1)
    p0 = [A0, float(crest_x), max((crest_x - xs) / 2.0, dx), max((xl - crest_x) / 2.0, dx)]
    lo_b = [0.0, xs, dx, dx]
    hi_b = [max(zceil - min(seaward_base, UE), 0.1), xl, span, span]
    p0 = np.clip(p0, lo_b, hi_b)
    try:
        sol = least_squares(lambda p: curve(p) - zr, p0, bounds=(lo_b, hi_b), max_nfev=2000)
    except Exception:
        return None
    A, x0, sf, sb = (float(v) for v in sol.x)
    # Keep the crest under the data, like the knot refinement's z-ceiling.
    if float(baseline(x0)) + A > zceil:
        A = zceil - float(baseline(x0))

    # Sample the fitted curve to knots; pin the toe endpoints to the baseline so
    # the dune segment meets the berm/upland cleanly.
    z_samp = baseline(xr) + A * np.exp(
        -0.5 * ((xr - x0) / np.maximum(np.where(xr <= x0, sf, sb), 1e-6)) ** 2
    )
    z_samp = z_samp.copy()
    z_samp[0], z_samp[-1] = seaward_base, UE

    kx: list[float] = [float(x_shore)]
    kz: list[float] = [float(datum)]
    if berm_present:
        berm_land = seaward_toe_idx if seaward_toe_idx > berm_end_idx else berm_end_idx
        kx += [float(x[berm_start_idx]), float(x[berm_land])]
        kz += [float(berm_elev_meas), float(berm_elev_meas)]
    kx += [float(v) for v in xr]
    kz += [float(v) for v in z_samp]
    kx.append(float(x[-1]))
    kz.append(float(UE))
    ideal = _knots_to_ideal(kx, kz)

    rms = _fit_quality(x, zb, ideal, datum)
    geom = _gaussian_geom(A, x0, sf, sb, xs, xl, seaward_base, UE, baseline)
    return _FormFit(ideal, rms, 4, geom)


def _gaussian_geom(A, x0, sf, sb, xs, xl, seaward_base, UE, baseline):
    """Dune geometry from the skew-gaussian parameters (crest from ``x₀/A``; widths
    are toe→crest footprints, slopes toe-to-crest secants — the same shape-agnostic
    definitions the knot forms use).  A gaussian has no plateau, so top width = 0."""
    crest_x = float(np.clip(x0, xs, xl))
    DE = float(baseline(crest_x) + A)
    front_width = crest_x - xs
    back_width = xl - crest_x
    front_relief = DE - seaward_base
    back_relief = DE - UE
    return (
        DE,
        crest_x,
        0.0,  # top width (peaked, no plateau)
        front_width,
        back_width,
        xl - xs,
        front_relief,
        back_relief,
        max(0.0, front_relief / max(front_width, 1e-9)),
        max(0.0, back_relief / max(back_width, 1e-9)),
    )


def _dune_geom_from_knots(kx, kz, berm_elev, BE, berm_present, UE):
    """Read dune geometry (crest, widths, slopes, reliefs) back from the refined
    idealized knots.  The crest is the highest knot (a plateau spans several)."""
    DE = float(np.max(kz))
    crest_ks = np.where(kz >= DE - 1e-9)[0]
    c0, c1 = int(crest_ks[0]), int(crest_ks[-1])
    crest_x = float(np.mean(kx[crest_ks]))
    top_width = float(kx[c1] - kx[c0])
    st, lt = max(c0 - 1, 0), min(c1 + 1, len(kx) - 1)
    st_x, st_z = float(kx[st]), float(kz[st])
    lt_x, lt_z = float(kx[lt]), float(kz[lt])
    seaward_base = berm_elev if berm_present else BE
    front_slope = max(0.0, (DE - st_z) / max(float(kx[c0]) - st_x, 1e-9))
    back_slope = max(0.0, (DE - lt_z) / max(lt_x - float(kx[c1]), 1e-9))
    return (
        DE,
        crest_x,
        top_width,
        crest_x - st_x,
        lt_x - crest_x,
        lt_x - st_x,  # front/back/total width
        DE - seaward_base,
        DE - UE,  # front/back relief
        front_slope,
        back_slope,
    )


def _fit_quality(x, zb, ideal: IdealizedProfile, datum: float) -> float:
    mask = zb > datum
    if not np.any(mask):
        return _NAN
    resid = zb[mask] - ideal.evaluate(x[mask])
    return float(np.sqrt(np.mean(resid**2)))


def _scarp(
    x, zb, ideal: IdealizedProfile, lo: int, hi: int, datum: float, residual: bool = True
) -> tuple[bool, float]:
    """Flag a scarp in [lo, hi] via a steep run OR a large idealized residual.
    Returns (present, scarp_height).  Set ``residual=False`` to use the steep-run
    signal only — needed where the idealized base is unreliable (a fuzzy
    HIGH_UPLAND berm/upland boundary would otherwise read a rising back as a
    residual "scarp")."""
    if hi - lo < 1:
        return False, _NAN
    xs, zs = x[lo : hi + 1], zb[lo : hi + 1]
    dz = np.abs(np.diff(zs))
    dxs = np.diff(xs)
    steep = dz / np.maximum(dxs, 1e-9) >= _SCARP_SLOPE
    steep_h = float(np.sum(dz[steep])) if np.any(steep) else 0.0
    resid_max = float(np.max(np.abs(zs - ideal.evaluate(xs)))) if residual else 0.0
    present = (steep_h >= _SCARP_MIN_H) or (resid_max >= _SCARP_RESID)
    return present, max(steep_h, resid_max) if present else _NAN
