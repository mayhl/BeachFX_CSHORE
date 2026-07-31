"""Storm-driven beach erosion simulation on CSHORE — the erosion module of CHART FEAT.

Orientation (the run path, in call order):

    pipeline.py     config -> (reach x alternative x lifecycle) jobs -> Dask dispatch
    reach.py        the interval loop over one lifecycle's storm schedule
    interstorm.py   background erosion + sea-level ticks between storms
    storm.py        one storm: parallel CSHORE per profile + response classification
    nourishment/    post-storm campaigns: assess deficits, execute placements
    profile.py      Profile data + the event taxonomy (tick/storm/recovery/nourishment)
    runner/         CSHORE execution boundary (subprocess, file I/O, mock)
    results.py      the parquet sink (one output dir per job)

The pure side:

    decision/       management-calendar planning: campaign gate, crew clock,
                    planned cycles.  Scalars and dates in, decision dataclasses
                    out; never imports physics.
    metrics/        morphology fitting: berm/dune detection + idealized forms.

Offline (not part of a run):

    postprocess.py  common-grid registration pass over a finished run dir
    summary.py      derived metrics over a finished run dir
    viz.py          plots and animations over run output
    sweep.py        gen-alternatives CLI (cartesian nourishment-plan tables)

Shared vocabulary lives in ``types.py`` (snapshot labels, campaign/decision
kinds), unit-aware config fields in ``units.py``, and the config root model in
``config.py``.  Start reading at ``reach.py`` — its phase map is the story of
one lifecycle.
"""
