# CHART FEAT — erosion module

The **erosion module** of CHART FEAT: a lifecycle orchestrator that chains live
CSHORE cross-shore morphology runs storm-to-storm, with recovery, sea-level
change, and emergency nourishment layered on the gridded-array chain.

## Where to start

- **The repository `README.md`** — orientation, quick start, config format,
  output layout. Start here.
- **API reference** — generated from the source docstrings:
  [Nourishment](api/nourishment.md) · [Metrics / fitter](api/metrics.md) ·
  [Storm & recovery](api/storm.md).
- **Design specs (local only)** — [Nourishment model](design/NOURISHMENT_MODEL.md)
  (the per-storm decision tree and its numerics) and
  [Architecture of record](design/ARCHITECTURE_nourishment.md) (the BeachFX ↔
  MATLAB lineage merge and design rationale). These are untracked working
  documents — the links resolve only in a checkout that has them.

## The one-paragraph model

Each storm runs CSHORE per profile (Phase 2), erosion/SLC fill the inter-storm
gaps (Phases 1 & 2.5), and a campaign (Phase 3) assesses each profile
(**Tier 1**), decides reach-scope go/no-go and order (**Tier 2**), and places
nourishment with a serial crew (**Tier 3**) before recovering whatever the crew
did not reach. A recovery a following storm cuts short of `T_recover` is labelled
`RECS`; one that runs its full period is `REC`.

> **Note:** the two Design pages are local, untracked design specs
> (`.git/info/exclude`); this site renders them locally via `mkdocs serve`.
