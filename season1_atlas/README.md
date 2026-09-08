# Season 1 Research Atlas

This directory is a navigation and evidence layer for the original Gravimem/SubQ research. It does not replace or rewrite the historical record.

## Preservation Rule

The following remain the detailed Season 1 sources of record:

- `../experiments/RESEARCH_JOURNEY.md`
- `../experiments/README.md`
- `../STATE_OF_SUBQ.md`
- `../experiments/*.py`
- `../checkpoints/`
- `../experiments/*.png`
- `../Feature-Escrow-Networks/`
- `../archive/gravimem_retrieval/`

The atlas records what was asked, what was reported, and where the supporting artifact lives. It does not delete an experiment because its interpretation was weak.

## Evidence Labels

- `documented-run`: a result is recorded in the Season 1 documents, but the raw output is not present locally.
- `local-artifact`: a checkpoint, image, result file, or runnable implementation is present locally.
- `reproduced`: independently rerun and checked in the current environment.
- `confounded`: an interesting result exists, but a comparison has a parameter, causality, training-budget, or implementation confound.
- `hypothesis`: a proposed mechanism or explanation without a decisive isolation experiment.
- `unresolved`: still useful as a Season 2 question.
- `superseded`: an older implementation or interpretation replaced by later work, while the original record remains preserved.

## Contents

- [Study catalog](study_catalog.md): questions, recorded results, status, and source locations.
- [Core result matrix](result_matrix.md): the main numerical comparisons worth carrying forward.
- [Unresolved ideas](unresolved_ideas.md): hypotheses retained for future use, including ideas whose first explanation was overconfident.

## How To Use This

When a Season 2 result appears related to Season 1, add a short cross-reference to the relevant atlas entry. Do not copy a large historical table into Season 2 unless the comparison is being rerun under the new protocol.
