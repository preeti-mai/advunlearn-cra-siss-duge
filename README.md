# CRA + SISS + DUGE (Zero-Edit Wrapper for AdvUnlearn)

This repo reproduces our unlearning recipe **without modifying** the upstream trainer. We orchestrate phases, curriculum, and alignment externally.

## Summary
- **Base trainer:** [OPTML-Group/AdvUnlearn] as a pinned submodule (`vendor/AdvUnlearn`).
- **CRA (anchor reconditioning):** small external micro-finetunes that match the current text encoder to the base encoder on benign anchors.
- **SISS (importance sampling):** we duplicate hard prompts in the training list according to softmax(difficulty/τ).
- **DUGE (gated schedule):** phase-by-phase runs of the trainer with decreasing attack strength and retain weight; each phase advances only if ASR/FID gates are satisfied.

## Reproducibility
- Upstream commit pinned via `git submodule` (see below).
- Exact seeds, configs, and environment recorded.
- CLI orchestrator logs ASR/FID JSON after every phase.
- Checkpoints saved at each phase boundary.

## Environment
```bash
bash scripts/setup_env.sh
