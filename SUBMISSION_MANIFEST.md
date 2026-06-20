# Submission Manifest

## Kept

- Core EB-JEPA modules required for world-model training and rollout planning.
- Dynamic maze dataset, maze solver/generator helpers, and Two Rooms helpers
  needed by shared dataset utilities and original Track 7 configs.
- Dynamic maze scripts:
  - strict A* baselines,
  - A*-distilled subgoal N=2/N=4 evaluation,
  - bootstrap value / probe-pos / repr-dist evaluation,
  - BeliefLSTM value evaluation,
  - gate-layout visualization,
  - report builder.
- Slurm launchers used for Dalia runs.
- Curated outputs: metrics, GIFs, PDF report, and PPTX presentation.

## Removed From The Clean Snapshot

- Unrelated EB-JEPA examples: audio, EEG, image JEPA, video JEPA, Gray-Scott,
  LTSF, pointcloud, intuitive physics, factors of variation.
- Local virtual environments and dependency caches.
- `.pytest_cache`, `__pycache__`, `.DS_Store`, `.pyc`.
- SSH material and personal credentials.
- Raw checkpoints and full Slurm run folders.
- Hackathon subject PDFs and temporary extraction folders.

## Known Non-Minimal Items Kept Deliberately

- `eb_jepa/datasets/two_rooms/`: retained because the original Track 7 target is
  Two Rooms and shared dataset utility imports still reference it.
- `examples/ac_video_jepa/track7_value/`: retained to document the direct Track 7
  learned-value implementation path.
- PDF/PowerPoint reports: retained because this repo is meant for judging, not
  only for source release.
