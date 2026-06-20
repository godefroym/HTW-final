# HackTheWorld(s) Track 7 — Dynamic Maze JEPA

This is the cleaned submission snapshot for our Track 7 work:

> Replace raw latent-distance planning costs with learned or task-correlated
> planning objectives for a JEPA world model.

Our main experiment is a dynamic maze variant: 21x21 mazes, stochastic doors,
local fog-of-war, and closed-loop replanning. The goal is to test whether a
world model can plan under changing topology, and whether learned/representation
costs beat brittle symbolic baselines when the visible map is incomplete.

## What Is Included

- `eb_jepa/`: minimal EB-JEPA package needed for the Track 7 experiments.
- `eb_jepa/datasets/dynamic_maze/`: stochastic-door fog-of-war maze.
- `examples/ac_video_jepa/dynamic_maze/`: evaluation, visualization, bootstrap
  value, BeliefLSTM value, and A* baseline scripts.
- `examples/ac_video_jepa/maze/main_subgoal.py`: A*-distilled subgoal head with
  optional visible-obstacle penalty.
- `examples/ac_video_jepa/track7_value/`: original Two Rooms Track 7 value-plan
  notes and launch scripts.
- `scripts/`: Slurm launchers used on Dalia.
- `outputs/dynamic_maze_solver_report_20260620/`: curated metrics, GIFs, PDF
  report, and PowerPoint deck with embedded animated GIFs.

Intentionally excluded: unrelated EB-JEPA examples, local virtualenvs, caches,
SSH keys, checkpoints, full Slurm run folders, and raw temporary files.

## Key Results

All rows below use 64 closed-loop evaluation episodes per solver.

| Method | Success | Mean steps | Blocked moves | Combined success/step ratio |
|---|---:|---:|---:|---:|
| Oracle A* replanning | 100.0% | 70.7 | 0.0 | 100.0% |
| Memory optimistic A* | 100.0% | 79.7 | 0.0 | 88.7% |
| JEPA subgoal N=2 | 57.8% | 280.8 | 103.3 | 14.6% |
| JEPA subgoal N=4 | 73.4% | 230.7 | 93.0 | 22.5% |
| Bootstrap learned value | 70.3% | 289.5 | 88.6 | 16.4% |
| Bootstrap probe-pos | 65.6% | 323.2 | 84.7 | 13.7% |
| Bootstrap repr-dist | 75.0% | 244.9 | 85.1 | 20.7% |
| BeliefLSTM value | 73.4% | 271.8 | 100.5 | 18.2% |

Interpretation: A*-free world-model objectives reach the same success band as
the A*-distilled JEPA N=4 controller, but they still pay extra steps and blocked
moves. The next queued experiment explicitly penalizes visible blocked cells in
the subgoal training loss.

## Main Artifacts

- Presentation: `outputs/dynamic_maze_solver_report_20260620/DYNAMIC_MAZE_SOLVER_PRESENTATION.pptx`
- Full report: `outputs/dynamic_maze_solver_report_20260620/DYNAMIC_MAZE_SOLVER_REPORT.pdf`
- Animated GIF evidence: `outputs/dynamic_maze_solver_report_20260620/gifs/`
- Machine-readable metrics: `outputs/dynamic_maze_solver_report_20260620/summary_metrics.json`

## Reproduce On Dalia

Set up the environment:

```bash
source env.sh
uv sync
```

Run the A*-distilled JEPA subgoal baseline:

```bash
bash scripts/htw_dynamic_maze_jepa_subgoal_slurm.sh
```

Run the A*-free bootstrap value / rollout objectives:

```bash
bash scripts/htw_dynamic_maze_bootstrap_slurm.sh
```

Run the BeliefLSTM value variant:

```bash
bash scripts/htw_dynamic_maze_belief_lstm_slurm.sh
```

Run the currently queued blocked-transition penalty variant:

```bash
HTW_DYN_SUBGOAL_BLOCK_PENALTY=5.0 \
HTW_DYN_SUBGOAL_BLOCK_RADIUS=1 \
bash scripts/htw_dynamic_maze_jepa_subgoal_blockpen_slurm.sh
```

Regenerate the report from available run outputs:

```bash
python scripts/build_dynamic_maze_solver_report.py
```

## Current Follow-Up

The blocked-transition penalty run was submitted on June 20, 2026:

- shared WM job: `76874`
- N=2 subgoal/eval: `76875`
- N=4 subgoal/eval: `76876`
- summary job: `76877`

Once it finishes, update `outputs/dynamic_maze_solver_report_20260620/` and the
result table above.
