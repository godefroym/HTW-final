# HTW Dynamic Maze JEPA + Distilled A* Subgoals

Created: 20260620_034609

Goal: train a dynamic-maze JEPA on oracle A* replanning trajectories, then train
two A*-distilled subgoal heads (`N=2`, `N=4`) and evaluate them without A* in
the action loop.

Run directory:

`$EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609`

WM:

- config: `examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_astar.yaml`
- epochs: `3`
- samples/epoch: `8192`
- batch: `64`
- n_steps: `65`
- sample_length: `17`

Subgoal:

- N values: `2 4`
- epochs: `4`
- samples/epoch: `8192`

Evaluation:

- episodes: `64`
- lookahead: `auto`
- budget: `6 x A*_initial + 20`

Checklist:

- [ ] Train JEPA world model.
- [ ] Train subgoal N=2.
- [ ] Evaluate subgoal N=2.
- [ ] Train subgoal N=4.
- [ ] Evaluate subgoal N=4.
- [ ] Compare against dynamic-maze A* baselines.

Status:

- 2026-06-20T03:46:09+02:00: Submitted Slurm job 75528.

- 2026-06-20T03:46:10+02:00: Started on compute-node, job 75528.

- 2026-06-20T04:00:33+02:00: Completed dynamic JEPA WM: $EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609/00_wm_astar/latest.pth.tar.

- 2026-06-20T04:19:28+02:00: Completed subgoal N=2: $EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609/01_subgoal_N2/subgoal.pth.tar.

- 2026-06-20T04:24:46+02:00: Completed eval N=2: $EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609/02_eval_N2/subgoal_eval.json.

- 2026-06-20T04:44:08+02:00: Completed subgoal N=4: $EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609/01_subgoal_N4/subgoal.pth.tar.

- 2026-06-20T04:51:54+02:00: Completed eval N=4: $EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609/02_eval_N4/subgoal_eval.json.

Metrics:

| N | success | efficiency | steps | blocked |
|---:|---:|---:|---:|---:|
| 2 | 57.8% | 0.403 | 280.8 | 103.3 |
| 4 | 73.4% | 0.494 | 230.7 | 93.0 |

- 2026-06-20T04:51:54+02:00: Completed all dynamic JEPA/subgoal stages. Summary: $EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609/metrics.tsv.
