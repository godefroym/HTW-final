# Dynamic Maze Pivot

This is a controlled extension of the static Maze benchmark:

- random DFS maze;
- a small set of wall segments become stochastic doors;
- each door toggles open/closed with probability `p` at every step;
- the agent observes only a local fog-of-war window;
- A* baselines replan myopically on either the full current grid or the
  observed grid.

The goal is not to claim that A* is obsolete. The goal is to create a setting
where static shortest-path distance is the wrong objective, so a world model can
be useful because it predicts the future state distribution of the world.

## Observation Channels

1. agent dot;
2. observed walls;
3. unknown/fog mask;
4. observed dynamic-door mask.

Closed doors appear as walls in channel 2 and as doors in channel 4. Open doors
appear as free cells in channel 2 and as doors in channel 4.

## Baselines

- `oracle_replan`: A* on the current full dynamic grid. Strong but not
  stochastic-optimal because it ignores future toggles.
- `fog_optimistic`: A* on the visible grid, treating unknown cells as free.
- `fog_conservative`: A* on the visible grid, treating unknown cells as walls.
- `memory_optimistic`: A* on the accumulated visible grid, treating never-seen
  unknown cells as free.
- `memory_conservative`: A* on the accumulated visible grid, treating never-seen
  unknown cells as walls.
- `random`: random cardinal actions.

## Checklist

- [x] Add dynamic maze environment and dataset.
- [x] Add fog-of-war observation.
- [x] Add stochastic door toggles.
- [x] Add A* teacher policies for dataset/loss baselines.
- [x] Add baseline eval and GIF export.
- [x] Train first JEPA world model on `oracle_replan` trajectories.
- [x] Train distilled A* subgoal heads for `N=2` and `N=4`.
- [x] Add no-fog gate visualizations.
- [x] Add no-A* exploration policy and bootstrap WM/value config.
- [x] Add A*-free bootstrap value evaluator.
- [x] Run first no-A* bootstrap WM/value experiment.
- [x] Add BeliefLSTM memory value variant.
- [x] Run and evaluate BeliefLSTM memory value variant.
- [x] Add lightweight 3D-vision POC renderer for egocentric WM observations.
- [ ] Run sweeps over `door_toggle_prob`, `num_doors`, and `fog_radius`.
- [ ] Improve learned value so it beats `repr_dist` under the same budget.
- [ ] Add uncertainty/Gaussian regularization experiment.

## Quick Eval

```bash
python -m examples.ac_video_jepa.dynamic_maze.eval_astar_baselines \
  --out /tmp/dynamic_maze_baseline \
  --episodes 32 \
  --num-doors 8 \
  --door-toggle-prob 0.04 \
  --fog-radius 4
```

## First WM Config

`examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_astar.yaml`
uses A* replanning trajectories as the action-conditioned dynamics data. This is
a baseline, not the final Track 7 claim.

## JEPA + Distilled Subgoals

```bash
bash scripts/htw_dynamic_maze_jepa_subgoal_slurm.sh
```

This trains one dynamic-maze JEPA and then two A*-distilled subgoal heads:

- `N=2`: short-horizon waypoint distillation;
- `N=4`: slightly more global waypoint distillation.

The eval loop does not query A* for actions. A* is only used to set the initial
episode budget and to report an efficiency score.

## Bootstrap WM Without A*

The more world-model-faithful path is:

1. collect trajectories from local exploration, not from A*;
2. train the JEPA dynamics model on those action-conditioned observations;
3. train `GoalValueHead(z, z_goal)` with hindsight TD on future states observed
   in the same trajectory window;
4. evaluate by rolling the WM for each candidate action and choosing the action
   with the highest learned value.

Config:

```text
examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_bootstrap.yaml
```

Launcher:

```bash
bash scripts/htw_dynamic_maze_bootstrap_slurm.sh
```

This setup disables A* layout filtering and uses:

```yaml
teacher_policy: local_goal_frontier
layout_use_astar_filter: false
value_mode: multi_horizon_td
```

The exploration policy is deliberately myopic: it scores one-step moves by local
fog novelty, visit penalty and a weak Manhattan goal bias. It never calls A* or
any graph solver.

The evaluator:

```bash
python -m examples.ac_video_jepa.dynamic_maze.eval_bootstrap_value \
  <ckpt> <out_dir> learned_value 64 4
```

uses a fixed move budget by default, so the control loop and stopping budget are
both A*-free. It can be compared against `probe_pos` and `repr_dist` under the
same world-model rollouts.

First bootstrap run:

```text
$EBJEPA_WORK/runs/dynamic_maze_bootstrap_20260620_051011
outputs/dynamic_maze_bootstrap_20260620_051011/
```

The run completed on Slurm job `75812` in `01:32:25` on one B200. Evaluation
uses 64 episodes, lookahead 4, and a fixed 800-move budget.

| objective | success | mean steps | blocked moves | final manhattan |
|---|---:|---:|---:|---:|
| `learned_value` | 70.3% | 289.5 | 88.6 | 7.0 |
| `probe_pos` | 65.6% | 323.2 | 84.7 | 6.8 |
| `repr_dist` | 75.0% | 244.9 | 85.1 | 4.1 |

Sequential oracle A* on the same unfiltered bootstrap distribution reaches
100.0% success in 67.5 mean steps. Against that reference, the current
`learned_value` reaches 70.3% of the ideal success rate and about 23.3% of the
ideal step efficiency. The best current control objective is still `repr_dist`
at 75.0% success and about 27.6% step efficiency, so the value head needs tuning
before it supports the Track 7 claim that learned cost beats latent distance.

## Bootstrap WM With BeliefLSTM Memory

The fog-of-war setting is partially observable, so a memory-less latent value
can confuse unexplored cells, previously seen corridors, and recently blocked
doors. The BeliefLSTM variant keeps the frozen bootstrap JEPA and trains only:

```text
pooled(z_t), action_{t-1} -> BeliefLSTM -> h_t
V(h_t, z_goal) -> scalar value
```

Trainer:

```bash
python -m examples.ac_video_jepa.dynamic_maze.train_bootstrap_belief_value \
  --wm-ckpt <bootstrap_wm/latest.pth.tar> \
  --out-dir <belief_value_dir>
```

Evaluator:

```bash
python -m examples.ac_video_jepa.dynamic_maze.eval_bootstrap_belief_value \
  <belief_value_dir/latest.pth.tar> <eval_dir> 64 4
```

Launcher:

```bash
bash scripts/htw_dynamic_maze_belief_lstm_slurm.sh
```

This keeps the comparison tight: same A*-free data policy, same frozen JEPA, but
the learned value receives recurrent episode memory. The target to beat is the
first bootstrap `repr_dist` score: 75.0% success and 244.9 mean steps.

First BeliefLSTM run:

```text
$EBJEPA_WORK/runs/dynamic_maze_belief_lstm_20260620_085948
outputs/dynamic_maze_belief_lstm_20260620_085948/
```

The run completed on Slurm job `76388` in `01:12:56` on one B200. It trains the
memory/value module for 4 epochs on the frozen bootstrap WM.

| objective | success | mean steps | blocked moves | final manhattan |
|---|---:|---:|---:|---:|
| `learned_value` | 70.3% | 289.5 | 88.6 | 7.0 |
| `belief_value` | 73.4% | 271.8 | 100.5 | 6.8 |
| `repr_dist` | 75.0% | 244.9 | 85.1 | 4.1 |

The memory variant improves over the first learned value head, but it still does
not beat `repr_dist`. It is nevertheless closer to the Track 7 story because it
uses an explicit recurrent belief state under partial observability.

## Lightweight 3D-Vision POC

The POC renderer replaces the symbolic 2D grid observation with four
egocentric grayscale views:

```text
[up, down, left, right] x 128 x 128
```

It uses an analytic 2.5D raycaster over the dynamic grid, so it does not depend
on Three.js, Chromium, OpenGL, or EGL. At dataset time, the current occupancy
grid and stochastic door states are rendered on the fly from the agent cell.

```bash
python -m examples.ac_video_jepa.dynamic_maze.render_vision_poc \
  --out outputs/dynamic_maze_vision_poc_20260620 \
  --seed 7 \
  --image-size 128
```

The output contains:

- `topdown_start.png`: debug top-down map;
- `four_views_start.png`: the initial 4-channel egocentric observation;
- `four_views_start.npy`: raw `[4, 128, 128]` uint8 tensor;
- `demo_frame_start.png`, `demo_frame_mid.png`: combined debug frames;
- `vision_episode.gif`: oracle-replan episode rendered with perceptual views.

The full image-based training run uses the same renderer through
`data.observation_mode=vision`:

```bash
bash scripts/htw_dynamic_maze_vision_slurm.sh
```

This trains a JEPA directly on `[up, down, left, right] x 128 x 128` rendered
frames, then evaluates an A*-free `vision_repr_dist` planner with GIF export.

## Current Starter Run

Run directory on Dalia:

```text
$EBJEPA_WORK/runs/dynamic_maze_jepa_subgoal_starter_20260620_034609
```

Local copied outputs:

```text
outputs/dynamic_maze_jepa_subgoal_starter_20260620_034609/
outputs/dynamic_maze_visualizations/
```

Training budget:

- WM: `3 x 8192 = 24576` sampled trajectories.
- Subgoal `N=2`: `4 x 8192 = 32768` sampled trajectories.
- Subgoal `N=4`: `4 x 8192 = 32768` sampled trajectories.
- Total training samples consumed: `90112`.

The first starter run used the CPU stream path before per-worker dynamic-maze
RNG reseeding was fixed. With `chunk_size=512` and `num_gen_workers=16`, the
effective unique layout count in consumed chunks is approximately:

- WM: `48 chunks x 32 unique layouts = 1536`.
- Subgoal `N=2`: `64 chunks x 32 = 2048`.
- Subgoal `N=4`: `64 chunks x 32 = 2048`.
- Total effective unique training layouts: about `5632`.

This is fixed for future runs in `eb_jepa/datasets/precomputed.py`: each CPU
stream worker part now resets the dataset RNG from its assigned chunk seed.

Evaluation budget:

- JEPA subgoal `N=2`: 64 closed-loop episodes.
- JEPA subgoal `N=4`: 64 closed-loop episodes.
- The two JEPA evals reuse the same deterministic 64-layout sequence, so this is
  128 rollouts over 64 unique eval layouts.
- Strict A* baseline eval: 6 policies x 64 rollouts, sharing 64 unique layouts
  across policies.
- Bootstrap WM eval: 3 objectives x 64 rollouts, using the sequential dynamic
  maze RNG from the bootstrap config.
