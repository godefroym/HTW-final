# HTW Dynamic Maze Vision JEPA

Created: 20260620_121344

Goal: train a Dynamic Maze JEPA directly on lightweight egocentric rendered
images instead of symbolic 2D grid/fog channels.

Run directory:

`$EBJEPA_WORK/runs/dynamic_maze_vision_jepa_20260620_121344`

Training:

- config: `examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_vision.yaml`
- observation: `[up, down, left, right] x 128 x 128`
- renderer: analytic 2.5D raycaster, no Chromium/Three.js/OpenGL
- teacher trajectories: `oracle_replan` for stable dynamics data
- epochs: `3`
- samples/epoch: `4096`
- batch: `64`
- n_steps: `65`
- sample_length: `17`

Evaluation:

- objective: `vision_repr_dist`
- A* in action loop: no
- episodes: `32`
- lookahead: `4`
- budget: fixed `800` moves

Checklist:

- [x] Train image-based JEPA world model.
- [x] Evaluate repr-dist planner on image latents.
- [x] Export GIFs with top-down debug and four egocentric views.
- [x] Write metrics summary.

Results:

- success: `87.5%`
- mean steps: `215.125`
- mean blocked moves: `107.688`
- episodes: `32`
- normal GIFs: `01_eval_vision_reprdist/ep*_vision_reprdist_succ.gif`
- slow GIFs: `01_eval_vision_reprdist/slow_3fps/`, `slow_2fps/`, `slow_1_5fps/`

Status:

- 2026-06-20T12:13:44+02:00: Submitted Slurm job 77323.

- 2026-06-20T12:13:45+02:00: Started on dalianvl13, job 77323.

- 2026-06-20T12:19:59+02:00: Completed vision JEPA WM: `$EBJEPA_WORK/runs/dynamic_maze_vision_jepa_20260620_121344/00_vision_wm/latest.pth.tar`.

- 2026-06-20T12:22:06+02:00: Completed vision repr-dist eval: `$EBJEPA_WORK/runs/dynamic_maze_vision_jepa_20260620_121344/01_eval_vision_reprdist/vision_reprdist_eval.json`.

- 2026-06-20T12:22:06+02:00: Completed all vision stages. Summary: `$EBJEPA_WORK/runs/dynamic_maze_vision_jepa_20260620_121344/metrics.tsv`.
