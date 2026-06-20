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
- `outputs/dynamic_maze_vision_poc_20260620/`: lightweight egocentric 3D-vision
  POC that renders `[up, down, left, right] x 128 x 128` grayscale observations
  without Chromium, Three.js, or OpenGL.
- `outputs/dynamic_maze_vision_jepa_20260620_121344/`: completed image-based
  JEPA run with metrics, logs, and normal/slow GIFs for the 3D-vision planner.

Intentionally excluded: unrelated EB-JEPA examples, local virtualenvs, caches,
SSH keys, checkpoints, full Slurm run folders, and raw temporary files.

## Key Results

Rows use 64 closed-loop evaluation episodes per solver, except the vision JEPA
run which uses 32 episodes.

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
| Vision JEPA repr-dist | 87.5% | 215.1 | 107.7 | 28.8% |

Interpretation: A*-free world-model objectives reach the same success band as
the A*-distilled JEPA N=4 controller, but they still pay extra steps and blocked
moves. The next queued experiment explicitly penalizes visible blocked cells in
the subgoal training loss.

## Main Artifacts

- Presentation: `outputs/dynamic_maze_solver_report_20260620/DYNAMIC_MAZE_SOLVER_PRESENTATION.pptx`
- Full report: `outputs/dynamic_maze_solver_report_20260620/DYNAMIC_MAZE_SOLVER_REPORT.pdf`
- Animated GIF evidence: `outputs/dynamic_maze_solver_report_20260620/gifs/`
- Machine-readable metrics: `outputs/dynamic_maze_solver_report_20260620/summary_metrics.json`
- 3D-vision POC GIF: `outputs/dynamic_maze_vision_poc_20260620/vision_episode.gif`
- 3D-vision JEPA run: `outputs/dynamic_maze_vision_jepa_20260620_121344/metrics.tsv`
- 3D-vision slowed GIFs: `outputs/dynamic_maze_vision_jepa_20260620_121344/01_eval_vision_reprdist/slow_2fps/`

## Reproducibility

The repo is self-contained for code, configs, metrics, reports, and GIF
evidence. Raw checkpoints and full Slurm folders are intentionally excluded to
keep the submission small; rerunning the launchers below recreates them under
`$EBJEPA_WORK/runs/...`.

### Verify Included Artifacts

This does not need a GPU. It checks that the committed metrics and slide GIFs
are present and prints the main result table.

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path(".")
summary = root / "outputs/dynamic_maze_solver_report_20260620/summary_metrics.json"
rows = json.loads(summary.read_text())
print("method\tsuccess\tsteps\tblocked\tepisodes")
for r in rows:
    print(
        f"{r['slug']}\t{100*r['success_rate']:.1f}%\t"
        f"{r['mean_steps']:.1f}\t{r['mean_blocked_moves']}\t{r['episodes']}"
    )

required = [
    "outputs/dynamic_maze_solver_report_20260620/DYNAMIC_MAZE_SOLVER_PRESENTATION.pptx",
    "outputs/dynamic_maze_solver_report_20260620/DYNAMIC_MAZE_SOLVER_REPORT.pdf",
    "outputs/dynamic_maze_vision_jepa_20260620_121344/metrics.tsv",
    "outputs/slides_assets/fog_overlay_2d/repr_dist_positive_fullmaze_fog_overlay.gif",
    "outputs/slides_assets/fog_overlay_2d/learned_value_positive_fullmaze_fog_overlay.gif",
    "outputs/slides_assets/fog_overlay_2d/probe_pos_positive_fullmaze_fog_overlay.gif",
    "outputs/slides_assets/fog_overlay_2d/belief_value_positive_fullmaze_fog_overlay.gif",
    "outputs/slides_assets/fog_overlay_2d/jepa_subgoal_n2_positive_fullmaze_fog_overlay.gif",
    "outputs/slides_assets/fog_overlay_2d/jepa_subgoal_n4_positive_fullmaze_fog_overlay.gif",
]
missing = [p for p in required if not (root / p).exists()]
if missing:
    raise SystemExit("missing artifacts:\n" + "\n".join(missing))
print("artifact check: ok")
PY
```

### Dalia Setup

Run from a clone in the Dalia work filesystem, not from home quota:

```bash
cd /lustre/work/<team>/<user>
git clone https://github.com/godefroym/HTW-final.git
cd HTW-final
source env.sh
module load python312 2>/dev/null || true
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi
uv sync
```

If the hackathon reservation is not active, disable it before submitting:

```bash
export HTW_SLURM_RESERVATION=""
```

All Slurm launchers support a dry run:

```bash
HTW_DRY_RUN=true bash scripts/htw_dynamic_maze_bootstrap_slurm.sh
```

### Full Dalia Rerun

This reproduces the experiment families used for the table. The datasets are
synthetic and generated on the fly, so there is no external dataset download.
The default scripts request one GPU where training is needed.

```bash
source env.sh
export WANDB_DISABLED=true
export REPRO_ROOT="${EBJEPA_WORK}/runs/htw_final_repro_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPRO_ROOT"

# 1. Strict symbolic baselines, 64 episodes to match the report table.
HTW_DYN_MAZE_EPISODES=64 \
bash scripts/htw_dynamic_maze_baseline_slurm.sh \
  "$REPRO_ROOT/00_baselines"

# 2. A*-distilled JEPA world model + subgoal heads N=2 and N=4.
bash scripts/htw_dynamic_maze_jepa_subgoal_slurm.sh \
  "$REPRO_ROOT/01_jepa_subgoal"

# 3. A*-free bootstrap JEPA + learned value / probe-pos / repr-dist evals.
bash scripts/htw_dynamic_maze_bootstrap_slurm.sh \
  "$REPRO_ROOT/02_bootstrap"

# 4. Image-based 2.5D egocentric vision JEPA.
bash scripts/htw_dynamic_maze_vision_slurm.sh \
  "$REPRO_ROOT/04_vision"

# 5. Optional follow-up: visible blocked-transition penalty for subgoals.
# This launcher creates its own Slurm dependencies internally.
HTW_DYN_SUBGOAL_BLOCK_PENALTY=5.0 \
HTW_DYN_SUBGOAL_BLOCK_RADIUS=1 \
bash scripts/htw_dynamic_maze_jepa_subgoal_blockpen_slurm.sh \
  "$REPRO_ROOT/05_subgoal_blockpen"
```

The BeliefLSTM run depends on the bootstrap checkpoint. Submit it only after
`$REPRO_ROOT/02_bootstrap/00_bootstrap_wm_value/latest.pth.tar` exists:

```bash
while [ ! -f "$REPRO_ROOT/02_bootstrap/00_bootstrap_wm_value/latest.pth.tar" ]; do
  squeue -u "$USER"
  sleep 300
done

bash scripts/htw_dynamic_maze_belief_lstm_slurm.sh \
  "$REPRO_ROOT/02_bootstrap/00_bootstrap_wm_value/latest.pth.tar" \
  "$REPRO_ROOT/03_belief_lstm"
```

Monitor jobs and logs:

```bash
squeue -u "$USER"
find "$REPRO_ROOT" -maxdepth 4 -path '*slurm_logs*' -type f
tail -f "$REPRO_ROOT"/02_bootstrap/slurm_logs/*.out
```

Expected metric files after completion:

```text
$REPRO_ROOT/00_baselines/eval/summary.tsv
$REPRO_ROOT/01_jepa_subgoal/metrics.tsv
$REPRO_ROOT/02_bootstrap/metrics.tsv
$REPRO_ROOT/03_belief_lstm/metrics.tsv
$REPRO_ROOT/04_vision/metrics.tsv
$REPRO_ROOT/05_subgoal_blockpen/metrics.tsv
```

A compact metric check:

```bash
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["REPRO_ROOT"])
for rel in [
    "00_baselines/eval/summary.tsv",
    "01_jepa_subgoal/metrics.tsv",
    "02_bootstrap/metrics.tsv",
    "03_belief_lstm/metrics.tsv",
    "04_vision/metrics.tsv",
]:
    path = root / rel
    print(f"\n## {rel}")
    print(path.read_text())
PY
```

Small stochastic differences are expected because doors toggle and fresh maze
layouts are sampled, but the success bands and ordering should be comparable to
the committed table.

### Regenerate Visual Evidence

Generate the lightweight egocentric vision POC without training:

```bash
uv run --no-sync --project "$PWD" python -m examples.ac_video_jepa.dynamic_maze.render_vision_poc \
  --out outputs/dynamic_maze_vision_poc_20260620 \
  --seed 7 \
  --image-size 128
```

Generate slide-ready full-maze/fog GIFs for the JEPA/WM algorithms from rerun
checkpoints:

```bash
uv run --no-sync --project "$PWD" python -m examples.ac_video_jepa.dynamic_maze.export_slide_fog_overlay_algorithms \
  --out outputs/slides_assets/fog_overlay_2d_algorithms \
  --bootstrap-ckpt "$REPRO_ROOT/02_bootstrap/00_bootstrap_wm_value/latest.pth.tar" \
  --belief-ckpt "$REPRO_ROOT/03_belief_lstm/00_belief_value/latest.pth.tar" \
  --subgoal-fine-ckpt "$REPRO_ROOT/01_jepa_subgoal/00_wm_astar/latest.pth.tar" \
  --subgoal-n2-ckpt "$REPRO_ROOT/01_jepa_subgoal/01_subgoal_N2/subgoal.pth.tar" \
  --subgoal-n4-ckpt "$REPRO_ROOT/01_jepa_subgoal/01_subgoal_N4/subgoal.pth.tar" \
  --search-episodes 8 \
  --max-steps 800 \
  --fps 2
```

The committed PDF/PPTX report is already included. To rebuild the report from
fresh raw run outputs, copy or symlink the new run folders into the names
referenced at the top of `scripts/build_dynamic_maze_solver_report.py`, then run:

```bash
uv run --no-sync --project "$PWD" python scripts/build_dynamic_maze_solver_report.py
```

For quick inspection without rebuilding the full report, use the `metrics.tsv`
files above and the GIFs under each run's evaluation directory.
