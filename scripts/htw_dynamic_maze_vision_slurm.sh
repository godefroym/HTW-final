#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_vision_jepa_$STAMP}"
LOG_DIR="$RUN_DIR/slurm_logs"
WM_DIR="$RUN_DIR/00_vision_wm"
EVAL_DIR="$RUN_DIR/01_eval_vision_reprdist"

TIME_LIMIT="${HTW_VISION_TIME:-06:00:00}"
CPUS="${HTW_VISION_CPUS:-24}"
EPOCHS="${HTW_VISION_EPOCHS:-3}"
SIZE="${HTW_VISION_SIZE:-4096}"
BATCH="${HTW_VISION_BATCH:-64}"
CHUNK="${HTW_VISION_CHUNK:-512}"
GEN_WORKERS="${HTW_VISION_GEN_WORKERS:-16}"
IMG_SIZE="${HTW_VISION_IMG_SIZE:-128}"
N_STEPS="${HTW_VISION_N_STEPS:-65}"
SAMPLE_LENGTH="${HTW_VISION_SAMPLE_LENGTH:-17}"
EVAL_EPISODES="${HTW_VISION_EVAL_EPISODES:-32}"
EVAL_LOOKAHEAD="${HTW_VISION_EVAL_LOOKAHEAD:-4}"
EVAL_GIFS="${HTW_VISION_EVAL_GIFS:-4}"
EVAL_MAX_STEPS="${HTW_VISION_EVAL_MAX_STEPS:-800}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$WM_DIR" "$EVAL_DIR"

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze Vision JEPA

Created: $STAMP

Goal: train a Dynamic Maze JEPA directly on lightweight egocentric rendered
images instead of symbolic 2D grid/fog channels.

Run directory:

\`$RUN_DIR\`

Training:

- config: \`examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_vision.yaml\`
- observation: \`[up, down, left, right] x $IMG_SIZE x $IMG_SIZE\`
- renderer: analytic 2.5D raycaster, no Chromium/Three.js/OpenGL
- teacher trajectories: \`oracle_replan\` for stable dynamics data
- epochs: \`$EPOCHS\`
- samples/epoch: \`$SIZE\`
- batch: \`$BATCH\`
- n_steps: \`$N_STEPS\`
- sample_length: \`$SAMPLE_LENGTH\`

Evaluation:

- objective: \`vision_repr_dist\`
- A* in action loop: no
- episodes: \`$EVAL_EPISODES\`
- lookahead: \`$EVAL_LOOKAHEAD\`
- budget: fixed \`$EVAL_MAX_STEPS\` moves

Checklist:

- [ ] Train image-based JEPA world model.
- [ ] Evaluate repr-dist planner on image latents.
- [ ] Export GIFs with top-down debug and four egocentric views.
- [ ] Write metrics summary.

Status:
EOF

JOB_SCRIPT="$RUN_DIR/dynamic_maze_vision.sbatch"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=htw_dyn_vision"
  echo "#SBATCH --partition=$partition"
  [ -n "$account" ] && echo "#SBATCH --account=$account"
  [ -n "$qos" ] && echo "#SBATCH --qos=$qos"
  [ -n "$reservation" ] && echo "#SBATCH --reservation=$reservation"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --ntasks=1"
  echo "#SBATCH --cpus-per-task=$CPUS"
  echo "#SBATCH --gres=gpu:1"
  echo "#SBATCH --time=$TIME_LIMIT"
  echo "#SBATCH --output=$LOG_DIR/%x-%j.out"
  echo "#SBATCH --error=$LOG_DIR/%x-%j.err"
  echo
  echo "set -euo pipefail"
} > "$JOB_SCRIPT"

cat >> "$JOB_SCRIPT" <<EOF
ROOT="$ROOT"
RUN_DIR="$RUN_DIR"
WM_DIR="$WM_DIR"
EVAL_DIR="$EVAL_DIR"
EPOCHS="$EPOCHS"
SIZE="$SIZE"
BATCH="$BATCH"
CHUNK="$CHUNK"
GEN_WORKERS="$GEN_WORKERS"
IMG_SIZE="$IMG_SIZE"
N_STEPS="$N_STEPS"
SAMPLE_LENGTH="$SAMPLE_LENGTH"
EVAL_EPISODES="$EVAL_EPISODES"
EVAL_LOOKAHEAD="$EVAL_LOOKAHEAD"
EVAL_GIFS="$EVAL_GIFS"
EVAL_MAX_STEPS="$EVAL_MAX_STEPS"
EOF

cat >> "$JOB_SCRIPT" <<'EOF'
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
module load python312 2>/dev/null || true
if ! "$UV_INSTALL_DIR/uv" --version >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
fi
export PATH="$UV_INSTALL_DIR:$PATH"
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

note() {
  {
    echo
    echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): $1"
  } >> "$RUN_DIR/README.md"
}

note "Started on $(hostname), job ${SLURM_JOB_ID:-unknown}."

WANDB_DISABLED=true uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.main \
  --fname=examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_vision.yaml \
  --meta.model_folder="$WM_DIR" \
  --meta.load_model=False \
  --meta.enable_plan_eval=False \
  --logging.log_wandb=False \
  --logging.tqdm_silent=True \
  --optim.epochs="$EPOCHS" \
  --data.size="$SIZE" \
  --data.batch_size="$BATCH" \
  --data.img_size="$IMG_SIZE" \
  --data.n_steps="$N_STEPS" \
  --data.sample_length="$SAMPLE_LENGTH" \
  --data.pipeline.chunk_size="$CHUNK" \
  --data.pipeline.num_gen_workers="$GEN_WORKERS"

test -f "$WM_DIR/latest.pth.tar"
note "Completed vision JEPA WM: $WM_DIR/latest.pth.tar."

uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.eval_vision_reprdist \
  "$WM_DIR/latest.pth.tar" \
  "$EVAL_DIR" \
  "$EVAL_EPISODES" \
  "$EVAL_LOOKAHEAD" \
  "$EVAL_GIFS" \
  "$EVAL_MAX_STEPS"

test -f "$EVAL_DIR/vision_reprdist_eval.json"
note "Completed vision repr-dist eval: $EVAL_DIR/vision_reprdist_eval.json."

export RUN_DIR EVAL_DIR
uv run --no-sync --project "$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
eval_dir = Path(os.environ["EVAL_DIR"])
with (eval_dir / "vision_reprdist_eval.json").open() as f:
    data = json.load(f)
with (run_dir / "metrics.tsv").open("w") as f:
    f.write("objective\tsuccess_rate\tmean_steps\tmean_blocked_moves\tmean_final_manhattan\timage_size\tlookahead\n")
    f.write(
        f"{data['objective']}\t{data['success_rate']:.6f}\t"
        f"{data['mean_steps']:.3f}\t{data['mean_blocked_moves']:.3f}\t"
        f"{data['mean_final_manhattan']:.3f}\t{data['image_size']}\t{data['lookahead']}\n"
    )
PY

note "Completed all vision stages. Summary: $RUN_DIR/metrics.tsv."
EOF

chmod +x "$JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  echo "[dynamic-vision] dry run: $RUN_DIR"
  echo "[dynamic-vision] generated: $JOB_SCRIPT"
  exit 0
fi

JOB_ID="$(sbatch --parsable "$JOB_SCRIPT")"
{
  echo
  echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): Submitted Slurm job $JOB_ID."
} >> "$RUN_DIR/README.md"
echo "[dynamic-vision] run: $RUN_DIR"
echo "[dynamic-vision] job: $JOB_ID"
echo "[dynamic-vision] monitor: squeue -j $JOB_ID"
