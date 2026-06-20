#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
FOG_RADIUS="${HTW_REPR_FOG_RADIUS:-8}"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_repr_fog${FOG_RADIUS}_$STAMP}"
LOG_DIR="$RUN_DIR/slurm_logs"
WM_DIR="$RUN_DIR/00_wm_repr_fog${FOG_RADIUS}"
EVAL_DIR="$RUN_DIR/01_eval_repr_dist"

TIME_LIMIT="${HTW_REPR_FOG_TIME:-08:00:00}"
CPUS="${HTW_REPR_FOG_CPUS:-24}"
EPOCHS="${HTW_REPR_FOG_EPOCHS:-5}"
SIZE="${HTW_REPR_FOG_SIZE:-16384}"
BATCH="${HTW_REPR_FOG_BATCH:-128}"
CHUNK="${HTW_REPR_FOG_CHUNK:-2048}"
GEN_WORKERS="${HTW_REPR_FOG_GEN_WORKERS:-16}"
EVAL_EPISODES="${HTW_REPR_FOG_EVAL_EPISODES:-64}"
EVAL_LOOKAHEAD="${HTW_REPR_FOG_EVAL_LOOKAHEAD:-4}"
EVAL_MAX_STEPS="${HTW_REPR_FOG_EVAL_MAX_STEPS:-800}"
EVAL_GIFS="${HTW_REPR_FOG_EVAL_GIFS:-6}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$WM_DIR" "$EVAL_DIR"

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze Repr-Dist, Wider Fog

Created: $STAMP

Goal: train the classic grid-observation Dynamic Maze JEPA with a wider
fog-of-war radius, then evaluate the A*-free \`repr_dist\` planner.

Run directory:

\`$RUN_DIR\`

Training:

- base config: \`examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_bootstrap.yaml\`
- observation mode: classic grid/fog channels
- fog radius: \`$FOG_RADIUS\`
- policy: \`local_goal_frontier\`
- epochs: \`$EPOCHS\`
- samples/epoch: \`$SIZE\`
- batch: \`$BATCH\`

Evaluation:

- objective: \`repr_dist\`
- episodes: \`$EVAL_EPISODES\`
- lookahead: \`$EVAL_LOOKAHEAD\`
- budget: fixed \`$EVAL_MAX_STEPS\` moves

Checklist:

- [ ] Train grid JEPA with wider fog.
- [ ] Evaluate \`repr_dist\`.
- [ ] Write metrics summary.

Status:
EOF

JOB_SCRIPT="$RUN_DIR/dynamic_maze_repr_fog${FOG_RADIUS}.sbatch"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=htw_repr_fog${FOG_RADIUS}"
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
FOG_RADIUS="$FOG_RADIUS"
EPOCHS="$EPOCHS"
SIZE="$SIZE"
BATCH="$BATCH"
CHUNK="$CHUNK"
GEN_WORKERS="$GEN_WORKERS"
EVAL_EPISODES="$EVAL_EPISODES"
EVAL_LOOKAHEAD="$EVAL_LOOKAHEAD"
EVAL_MAX_STEPS="$EVAL_MAX_STEPS"
EVAL_GIFS="$EVAL_GIFS"
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
  --fname=examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_bootstrap.yaml \
  --meta.model_folder="$WM_DIR" \
  --meta.load_model=False \
  --meta.enable_plan_eval=False \
  --logging.log_wandb=False \
  --logging.tqdm_silent=True \
  --optim.epochs="$EPOCHS" \
  --data.size="$SIZE" \
  --data.batch_size="$BATCH" \
  --data.fog_radius="$FOG_RADIUS" \
  --data.pipeline.chunk_size="$CHUNK" \
  --data.pipeline.num_gen_workers="$GEN_WORKERS"

test -f "$WM_DIR/latest.pth.tar"
note "Completed wider-fog WM: $WM_DIR/latest.pth.tar."

uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.eval_bootstrap_value \
  "$WM_DIR/latest.pth.tar" \
  "$EVAL_DIR" \
  repr_dist \
  "$EVAL_EPISODES" \
  "$EVAL_LOOKAHEAD" \
  0.02 \
  "$EVAL_GIFS" \
  "$EVAL_MAX_STEPS"

test -f "$EVAL_DIR/repr_dist_eval.json"
note "Completed repr_dist eval: $EVAL_DIR/repr_dist_eval.json."

export RUN_DIR EVAL_DIR FOG_RADIUS
uv run --no-sync --project "$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
eval_dir = Path(os.environ["EVAL_DIR"])
with (eval_dir / "repr_dist_eval.json").open() as f:
    data = json.load(f)
with (run_dir / "metrics.tsv").open("w") as f:
    f.write("objective\tfog_radius\tsuccess_rate\tmean_steps\tmean_blocked_moves\tmean_final_manhattan\tlookahead\n")
    f.write(
        f"repr_dist\t{os.environ['FOG_RADIUS']}\t{data['success_rate']:.6f}\t"
        f"{data['mean_steps']:.3f}\t{data['mean_blocked_moves']:.3f}\t"
        f"{data['mean_final_manhattan']:.3f}\t{data['lookahead']}\n"
    )
PY

note "Completed all wider-fog repr stages. Summary: $RUN_DIR/metrics.tsv."
EOF

chmod +x "$JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  echo "[dynamic-repr-fog] dry run: $RUN_DIR"
  echo "[dynamic-repr-fog] generated: $JOB_SCRIPT"
  exit 0
fi

JOB_ID="$(sbatch --parsable "$JOB_SCRIPT")"
{
  echo
  echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): Submitted Slurm job $JOB_ID."
} >> "$RUN_DIR/README.md"
echo "[dynamic-repr-fog] run: $RUN_DIR"
echo "[dynamic-repr-fog] job: $JOB_ID"
echo "[dynamic-repr-fog] monitor: squeue -j $JOB_ID"
