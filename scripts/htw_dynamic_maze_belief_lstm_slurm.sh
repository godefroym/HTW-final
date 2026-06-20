#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_WM_CKPT="${EBJEPA_WORK:?source env.sh first}/runs/dynamic_maze_bootstrap_20260620_051011/00_bootstrap_wm_value/latest.pth.tar"
WM_CKPT="${HTW_BELIEF_WM_CKPT:-${1:-$DEFAULT_WM_CKPT}}"
RUN_DIR="${HTW_BELIEF_RUN_DIR:-${2:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_belief_lstm_$STAMP}}"
LOG_DIR="$RUN_DIR/slurm_logs"
BELIEF_DIR="$RUN_DIR/00_belief_value"

TIME_LIMIT="${HTW_BELIEF_TIME:-05:00:00}"
CPUS="${HTW_BELIEF_CPUS:-24}"
EPOCHS="${HTW_BELIEF_EPOCHS:-4}"
SIZE="${HTW_BELIEF_SIZE:-16384}"
BATCH="${HTW_BELIEF_BATCH:-128}"
CHUNK="${HTW_BELIEF_CHUNK:-2048}"
GEN_WORKERS="${HTW_BELIEF_GEN_WORKERS:-16}"
HIDDEN="${HTW_BELIEF_HIDDEN:-256}"
EVAL_EPISODES="${HTW_BELIEF_EVAL_EPISODES:-64}"
EVAL_LOOKAHEAD="${HTW_BELIEF_EVAL_LOOKAHEAD:-4}"
EVAL_MAX_STEPS="${HTW_BELIEF_EVAL_MAX_STEPS:-800}"
EVAL_GIFS="${HTW_BELIEF_EVAL_GIFS:-6}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$BELIEF_DIR"

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze Belief-LSTM Value

Created: $STAMP

Goal: add recurrent memory to the frozen bootstrap JEPA. The model trains a
\`BeliefLSTM\` over pooled JEPA latents and previous actions, then a
goal-conditioned \`BeliefValueHead\` for A*-free planning under fog-of-war.

Source WM checkpoint:

\`$WM_CKPT\`

Training:

- data policy: \`local_goal_frontier\`
- layout A* filter: disabled
- epochs: \`$EPOCHS\`
- samples/epoch: \`$SIZE\`
- batch: \`$BATCH\`
- belief hidden dim: \`$HIDDEN\`

Evaluation:

- objective: \`belief_value\`
- episodes: \`$EVAL_EPISODES\`
- lookahead: \`$EVAL_LOOKAHEAD\`
- budget: fixed \`$EVAL_MAX_STEPS\` moves

Checklist:

- [ ] Train BeliefLSTM + BeliefValueHead on frozen WM.
- [ ] Evaluate closed-loop belief planner.
- [ ] Compare against prior \`learned_value\` and \`repr_dist\`.

Status:
EOF

JOB_SCRIPT="$RUN_DIR/dynamic_maze_belief_lstm.sbatch"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=htw_dyn_belief"
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
WM_CKPT="$WM_CKPT"
BELIEF_DIR="$BELIEF_DIR"
EPOCHS="$EPOCHS"
SIZE="$SIZE"
BATCH="$BATCH"
CHUNK="$CHUNK"
GEN_WORKERS="$GEN_WORKERS"
HIDDEN="$HIDDEN"
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
test -f "$WM_CKPT"

WANDB_DISABLED=true uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.train_bootstrap_belief_value \
  --wm-ckpt "$WM_CKPT" \
  --out-dir "$BELIEF_DIR" \
  --epochs "$EPOCHS" \
  --size "$SIZE" \
  --batch-size "$BATCH" \
  --chunk-size "$CHUNK" \
  --num-gen-workers "$GEN_WORKERS" \
  --hidden-dim "$HIDDEN"

test -f "$BELIEF_DIR/latest.pth.tar"
note "Completed BeliefLSTM value training: $BELIEF_DIR/latest.pth.tar."

EV_DIR="$RUN_DIR/01_eval_belief_value"
mkdir -p "$EV_DIR"
uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.eval_bootstrap_belief_value \
  "$BELIEF_DIR/latest.pth.tar" \
  "$EV_DIR" \
  "$EVAL_EPISODES" \
  "$EVAL_LOOKAHEAD" \
  0.02 \
  "$EVAL_GIFS" \
  "$EVAL_MAX_STEPS"
note "Completed belief_value eval: $EV_DIR/belief_value_eval.json."

export RUN_DIR
uv run --no-sync --project "$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
path = run_dir / "01_eval_belief_value" / "belief_value_eval.json"
with path.open() as f:
    data = json.load(f)
with (run_dir / "metrics.tsv").open("w") as f:
    f.write("objective\tsuccess_rate\tmean_steps\tmean_blocked_moves\tmean_final_manhattan\n")
    f.write(
        f"belief_value\t{data['success_rate']:.6f}\t"
        f"{data['mean_steps']:.3f}\t{data['mean_blocked_moves']:.3f}\t"
        f"{data['mean_final_manhattan']:.3f}\n"
    )
PY

note "Completed all belief stages. Summary: $RUN_DIR/metrics.tsv."
EOF

chmod +x "$JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  echo "[dynamic-belief] dry run: $RUN_DIR"
  echo "[dynamic-belief] wm: $WM_CKPT"
  echo "[dynamic-belief] generated: $JOB_SCRIPT"
  exit 0
fi

JOB_ID="$(sbatch --parsable "$JOB_SCRIPT")"
{
  echo
  echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): Submitted Slurm job $JOB_ID."
} >> "$RUN_DIR/README.md"
echo "[dynamic-belief] run: $RUN_DIR"
echo "[dynamic-belief] wm: $WM_CKPT"
echo "[dynamic-belief] job: $JOB_ID"
echo "[dynamic-belief] monitor: squeue -j $JOB_ID"
