#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_bootstrap_$STAMP}"
LOG_DIR="$RUN_DIR/slurm_logs"
WM_DIR="$RUN_DIR/00_bootstrap_wm_value"

TIME_LIMIT="${HTW_BOOT_TIME:-08:00:00}"
CPUS="${HTW_BOOT_CPUS:-24}"
EPOCHS="${HTW_BOOT_EPOCHS:-5}"
SIZE="${HTW_BOOT_SIZE:-16384}"
BATCH="${HTW_BOOT_BATCH:-128}"
CHUNK="${HTW_BOOT_CHUNK:-2048}"
GEN_WORKERS="${HTW_BOOT_GEN_WORKERS:-16}"
EVAL_EPISODES="${HTW_BOOT_EVAL_EPISODES:-64}"
EVAL_LOOKAHEAD="${HTW_BOOT_EVAL_LOOKAHEAD:-4}"
EVAL_MAX_STEPS="${HTW_BOOT_EVAL_MAX_STEPS:-800}"
EVAL_GIFS="${HTW_BOOT_EVAL_GIFS:-6}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$WM_DIR"

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze Bootstrap WM

Created: $STAMP

Goal: train a dynamic-maze world model without A* trajectory labels, using local
exploration and hindsight TD value learning.

Run directory:

\`$RUN_DIR\`

Training:

- config: \`examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_bootstrap.yaml\`
- policy: \`local_goal_frontier\`
- layout A* filter: disabled
- epochs: \`$EPOCHS\`
- samples/epoch: \`$SIZE\`
- batch: \`$BATCH\`

Evaluation:

- A*-free objectives: \`learned_value\`, \`probe_pos\`, \`repr_dist\`
- episodes/objective: \`$EVAL_EPISODES\`
- lookahead: \`$EVAL_LOOKAHEAD\`
- budget: fixed \`$EVAL_MAX_STEPS\` moves

Checklist:

- [ ] Train bootstrap WM + hindsight value head.
- [ ] Evaluate learned value planner.
- [ ] Evaluate probe-position planner.
- [ ] Evaluate representation-distance planner.
- [ ] Compare to A*-distilled N=2/N=4 and strict baselines.

Status:
EOF

JOB_SCRIPT="$RUN_DIR/dynamic_maze_bootstrap.sbatch"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=htw_dyn_boot"
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
  --data.pipeline.chunk_size="$CHUNK" \
  --data.pipeline.num_gen_workers="$GEN_WORKERS"

test -f "$WM_DIR/latest.pth.tar"
note "Completed bootstrap WM/value: $WM_DIR/latest.pth.tar."

for objective in learned_value probe_pos repr_dist; do
  EV_DIR="$RUN_DIR/01_eval_${objective}"
  mkdir -p "$EV_DIR"
  uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.eval_bootstrap_value \
    "$WM_DIR/latest.pth.tar" \
    "$EV_DIR" \
    "$objective" \
    "$EVAL_EPISODES" \
    "$EVAL_LOOKAHEAD" \
    0.02 \
    "$EVAL_GIFS" \
    "$EVAL_MAX_STEPS"
  note "Completed eval ${objective}: $EV_DIR/${objective}_eval.json."
done

export RUN_DIR
uv run --no-sync --project "$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
objectives = ["learned_value", "probe_pos", "repr_dist"]
with (run_dir / "metrics.tsv").open("w") as f:
    f.write("objective\tsuccess_rate\tmean_steps\tmean_blocked_moves\tmean_final_manhattan\n")
    for objective in objectives:
        path = run_dir / f"01_eval_{objective}" / f"{objective}_eval.json"
        with path.open() as jf:
            data = json.load(jf)
        f.write(
            f"{objective}\t{data['success_rate']:.6f}\t"
            f"{data['mean_steps']:.3f}\t{data['mean_blocked_moves']:.3f}\t"
            f"{data['mean_final_manhattan']:.3f}\n"
        )
PY

note "Completed all bootstrap stages. Summary: $RUN_DIR/metrics.tsv."
EOF

chmod +x "$JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  echo "[dynamic-bootstrap] dry run: $RUN_DIR"
  echo "[dynamic-bootstrap] generated: $JOB_SCRIPT"
  exit 0
fi

JOB_ID="$(sbatch --parsable "$JOB_SCRIPT")"
{
  echo
  echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): Submitted Slurm job $JOB_ID."
} >> "$RUN_DIR/README.md"
echo "[dynamic-bootstrap] run: $RUN_DIR"
echo "[dynamic-bootstrap] job: $JOB_ID"
echo "[dynamic-bootstrap] monitor: squeue -j $JOB_ID"
