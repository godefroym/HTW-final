#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_jepa_subgoal_$STAMP}"
LOG_DIR="$RUN_DIR/slurm_logs"
WM_DIR="$RUN_DIR/00_wm_astar"

TIME_LIMIT="${HTW_DYN_JEPA_TIME:-08:00:00}"
CPUS="${HTW_DYN_JEPA_CPUS:-24}"
WM_EPOCHS="${HTW_DYN_WM_EPOCHS:-3}"
WM_SIZE="${HTW_DYN_WM_SIZE:-8192}"
WM_BATCH="${HTW_DYN_WM_BATCH:-64}"
WM_CHUNK="${HTW_DYN_WM_CHUNK:-512}"
WM_GEN_WORKERS="${HTW_DYN_WM_GEN_WORKERS:-16}"
N_STEPS="${HTW_DYN_N_STEPS:-65}"
SAMPLE_LENGTH="${HTW_DYN_SAMPLE_LENGTH:-17}"
MIN_PATH_LENGTH="${HTW_DYN_MIN_PATH_LENGTH:-50}"
SUBGOAL_N_VALUES="${HTW_DYN_SUBGOAL_N_VALUES:-2 4}"
SUBGOAL_EPOCHS="${HTW_DYN_SUBGOAL_EPOCHS:-4}"
SUBGOAL_BATCH="${HTW_DYN_SUBGOAL_BATCH:-64}"
SUBGOAL_SIZE="${HTW_DYN_SUBGOAL_SIZE:-8192}"
SUBGOAL_WORKERS="${HTW_DYN_SUBGOAL_WORKERS:-8}"
EVAL_EPISODES="${HTW_DYN_SUBGOAL_EVAL_EPISODES:-64}"
EVAL_REVISIT_PEN="${HTW_DYN_SUBGOAL_REVISIT_PEN:-0.02}"
EVAL_GIFS="${HTW_DYN_SUBGOAL_GIFS:-6}"
EVAL_BUDGET_FACTOR="${HTW_DYN_SUBGOAL_BUDGET_FACTOR:-6}"
EVAL_BUDGET_MARGIN="${HTW_DYN_SUBGOAL_BUDGET_MARGIN:-20}"
EVAL_LOOKAHEAD="${HTW_DYN_SUBGOAL_LOOKAHEAD:-auto}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$WM_DIR"

now_iso() {
  date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"
}

append_status() {
  local readme="$1"
  local message="$2"
  {
    echo
    echo "- $(now_iso): $message"
  } >> "$readme"
}

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze JEPA + Distilled A* Subgoals

Created: $STAMP

Goal: train a dynamic-maze JEPA on oracle A* replanning trajectories, then train
two A*-distilled subgoal heads (\`N=2\`, \`N=4\`) and evaluate them without A* in
the action loop.

Run directory:

\`$RUN_DIR\`

WM:

- config: \`examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_astar.yaml\`
- epochs: \`$WM_EPOCHS\`
- samples/epoch: \`$WM_SIZE\`
- batch: \`$WM_BATCH\`
- n_steps: \`$N_STEPS\`
- sample_length: \`$SAMPLE_LENGTH\`

Subgoal:

- N values: \`$SUBGOAL_N_VALUES\`
- epochs: \`$SUBGOAL_EPOCHS\`
- samples/epoch: \`$SUBGOAL_SIZE\`

Evaluation:

- episodes: \`$EVAL_EPISODES\`
- lookahead: \`$EVAL_LOOKAHEAD\`
- budget: \`$EVAL_BUDGET_FACTOR x A*_initial + $EVAL_BUDGET_MARGIN\`

Checklist:

- [ ] Train JEPA world model.
- [ ] Train subgoal N=2.
- [ ] Evaluate subgoal N=2.
- [ ] Train subgoal N=4.
- [ ] Evaluate subgoal N=4.
- [ ] Compare against dynamic-maze A* baselines.

Status:
EOF

JOB_SCRIPT="$RUN_DIR/dynamic_maze_jepa_subgoal.sbatch"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=htw_dyn_jepa"
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
WM_EPOCHS="$WM_EPOCHS"
WM_SIZE="$WM_SIZE"
WM_BATCH="$WM_BATCH"
WM_CHUNK="$WM_CHUNK"
WM_GEN_WORKERS="$WM_GEN_WORKERS"
N_STEPS="$N_STEPS"
SAMPLE_LENGTH="$SAMPLE_LENGTH"
MIN_PATH_LENGTH="$MIN_PATH_LENGTH"
SUBGOAL_N_VALUES="$SUBGOAL_N_VALUES"
SUBGOAL_EPOCHS="$SUBGOAL_EPOCHS"
SUBGOAL_BATCH="$SUBGOAL_BATCH"
SUBGOAL_SIZE="$SUBGOAL_SIZE"
SUBGOAL_WORKERS="$SUBGOAL_WORKERS"
EVAL_EPISODES="$EVAL_EPISODES"
EVAL_REVISIT_PEN="$EVAL_REVISIT_PEN"
EVAL_GIFS="$EVAL_GIFS"
EVAL_BUDGET_FACTOR="$EVAL_BUDGET_FACTOR"
EVAL_BUDGET_MARGIN="$EVAL_BUDGET_MARGIN"
EVAL_LOOKAHEAD="$EVAL_LOOKAHEAD"
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
  local message="$1"
  {
    echo
    echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): $message"
  } >> "$RUN_DIR/README.md"
}

note "Started on $(hostname), job ${SLURM_JOB_ID:-unknown}."

WANDB_DISABLED=true uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.main \
  --fname=examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_astar.yaml \
  --meta.model_folder="$WM_DIR" \
  --meta.load_model=False \
  --meta.enable_plan_eval=False \
  --logging.log_wandb=False \
  --logging.tqdm_silent=True \
  --optim.epochs="$WM_EPOCHS" \
  --data.size="$WM_SIZE" \
  --data.batch_size="$WM_BATCH" \
  --data.n_steps="$N_STEPS" \
  --data.sample_length="$SAMPLE_LENGTH" \
  --data.min_path_length="$MIN_PATH_LENGTH" \
  --data.pipeline.chunk_size="$WM_CHUNK" \
  --data.pipeline.num_gen_workers="$WM_GEN_WORKERS"

test -f "$WM_DIR/latest.pth.tar"
note "Completed dynamic JEPA WM: $WM_DIR/latest.pth.tar."

for N in $SUBGOAL_N_VALUES; do
  SG_DIR="$RUN_DIR/01_subgoal_N$N"
  EV_DIR="$RUN_DIR/02_eval_N$N"
  mkdir -p "$SG_DIR" "$EV_DIR"
  cat > "$SG_DIR/README.md" <<EOS
# Dynamic Maze Subgoal N=$N

Purpose: distill oracle A* replanning trajectories into a learned waypoint
head. The world model is frozen.

Status:
EOS
  cat > "$EV_DIR/README.md" <<EOS
# Dynamic Maze A*-Free Eval N=$N

Purpose: evaluate the distilled subgoal policy without A* in the action loop.
A* is used only to compute the initial budget and reporting efficiency.

Status:
EOS
  {
    echo
    echo "- $(date -Is): Started subgoal N=$N."
  } >> "$SG_DIR/README.md"
  uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.maze.main_subgoal \
    "$WM_DIR/latest.pth.tar" \
    "$SG_DIR" \
    "$N" \
    "$SUBGOAL_EPOCHS" \
    "$SUBGOAL_BATCH" \
    "$SUBGOAL_SIZE" \
    "$SUBGOAL_WORKERS"
  test -f "$SG_DIR/subgoal.pth.tar"
  {
    echo
    echo "- $(date -Is): Completed subgoal N=$N."
  } >> "$SG_DIR/README.md"
  note "Completed subgoal N=$N: $SG_DIR/subgoal.pth.tar."

  if [ "$EVAL_LOOKAHEAD" = "auto" ]; then
    LOOKAHEAD="$N"
  else
    LOOKAHEAD="$EVAL_LOOKAHEAD"
  fi
  {
    echo
    echo "- $(date -Is): Started eval N=$N lookahead=$LOOKAHEAD."
  } >> "$EV_DIR/README.md"
  uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.eval_subgoal \
    "$WM_DIR/latest.pth.tar" \
    "$SG_DIR/subgoal.pth.tar" \
    "$EV_DIR" \
    "$EVAL_EPISODES" \
    "$LOOKAHEAD" \
    "$EVAL_REVISIT_PEN" \
    "$EVAL_GIFS" \
    "$EVAL_BUDGET_FACTOR" \
    "$EVAL_BUDGET_MARGIN"
  test -f "$EV_DIR/subgoal_eval.json"
  {
    echo
    echo "- $(date -Is): Completed eval N=$N."
  } >> "$EV_DIR/README.md"
  note "Completed eval N=$N: $EV_DIR/subgoal_eval.json."
done

export RUN_DIR SUBGOAL_N_VALUES
uv run --no-sync --project "$ROOT" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
rows = []
for n in os.environ["SUBGOAL_N_VALUES"].split():
    path = run_dir / f"02_eval_N{n}" / "subgoal_eval.json"
    with path.open() as f:
        data = json.load(f)
    rows.append((n, data))

with (run_dir / "metrics.tsv").open("w") as f:
    f.write("N\tsuccess_rate\tmean_efficiency\tmean_steps\tmean_blocked_moves\tlookahead\n")
    for n, data in rows:
        f.write(
            f"{n}\t{data['success_rate']:.6f}\t{data['mean_efficiency']:.6f}\t"
            f"{data['mean_steps']:.3f}\t{data['mean_blocked_moves']:.3f}\t"
            f"{data['lookahead']}\n"
        )

with (run_dir / "README.md").open("a") as f:
    f.write("\nMetrics:\n\n")
    f.write("| N | success | efficiency | steps | blocked |\n")
    f.write("|---:|---:|---:|---:|---:|\n")
    for n, data in rows:
        f.write(
            f"| {n} | {100 * data['success_rate']:.1f}% | "
            f"{data['mean_efficiency']:.3f} | {data['mean_steps']:.1f} | "
            f"{data['mean_blocked_moves']:.1f} |\n"
        )
PY

note "Completed all dynamic JEPA/subgoal stages. Summary: $RUN_DIR/metrics.tsv."
EOF

chmod +x "$JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  append_status "$RUN_DIR/README.md" "Dry run only: generated $JOB_SCRIPT."
  echo "[dynamic-jepa] dry run: $RUN_DIR"
  echo "[dynamic-jepa] generated: $JOB_SCRIPT"
  exit 0
fi

JOB_ID="$(sbatch --parsable "$JOB_SCRIPT")"
append_status "$RUN_DIR/README.md" "Submitted Slurm job $JOB_ID."
echo "[dynamic-jepa] run: $RUN_DIR"
echo "[dynamic-jepa] job: $JOB_ID"
echo "[dynamic-jepa] monitor: squeue -j $JOB_ID"
