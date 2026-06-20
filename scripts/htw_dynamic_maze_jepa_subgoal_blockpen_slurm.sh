#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_jepa_subgoal_blockpen_$STAMP}"
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
SUBGOAL_BLOCK_PENALTY="${HTW_DYN_SUBGOAL_BLOCK_PENALTY:-5.0}"
SUBGOAL_BLOCK_RADIUS="${HTW_DYN_SUBGOAL_BLOCK_RADIUS:-1}"
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

write_header() {
  local path="$1"
  local name="$2"
  local cpus="$3"
  local gres_gpu="${4:-1}"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=$name"
    echo "#SBATCH --partition=$partition"
    [ -n "$account" ] && echo "#SBATCH --account=$account"
    [ -n "$qos" ] && echo "#SBATCH --qos=$qos"
    [ -n "$reservation" ] && echo "#SBATCH --reservation=$reservation"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=$cpus"
    if [ "$gres_gpu" != "0" ]; then
      echo "#SBATCH --gres=gpu:$gres_gpu"
    fi
    echo "#SBATCH --time=$TIME_LIMIT"
    echo "#SBATCH --output=$LOG_DIR/%x-%j.out"
    echo "#SBATCH --error=$LOG_DIR/%x-%j.err"
    echo
    echo "set -euo pipefail"
  } > "$path"
}

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze JEPA Subgoals, Block-Penalized Variant

Created: $STAMP

Goal: rerun the dynamic-maze JEPA/subgoal experiment with a stronger training
penalty on predicted waypoints that land on visible blocked cells. Unknown fog is
not penalized.

Run directory:

\`$RUN_DIR\`

Training:

- WM config: \`examples/ac_video_jepa/cfgs/train/dynamic_maze/train_dynamic_maze_astar.yaml\`
- WM epochs: \`$WM_EPOCHS\`
- WM samples/epoch: \`$WM_SIZE\`
- subgoal N values: \`$SUBGOAL_N_VALUES\`
- subgoal epochs: \`$SUBGOAL_EPOCHS\`
- subgoal block penalty: \`$SUBGOAL_BLOCK_PENALTY\`
- subgoal block radius px: \`$SUBGOAL_BLOCK_RADIUS\`

Evaluation:

- episodes: \`$EVAL_EPISODES\`
- lookahead: \`$EVAL_LOOKAHEAD\`
- budget: \`$EVAL_BUDGET_FACTOR x A*_initial + $EVAL_BUDGET_MARGIN\`

Checklist:

- [ ] Train shared dynamic JEPA WM.
- [ ] Train/evaluate subgoal N=2 with block penalty.
- [ ] Train/evaluate subgoal N=4 with block penalty.
- [ ] Write metrics.tsv summary.

Status:
EOF

WM_JOB_SCRIPT="$RUN_DIR/00_train_wm.sbatch"
write_header "$WM_JOB_SCRIPT" "htw_dyn_wm_blk" "$CPUS" 1
cat >> "$WM_JOB_SCRIPT" <<EOF
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
EOF
cat >> "$WM_JOB_SCRIPT" <<'EOF'

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

note "Started shared WM on $(hostname), job ${SLURM_JOB_ID:-unknown}."

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
note "Completed shared dynamic JEPA WM: $WM_DIR/latest.pth.tar."
EOF
chmod +x "$WM_JOB_SCRIPT"

SG_JOB_SCRIPTS=()
for N in $SUBGOAL_N_VALUES; do
  SG_DIR="$RUN_DIR/01_subgoal_N$N"
  EV_DIR="$RUN_DIR/02_eval_N$N"
  mkdir -p "$SG_DIR" "$EV_DIR"
  cat > "$SG_DIR/README.md" <<EOF
# Dynamic Maze Block-Penalized Subgoal N=$N

Purpose: distill oracle A* replanning trajectories into a learned waypoint head,
with an added training penalty for waypoints on visible blocked cells.

Status:
EOF
  cat > "$EV_DIR/README.md" <<EOF
# Dynamic Maze A*-Free Eval N=$N, Block-Penalized

Purpose: evaluate the block-penalized subgoal policy without A* in the action loop.

Status:
EOF

  SG_JOB_SCRIPT="$RUN_DIR/subgoal_eval_N${N}.sbatch"
  write_header "$SG_JOB_SCRIPT" "htw_sgB_N${N}" "$CPUS" 1
  cat >> "$SG_JOB_SCRIPT" <<EOF
ROOT="$ROOT"
RUN_DIR="$RUN_DIR"
WM_DIR="$WM_DIR"
SG_DIR="$SG_DIR"
EV_DIR="$EV_DIR"
N="$N"
SUBGOAL_EPOCHS="$SUBGOAL_EPOCHS"
SUBGOAL_BATCH="$SUBGOAL_BATCH"
SUBGOAL_SIZE="$SUBGOAL_SIZE"
SUBGOAL_WORKERS="$SUBGOAL_WORKERS"
SUBGOAL_BLOCK_PENALTY="$SUBGOAL_BLOCK_PENALTY"
SUBGOAL_BLOCK_RADIUS="$SUBGOAL_BLOCK_RADIUS"
EVAL_EPISODES="$EVAL_EPISODES"
EVAL_REVISIT_PEN="$EVAL_REVISIT_PEN"
EVAL_GIFS="$EVAL_GIFS"
EVAL_BUDGET_FACTOR="$EVAL_BUDGET_FACTOR"
EVAL_BUDGET_MARGIN="$EVAL_BUDGET_MARGIN"
EVAL_LOOKAHEAD="$EVAL_LOOKAHEAD"
EOF
  cat >> "$SG_JOB_SCRIPT" <<'EOF'

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
module load python312 2>/dev/null || true
export PATH="$UV_INSTALL_DIR:$PATH"
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

note_root() {
  {
    echo
    echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): $1"
  } >> "$RUN_DIR/README.md"
}

note_local() {
  local readme="$1"
  local message="$2"
  {
    echo
    echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): $message"
  } >> "$readme"
}

note_local "$SG_DIR/README.md" "Started subgoal N=$N, job ${SLURM_JOB_ID:-unknown}."
uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.maze.main_subgoal \
  "$WM_DIR/latest.pth.tar" \
  "$SG_DIR" \
  "$N" \
  "$SUBGOAL_EPOCHS" \
  "$SUBGOAL_BATCH" \
  "$SUBGOAL_SIZE" \
  "$SUBGOAL_WORKERS" \
  "$SUBGOAL_BLOCK_PENALTY" \
  "$SUBGOAL_BLOCK_RADIUS"
test -f "$SG_DIR/subgoal.pth.tar"
note_local "$SG_DIR/README.md" "Completed block-penalized subgoal N=$N."
note_root "Completed block-penalized subgoal N=$N: $SG_DIR/subgoal.pth.tar."

if [ "$EVAL_LOOKAHEAD" = "auto" ]; then
  LOOKAHEAD="$N"
else
  LOOKAHEAD="$EVAL_LOOKAHEAD"
fi
note_local "$EV_DIR/README.md" "Started eval N=$N lookahead=$LOOKAHEAD."
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
note_local "$EV_DIR/README.md" "Completed eval N=$N."
note_root "Completed block-penalized eval N=$N: $EV_DIR/subgoal_eval.json."
EOF
  chmod +x "$SG_JOB_SCRIPT"
  SG_JOB_SCRIPTS+=("$SG_JOB_SCRIPT")
done

SUMMARY_JOB_SCRIPT="$RUN_DIR/99_summary.sbatch"
write_header "$SUMMARY_JOB_SCRIPT" "htw_sgB_sum" 2 0
cat >> "$SUMMARY_JOB_SCRIPT" <<EOF
ROOT="$ROOT"
RUN_DIR="$RUN_DIR"
SUBGOAL_N_VALUES="$SUBGOAL_N_VALUES"
EOF
cat >> "$SUMMARY_JOB_SCRIPT" <<'EOF'

cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
module load python312 2>/dev/null || true
export PATH="$UV_INSTALL_DIR:$PATH"
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
    f.write("N\tsuccess_rate\tmean_efficiency\tmean_steps\tmean_blocked_moves\tlookahead\tblock_penalty\n")
    for n, data in rows:
        f.write(
            f"{n}\t{data['success_rate']:.6f}\t{data['mean_efficiency']:.6f}\t"
            f"{data['mean_steps']:.3f}\t{data['mean_blocked_moves']:.3f}\t"
            f"{data['lookahead']}\t{data.get('block_penalty', '')}\n"
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
{
  echo
  echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): Completed summary: $RUN_DIR/metrics.tsv."
} >> "$RUN_DIR/README.md"
EOF
chmod +x "$SUMMARY_JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  append_status "$RUN_DIR/README.md" "Dry run only: generated Slurm scripts."
  echo "[dynamic-jepa-blockpen] dry run: $RUN_DIR"
  echo "[dynamic-jepa-blockpen] generated: $WM_JOB_SCRIPT ${SG_JOB_SCRIPTS[*]} $SUMMARY_JOB_SCRIPT"
  exit 0
fi

WM_JOB_ID="$(sbatch --parsable "$WM_JOB_SCRIPT")"
SG_JOB_IDS=()
for script in "${SG_JOB_SCRIPTS[@]}"; do
  SG_JOB_IDS+=("$(sbatch --parsable --dependency=afterok:"$WM_JOB_ID" "$script")")
done
summary_dep="$(IFS=:; echo "${SG_JOB_IDS[*]}")"
SUMMARY_JOB_ID="$(sbatch --parsable --dependency=afterok:"$summary_dep" "$SUMMARY_JOB_SCRIPT")"

{
  echo -e "stage\tN\tjob_id\tdependency\tpath"
  echo -e "wm\t-\t$WM_JOB_ID\t-\t$WM_DIR"
  i=0
  for N in $SUBGOAL_N_VALUES; do
    echo -e "subgoal_eval\t$N\t${SG_JOB_IDS[$i]}\tafterok:$WM_JOB_ID\t$RUN_DIR/01_subgoal_N$N"
    i=$((i + 1))
  done
  echo -e "summary\t-\t$SUMMARY_JOB_ID\tafterok:$summary_dep\t$RUN_DIR/metrics.tsv"
} > "$RUN_DIR/jobs.tsv"

append_status "$RUN_DIR/README.md" "Submitted WM job $WM_JOB_ID."
i=0
for N in $SUBGOAL_N_VALUES; do
  append_status "$RUN_DIR/README.md" "Submitted parallel subgoal/eval N=$N job ${SG_JOB_IDS[$i]} after WM."
  i=$((i + 1))
done
append_status "$RUN_DIR/README.md" "Submitted summary job $SUMMARY_JOB_ID."

echo "[dynamic-jepa-blockpen] run: $RUN_DIR"
echo "[dynamic-jepa-blockpen] wm job: $WM_JOB_ID"
echo "[dynamic-jepa-blockpen] subgoal/eval jobs: ${SG_JOB_IDS[*]}"
echo "[dynamic-jepa-blockpen] summary job: $SUMMARY_JOB_ID"
echo "[dynamic-jepa-blockpen] jobs table: $RUN_DIR/jobs.tsv"
echo "[dynamic-jepa-blockpen] monitor: squeue -u $USER"
