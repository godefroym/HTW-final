#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/dynamic_maze_baseline_$STAMP}"
LOG_DIR="$RUN_DIR/slurm_logs"

TIME_LIMIT="${HTW_DYN_MAZE_TIME:-01:00:00}"
CPUS="${HTW_DYN_MAZE_CPUS:-8}"
EPISODES="${HTW_DYN_MAZE_EPISODES:-128}"
NUM_DOORS="${HTW_DYN_MAZE_NUM_DOORS:-8}"
DOOR_TOGGLE_PROB="${HTW_DYN_MAZE_TOGGLE_PROB:-0.04}"
DOOR_OPEN_PROB="${HTW_DYN_MAZE_OPEN_PROB:-0.35}"
FOG_RADIUS="${HTW_DYN_MAZE_FOG_RADIUS:-4}"
N_GIFS="${HTW_DYN_MAZE_GIFS:-6}"
POLICIES="${HTW_DYN_MAZE_POLICIES:-oracle_replan,fog_optimistic,fog_conservative,memory_optimistic,memory_conservative,random}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

now_iso() {
  date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"
}

append_status() {
  local message="$1"
  {
    echo
    echo "- $(now_iso): $message"
  } >> "$RUN_DIR/README.md"
}

cat > "$RUN_DIR/README.md" <<EOF
# HTW Dynamic Maze Baseline Run

Created: $STAMP

Purpose: calibrate the stochastic-door + fog-of-war Maze pivot with naive A*
baselines before training a world model.

Config:

- episodes: \`$EPISODES\`
- policies: \`$POLICIES\`
- doors: \`$NUM_DOORS\`
- door toggle probability: \`$DOOR_TOGGLE_PROB\`
- door initial open probability: \`$DOOR_OPEN_PROB\`
- fog radius: \`$FOG_RADIUS\`

Checklist:

- [ ] Run A* baselines.
- [ ] Inspect summary and GIFs.
- [ ] Select difficulty for first WM run.

Status:
EOF

JOB_SCRIPT="$RUN_DIR/dynamic_maze_baseline.sbatch"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=htw_dyn_maze"
  echo "#SBATCH --partition=$partition"
  [ -n "$account" ] && echo "#SBATCH --account=$account"
  [ -n "$qos" ] && echo "#SBATCH --qos=$qos"
  [ -n "$reservation" ] && echo "#SBATCH --reservation=$reservation"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --ntasks=1"
  echo "#SBATCH --cpus-per-task=$CPUS"
  echo "#SBATCH --time=$TIME_LIMIT"
  echo "#SBATCH --output=$LOG_DIR/%x-%j.out"
  echo "#SBATCH --error=$LOG_DIR/%x-%j.err"
  echo
  echo "set -euo pipefail"
} > "$JOB_SCRIPT"

cat >> "$JOB_SCRIPT" <<EOF
ROOT="$ROOT"
RUN_DIR="$RUN_DIR"
EPISODES="$EPISODES"
NUM_DOORS="$NUM_DOORS"
DOOR_TOGGLE_PROB="$DOOR_TOGGLE_PROB"
DOOR_OPEN_PROB="$DOOR_OPEN_PROB"
FOG_RADIUS="$FOG_RADIUS"
N_GIFS="$N_GIFS"
POLICIES="$POLICIES"
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

{
  echo
  echo "- $(date -Is): Started on $(hostname), job ${SLURM_JOB_ID:-unknown}."
} >> "$RUN_DIR/README.md"

uv run --no-sync --project "$ROOT" python -m examples.ac_video_jepa.dynamic_maze.eval_astar_baselines \
  --out "$RUN_DIR/eval" \
  --episodes "$EPISODES" \
  --num-doors "$NUM_DOORS" \
  --door-toggle-prob "$DOOR_TOGGLE_PROB" \
  --door-open-prob "$DOOR_OPEN_PROB" \
  --fog-radius "$FOG_RADIUS" \
  --n-gifs "$N_GIFS" \
  --policies "$POLICIES"

{
  echo
  echo "- $(date -Is): Completed. Summary: $RUN_DIR/eval/summary.tsv."
  echo
  echo "Metrics:"
  echo
  sed 's/\t/ | /g' "$RUN_DIR/eval/summary.tsv"
} >> "$RUN_DIR/README.md"
EOF

chmod +x "$JOB_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  append_status "Dry run only: generated $JOB_SCRIPT."
  echo "[dynamic-maze] dry run: $RUN_DIR"
  echo "[dynamic-maze] generated: $JOB_SCRIPT"
  exit 0
fi

JOB_ID="$(sbatch --parsable "$JOB_SCRIPT")"
append_status "Submitted Slurm job $JOB_ID."
echo "[dynamic-maze] run: $RUN_DIR"
echo "[dynamic-maze] job: $JOB_ID"
echo "[dynamic-maze] monitor: squeue -j $JOB_ID"
