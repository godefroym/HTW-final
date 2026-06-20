#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f ./env.sh ]; then
  # shellcheck disable=SC1091
  source ./env.sh
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${1:-${EBJEPA_WORK:-$ROOT}/runs/track7_two_rooms_$STAMP}"
WM_CKPT="${2:-}"
LOG_DIR="$RUN_DIR/slurm_logs"
WM_DIR="$RUN_DIR/00_two_rooms_wm"
TRACK7_DIR="$RUN_DIR/01_track7_value"

WM_TIME="${HTW_TR7_WM_TIME:-04:00:00}"
VALUE_TIME="${HTW_TR7_VALUE_TIME:-03:00:00}"
WM_CPUS="${HTW_TR7_WM_CPUS:-24}"
VALUE_CPUS="${HTW_TR7_VALUE_CPUS:-24}"
WM_EPOCHS="${HTW_TR7_WM_EPOCHS:-12}"
DRY_RUN="${HTW_DRY_RUN:-false}"

partition="${EBJEPA_SLURM_PARTITION:-defq}"
account="${EBJEPA_SLURM_ACCOUNT:-}"
qos="${EBJEPA_SLURM_QOS:-}"
# HTW Dalia GPUs are exposed through the Vivatech reservation during the event.
# Keep env overrides for portability, but default to the reserved pool here.
reservation="${HTW_SLURM_RESERVATION:-${EBJEPA_SLURM_RESERVATION:-Vivatech}}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$WM_DIR" "$TRACK7_DIR"

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

write_sbatch_header() {
  local file="$1"
  local name="$2"
  local cpus="$3"
  local time_limit="$4"
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
    echo "#SBATCH --gres=gpu:1"
    echo "#SBATCH --time=$time_limit"
    echo "#SBATCH --output=$LOG_DIR/%x-%j.out"
    echo "#SBATCH --error=$LOG_DIR/%x-%j.err"
    echo
    echo "set -euo pipefail"
  } > "$file"
}

write_bootstrap() {
  local file="$1"
  cat >> "$file" <<'EOF'

bootstrap_ebjepa() {
  local repo="$1"
  cd "$repo"
  # shellcheck disable=SC1091
  source "$repo/env.sh"
  module load python312 2>/dev/null || true
  if ! "$UV_INSTALL_DIR/uv" --version >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
  fi
  export PATH="$UV_INSTALL_DIR:$PATH"
  export WANDB_DISABLED=true
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  local lock_file="$EBJEPA_WORK/.htw_uv_sync_${ARCH}.lock"
  local ready_file="$UV_PROJECT_ENVIRONMENT/.htw_uv_ready"
  mkdir -p "$(dirname "$lock_file")"
  (
    if command -v flock >/dev/null 2>&1; then
      flock 9
    fi
    if [ ! -f "$ready_file" ] || ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import torch" >/dev/null 2>&1; then
      uv sync --dev --project "$repo"
      "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import torch"
      touch "$ready_file"
    else
      echo "uv environment already ready: $UV_PROJECT_ENVIRONMENT"
    fi
  ) 9>"$lock_file"
  cd "$repo"
}

note() {
  local readme="$1"
  local message="$2"
  {
    echo
    echo "- $(date -Is 2>/dev/null || date "+%Y-%m-%dT%H:%M:%S%z"): $message"
  } >> "$readme"
}
EOF
}

cat > "$RUN_DIR/README.md" <<EOF
# HTW Track 7 Two Rooms Run

Created: $STAMP

Goal: compare \`repr_dist\` against \`learned_value\` on Two Rooms with matched
world model, MPPI settings, and eval episodes.

Run directory:

\`$RUN_DIR\`

Stages:

1. \`00_two_rooms_wm\`: train a stable Two Rooms world model if no checkpoint was supplied.
2. \`01_track7_value\`: train the value head and run matched evals at 50 and 200 MPPI samples.

Inputs:

- supplied WM checkpoint: \`${WM_CKPT:-none; train first}\`
- WM epochs if training: \`$WM_EPOCHS\`

Remaining work:

- Monitor Slurm jobs.
- Run \`summarize_two_rooms_track7.py\` after evals finish.
- Move table/plot/GIFs into final slides.

Status:
EOF

WM_JOB_ID=""
if [ -z "$WM_CKPT" ]; then
  WM_SCRIPT="$RUN_DIR/00_train_two_rooms_wm.sbatch"
  write_sbatch_header "$WM_SCRIPT" "htw_tr7_wm" "$WM_CPUS" "$WM_TIME"
  cat >> "$WM_SCRIPT" <<EOF
ROOT="$ROOT"
WM_DIR="$WM_DIR"
README="$RUN_DIR/README.md"
EOF
  write_bootstrap "$WM_SCRIPT"
  cat >> "$WM_SCRIPT" <<EOF

note "\$README" "Started Two Rooms WM training on \$(hostname), job \${SLURM_JOB_ID:-unknown}."
bootstrap_ebjepa "\$ROOT"
WANDB_DISABLED=true uv run --no-sync --project "\$ROOT" python -m examples.ac_video_jepa.main \
  --fname=examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
  --meta.model_folder="\$WM_DIR" \
  --meta.load_model=false \
  --meta.enable_plan_eval=false \
  --logging.log_wandb=false \
  --logging.tqdm_silent=true \
  --optim.epochs="$WM_EPOCHS"
test -f "\$WM_DIR/latest.pth.tar"
note "\$README" "Completed Two Rooms WM: \$WM_DIR/latest.pth.tar."
EOF
  chmod +x "$WM_SCRIPT"
  WM_CKPT="$WM_DIR/latest.pth.tar"
fi

TR7_SCRIPT="$RUN_DIR/01_track7_value.sbatch"
write_sbatch_header "$TR7_SCRIPT" "htw_tr7_value" "$VALUE_CPUS" "$VALUE_TIME"
cat >> "$TR7_SCRIPT" <<EOF
ROOT="$ROOT"
WM_CKPT="$WM_CKPT"
TRACK7_DIR="$TRACK7_DIR"
README="$RUN_DIR/README.md"
EOF
write_bootstrap "$TR7_SCRIPT"
cat >> "$TR7_SCRIPT" <<'EOF'

note "$README" "Started Track 7 value/eval on $(hostname), job ${SLURM_JOB_ID:-unknown}."
bootstrap_ebjepa "$ROOT"
export EBJEPA_USE_UV=1
export EBJEPA_REPO="$ROOT"
bash "$ROOT/examples/ac_video_jepa/track7_value/run_two_rooms_track7.sh" "$WM_CKPT" "$TRACK7_DIR"
uv run --no-sync --project "$ROOT" python "$ROOT/examples/ac_video_jepa/track7_value/summarize_two_rooms_track7.py" "$TRACK7_DIR"
note "$README" "Completed Track 7 value/eval. Summary: $TRACK7_DIR/track7_summary.csv."
EOF
chmod +x "$TR7_SCRIPT"

if [ "$DRY_RUN" = "true" ]; then
  append_status "Dry run only: sbatch scripts generated but not submitted."
  find "$RUN_DIR" -maxdepth 1 -name '*.sbatch' -print | sort
  exit 0
fi

if [ -z "${2:-}" ]; then
  WM_JOB_ID="$(sbatch --parsable "$WM_SCRIPT")"
  TR7_JOB_ID="$(sbatch --parsable --dependency=afterok:"$WM_JOB_ID" "$TR7_SCRIPT")"
  append_status "Submitted WM job $WM_JOB_ID and dependent Track 7 job $TR7_JOB_ID."
  echo "Submitted WM job: $WM_JOB_ID"
  echo "Submitted Track 7 job: $TR7_JOB_ID"
else
  TR7_JOB_ID="$(sbatch --parsable "$TR7_SCRIPT")"
  append_status "Submitted Track 7 job $TR7_JOB_ID using supplied WM checkpoint."
  echo "Submitted Track 7 job: $TR7_JOB_ID"
fi

echo "Run dir: $RUN_DIR"
