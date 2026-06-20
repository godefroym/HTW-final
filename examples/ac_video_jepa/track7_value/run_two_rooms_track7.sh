#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <two_rooms_world_model_ckpt> <output_dir>" >&2
  exit 2
fi

WM_CKPT="$1"
OUT_DIR="$2"
VALUE_DIR="${OUT_DIR}/value_head"

TRAIN_CFG="examples/ac_video_jepa/cfgs/train/two_rooms/train_value.yaml"
EVAL_CFG="examples/ac_video_jepa/cfgs/eval/two_rooms/eval_track7.yaml"

mkdir -p "${OUT_DIR}"

run_python () {
  if [[ -n "${EBJEPA_USE_UV:-}" ]]; then
    uv run --no-sync --project "${EBJEPA_REPO:-.}" python "$@"
  else
    python "$@"
  fi
}

run_python -m examples.ac_video_jepa.main "${TRAIN_CFG}" \
  --meta.init_from="${WM_CKPT}" \
  --meta.model_folder="${VALUE_DIR}"

run_eval () {
  local label="$1"
  local plan_cfg="$2"
  local run_dir="${OUT_DIR}/${label}"
  mkdir -p "${run_dir}"
  cp -f "${VALUE_DIR}/latest.pth.tar" "${run_dir}/latest.pth.tar"
  run_python -m examples.ac_video_jepa.main "${TRAIN_CFG}" \
    --meta.load_model=true \
    --meta.eval_only_mode=true \
    --meta.skip_unroll_eval=true \
    --meta.enable_plan_eval=true \
    --meta.model_folder="${run_dir}" \
    --eval.plan_cfg_path="${plan_cfg}" \
    --eval.eval_cfg_path="${EVAL_CFG}"
}

run_eval "repr_s50" "examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_repr_track7_s50.yaml"
run_eval "value_s50" "examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_value_track7_s50.yaml"
run_eval "repr_s200" "examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_repr_track7.yaml"
run_eval "value_s200" "examples/ac_video_jepa/cfgs/planning/two_rooms/planning_mppi_value_track7.yaml"

echo "Track 7 runs finished under ${OUT_DIR}"
