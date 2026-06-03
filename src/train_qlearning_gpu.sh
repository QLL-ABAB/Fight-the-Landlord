#!/usr/bin/env bash
set -euo pipefail

GPU_DEVICE="${GPU_DEVICE:-0}"
EPISODES="${EPISODES:-10000}"
TASK_NAME="${TASK_NAME:-task_name}"
SAVEDIR="${SAVEDIR:-qlearning_checkpoints/qlearning}"
OUTPUT="${OUTPUT:-}"
RESUME="${RESUME:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export PYTHONPATH="${PYTHONPATH:-src}"

python - <<'PY'
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("CUDA check skipped:", repr(exc))
PY

echo "Note: current Q-learning is tabular, so CUDA is visible but training logic runs on CPU."

ARGS=(
  --episodes "${EPISODES}"
  --name "${TASK_NAME}"
  --savedir "${SAVEDIR}"
  --alpha 0.1
  --gamma 0.95
  --epsilon 0.2
  --min_epsilon 0.02
  --epsilon_decay 0.9995
  --objective wp
)

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  ARGS+=(--resume)
fi

python src/train_qlearning.py "${ARGS[@]}"
