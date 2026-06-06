#!/usr/bin/env bash
set -euo pipefail

#note: 默认训练配置。常规修改优先改这里，外部环境变量仍可临时覆盖。
GPU_DEVICE="${GPU_DEVICE:-0}"
TASK_NAME="${TASK_NAME:-qlearning_adlog_10000k_shaped10}"
EPISODES="${EPISODES:-10000000}"
SAVEDIR="${SAVEDIR:-qlearning_checkpoints/qlearning}"
ALPHA="${ALPHA:-0.05}"
GAMMA="${GAMMA:-0.98}"
EPSILON="${EPSILON:-0.2}"
MIN_EPSILON="${MIN_EPSILON:-0.2}"
EPSILON_DECAY="${EPSILON_DECAY:-0.999999977}"
OBJECTIVE="${OBJECTIVE:-logadp}"
REWARD_SCALE="${REWARD_SCALE:-10}"
REWARD_SHAPING="${REWARD_SHAPING:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-125000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500000}"

#note: 可选配置。OUTPUT 指定单个输出文件；RESUME=1 表示从最新 checkpoint 继续。
OUTPUT="${OUTPUT:-}"
RESUME="${RESUME:-0}"
LOG_PATH="${LOG_PATH:-${SAVEDIR}/${TASK_NAME}/train.log}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "Training log: ${LOG_PATH}"
echo "Task=${TASK_NAME} Episodes=${EPISODES} RewardScale=${REWARD_SCALE} RewardShaping=${REWARD_SHAPING} ProgressInterval=${PROGRESS_INTERVAL} LogInterval=${LOG_INTERVAL} SaveInterval=${SAVE_INTERVAL}"

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

#note: 这里才是真正传给 train_qlearning.py 的参数列表。
ARGS=(
  --episodes "${EPISODES}"
  --name "${TASK_NAME}"
  --savedir "${SAVEDIR}"
  --alpha "${ALPHA}"
  --gamma "${GAMMA}"
  --epsilon "${EPSILON}"
  --min_epsilon "${MIN_EPSILON}"
  --epsilon_decay "${EPSILON_DECAY}"
  --objective "${OBJECTIVE}"
  --reward_scale "${REWARD_SCALE}"
  --log_interval "${LOG_INTERVAL}"
  --progress_interval "${PROGRESS_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
)

if [[ "${REWARD_SHAPING}" == "0" || "${REWARD_SHAPING}" == "false" || "${REWARD_SHAPING}" == "no" ]]; then
  ARGS+=(--no_reward_shaping)
fi

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  ARGS+=(--resume)
fi

python src/train_qlearning.py "${ARGS[@]}"
