#!/usr/bin/env bash
set -euo pipefail

#note: Feature-based approximate Q-learning，默认 reward 与原版 DouZero 的 logadp 对齐。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

#note: 默认配置对齐 DouZero 对照脚本：直接运行会训练 100w 局 history-aware approximate Q-learning。
GPU_DEVICE="${GPU_DEVICE:-0}"
TASK_NAME="${TASK_NAME:-approxq_logadp_cmp_1m_history}"
EPISODES="${EPISODES:-1000000}"
SAVEDIR="${SAVEDIR:-approx_qlearning_checkpoints/approx_qlearning}"
ALPHA="${ALPHA:-0.05}"
GAMMA="${GAMMA:-0.98}"
EPSILON="${EPSILON:-0.1}"
MIN_EPSILON="${MIN_EPSILON:-0.02}"
EPSILON_DECAY="${EPSILON_DECAY:-0.99998}"
OBJECTIVE="${OBJECTIVE:-logadp}"
REWARD_SCALE="${REWARD_SCALE:-1}"
REWARD_SHAPING="${REWARD_SHAPING:-0}"
FEATURE_MODE="${FEATURE_MODE:-history}"
MAX_CANDIDATE_ACTIONS="${MAX_CANDIDATE_ACTIONS:-64}"
L2="${L2:-0.00001}"
CLIP_TD="${CLIP_TD:-10}"
DEVICE="${DEVICE:-cpu}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"

#note: 可选配置。OUTPUT 指定单个输出文件；RESUME=1 表示从最新 checkpoint 继续。
OUTPUT="${OUTPUT:-}"
RESUME="${RESUME:-0}"
if [[ "${SAVEDIR}" = /* ]]; then
  CHECKPOINT_ROOT="${SAVEDIR}"
else
  CHECKPOINT_ROOT="${REPO_ROOT}/${SAVEDIR}"
fi
LOG_PATH="${LOG_PATH:-${CHECKPOINT_ROOT}/${TASK_NAME}/train.log}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/run_logs}"
TIME_LOG="${TIME_LOG:-${RUN_LOG_DIR}/${TASK_NAME}.time.log}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}/src}"

mkdir -p "$(dirname "${LOG_PATH}")" "${RUN_LOG_DIR}"
exec > >(tee -a "${LOG_PATH}" "${TIME_LOG}") 2>&1

echo "Training log: ${LOG_PATH}"
echo "Time log: ${TIME_LOG}"
echo "Task=${TASK_NAME} Episodes=${EPISODES} Objective=${OBJECTIVE} RewardScale=${REWARD_SCALE} RewardShaping=${REWARD_SHAPING}"
echo "Alpha=${ALPHA} Gamma=${GAMMA} Epsilon=${EPSILON} MinEpsilon=${MIN_EPSILON} EpsilonDecay=${EPSILON_DECAY}"
echo "Device=${DEVICE} FeatureMode=${FEATURE_MODE} MaxCandidateActions=${MAX_CANDIDATE_ACTIONS} ProgressInterval=${PROGRESS_INTERVAL} LogInterval=${LOG_INTERVAL} SaveInterval=${SAVE_INTERVAL}"

python - <<'PY'
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("CUDA check skipped:", repr(exc))
PY

echo "Note: this model has fixed-size linear weights. CUDA can be used for batched dot products, but CPU may be faster for small candidate sets."

#note: 这里是真正传给 train_approx_qlearning.py 的参数列表。
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
  --feature_mode "${FEATURE_MODE}"
  --max_candidate_actions "${MAX_CANDIDATE_ACTIONS}"
  --l2 "${L2}"
  --clip_td "${CLIP_TD}"
  --device "${DEVICE}"
  --log_interval "${LOG_INTERVAL}"
  --progress_interval "${PROGRESS_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
)

if [[ "${REWARD_SHAPING}" == "1" || "${REWARD_SHAPING}" == "true" || "${REWARD_SHAPING}" == "yes" ]]; then
  ARGS+=(--reward_shaping)
fi

if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi

if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  ARGS+=(--resume)
fi

cd "${REPO_ROOT}"
/usr/bin/time -v python src/train_approx_qlearning.py "${ARGS[@]}"
