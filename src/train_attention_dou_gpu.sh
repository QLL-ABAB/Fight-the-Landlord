#!/usr/bin/env bash
set -euo pipefail

#TODO: AttentionDou 使用 DouZero 54 维 token + multi-head attention 估计 Q-value。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HAS_TASK_NAME=0; [[ -v TASK_NAME ]] && HAS_TASK_NAME=1
HAS_EPISODES=0; [[ -v EPISODES ]] && HAS_EPISODES=1
HAS_SAVEDIR=0; [[ -v SAVEDIR ]] && HAS_SAVEDIR=1
HAS_OBJECTIVE=0; [[ -v OBJECTIVE ]] && HAS_OBJECTIVE=1
HAS_REWARD_SCALE=0; [[ -v REWARD_SCALE ]] && HAS_REWARD_SCALE=1
HAS_UPDATE_MODE=0; [[ -v UPDATE_MODE ]] && HAS_UPDATE_MODE=1
HAS_LEARNING_RATE=0; [[ -v LEARNING_RATE ]] && HAS_LEARNING_RATE=1
HAS_GAMMA=0; [[ -v GAMMA ]] && HAS_GAMMA=1
HAS_EPSILON=0; [[ -v EPSILON ]] && HAS_EPSILON=1
HAS_MIN_EPSILON=0; [[ -v MIN_EPSILON ]] && HAS_MIN_EPSILON=1
HAS_EPSILON_DECAY=0; [[ -v EPSILON_DECAY ]] && HAS_EPSILON_DECAY=1
HAS_WEIGHT_DECAY=0; [[ -v WEIGHT_DECAY ]] && HAS_WEIGHT_DECAY=1
HAS_MAX_GRAD_NORM=0; [[ -v MAX_GRAD_NORM ]] && HAS_MAX_GRAD_NORM=1
HAS_HIDDEN_DIM=0; [[ -v HIDDEN_DIM ]] && HAS_HIDDEN_DIM=1
HAS_NUM_HEADS=0; [[ -v NUM_HEADS ]] && HAS_NUM_HEADS=1
HAS_NUM_LAYERS=0; [[ -v NUM_LAYERS ]] && HAS_NUM_LAYERS=1
HAS_DROPOUT=0; [[ -v DROPOUT ]] && HAS_DROPOUT=1
HAS_DEVICE=0; [[ -v DEVICE ]] && HAS_DEVICE=1
HAS_ACTOR_DEVICE=0; [[ -v ACTOR_DEVICE ]] && HAS_ACTOR_DEVICE=1
HAS_NUM_WORKERS=0; [[ -v NUM_WORKERS ]] && HAS_NUM_WORKERS=1
HAS_NUM_THREADS=0; [[ -v NUM_THREADS ]] && HAS_NUM_THREADS=1
HAS_CPU_THREADS=0; [[ -v CPU_THREADS ]] && HAS_CPU_THREADS=1
HAS_BUFFER_SIZE=0; [[ -v BUFFER_SIZE ]] && HAS_BUFFER_SIZE=1
HAS_LEARN_BATCH_SIZE=0; [[ -v LEARN_BATCH_SIZE ]] && HAS_LEARN_BATCH_SIZE=1
HAS_LEARN_STEPS=0; [[ -v LEARN_STEPS ]] && HAS_LEARN_STEPS=1
HAS_BASELINE_BETA=0; [[ -v BASELINE_BETA ]] && HAS_BASELINE_BETA=1
HAS_LOG_INTERVAL=0; [[ -v LOG_INTERVAL ]] && HAS_LOG_INTERVAL=1
HAS_PROGRESS_INTERVAL=0; [[ -v PROGRESS_INTERVAL ]] && HAS_PROGRESS_INTERVAL=1
HAS_SAVE_INTERVAL=0; [[ -v SAVE_INTERVAL ]] && HAS_SAVE_INTERVAL=1
HAS_MAX_STEPS=0; [[ -v MAX_STEPS ]] && HAS_MAX_STEPS=1
HAS_UNROLL_LENGTH=0; [[ -v UNROLL_LENGTH ]] && HAS_UNROLL_LENGTH=1
HAS_NUM_BUFFERS=0; [[ -v NUM_BUFFERS ]] && HAS_NUM_BUFFERS=1
HAS_RMSPROP_ALPHA=0; [[ -v RMSPROP_ALPHA ]] && HAS_RMSPROP_ALPHA=1
HAS_MOMENTUM=0; [[ -v MOMENTUM ]] && HAS_MOMENTUM=1
HAS_OPTIMIZER_EPS=0; [[ -v OPTIMIZER_EPS ]] && HAS_OPTIMIZER_EPS=1

CONFIG="${CONFIG:-}"
TASK_NAME="${TASK_NAME:-attention_dou_logadp_td}"
if [[ -n "${CONFIG}" && "${HAS_TASK_NAME}" -eq 0 ]]; then
  TASK_NAME="${CONFIG}"
fi
EPISODES="${EPISODES:-100000}"
SAVEDIR="${SAVEDIR:-attention_dou_checkpoints/attention_dou}"
GPU_DEVICE="${GPU_DEVICE:-0}"
DEVICE="${DEVICE:-auto}"
ACTOR_DEVICE="${ACTOR_DEVICE:-auto}"
OBJECTIVE="${OBJECTIVE:-logadp}"
REWARD_SCALE="${REWARD_SCALE:-1}"
REWARD_SHAPING="${REWARD_SHAPING:-0}"
UPDATE_MODE="${UPDATE_MODE:-td}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
GAMMA="${GAMMA:-1}"
EPSILON="${EPSILON:-0.1}"
MIN_EPSILON="${MIN_EPSILON:-0.02}"
EPSILON_DECAY="${EPSILON_DECAY:-0.99998}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_THREADS="${NUM_THREADS:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
MAX_STEPS="${MAX_STEPS:-1000}"
UNROLL_LENGTH="${UNROLL_LENGTH:-20}"
NUM_BUFFERS="${NUM_BUFFERS:-64}"
CPU_THREADS="${CPU_THREADS:-1}"
BUFFER_SIZE="${BUFFER_SIZE:-0}"
LEARN_BATCH_SIZE="${LEARN_BATCH_SIZE:-4}"
LEARN_STEPS="${LEARN_STEPS:-1}"
BASELINE_BETA="${BASELINE_BETA:-0.01}"
RMSPROP_ALPHA="${RMSPROP_ALPHA:-0.99}"
MOMENTUM="${MOMENTUM:-0}"
OPTIMIZER_EPS="${OPTIMIZER_EPS:-0.00001}"
NICE="${NICE:-0}"
LOAD="${LOAD:-}"
RESUME="${RESUME:-0}"
OUTPUT="${OUTPUT:-}"

RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/run_logs}"
TIME_LOG="${TIME_LOG:-${RUN_LOG_DIR}/${TASK_NAME}.time.log}"
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}/src}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"

mkdir -p "${RUN_LOG_DIR}"
{
  if [[ -n "${CONFIG}" ]]; then
    echo "Task=${TASK_NAME} Config=${CONFIG}"
    echo "Config controls training parameters; explicitly provided env values can override it."
  else
    echo "Task=${TASK_NAME} Episodes=${EPISODES} UpdateMode=${UPDATE_MODE} Objective=${OBJECTIVE}"
    echo "LR=${LEARNING_RATE} Gamma=${GAMMA} Epsilon=${EPSILON} Workers=${NUM_WORKERS} Device=${DEVICE} ActorDevice=${ACTOR_DEVICE}"
  fi
  if [[ -n "${CONFIG}" ]]; then
    echo "Shell defaults before config override: DEVICE=${DEVICE} ACTOR_DEVICE=${ACTOR_DEVICE} NUM_WORKERS=${NUM_WORKERS} BUFFER_SIZE=${BUFFER_SIZE}"
  fi
  echo "Attention: HIDDEN_DIM=${HIDDEN_DIM} NUM_HEADS=${NUM_HEADS} NUM_LAYERS=${NUM_LAYERS} DROPOUT=${DROPOUT}"
  echo "DouZeroBuffer: UNROLL_LENGTH=${UNROLL_LENGTH} NUM_BUFFERS=${NUM_BUFFERS}"
  echo "Learner: LEARN_BATCH_SIZE=${LEARN_BATCH_SIZE} LEARN_STEPS=${LEARN_STEPS} NUM_THREADS=${NUM_THREADS} BASELINE_BETA=${BASELINE_BETA}"
  echo "Optimizer: RMSPROP_ALPHA=${RMSPROP_ALPHA} MOMENTUM=${MOMENTUM} OPTIMIZER_EPS=${OPTIMIZER_EPS}"
  echo "CPU thread caps: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS} NICE=${NICE}"
  python3 - <<'PY'
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("CUDA check skipped:", repr(exc))
PY
  ARGS=()
  if [[ -n "${CONFIG}" ]]; then
    ARGS+=(--config "${CONFIG}")
    (( HAS_TASK_NAME )) && ARGS+=(--name "${TASK_NAME}")
    (( HAS_EPISODES )) && ARGS+=(--episodes "${EPISODES}")
    (( HAS_SAVEDIR )) && ARGS+=(--savedir "${SAVEDIR}")
    (( HAS_OBJECTIVE )) && ARGS+=(--objective "${OBJECTIVE}")
    (( HAS_REWARD_SCALE )) && ARGS+=(--reward_scale "${REWARD_SCALE}")
    (( HAS_UPDATE_MODE )) && ARGS+=(--update_mode "${UPDATE_MODE}")
    (( HAS_LEARNING_RATE )) && ARGS+=(--learning_rate "${LEARNING_RATE}")
    (( HAS_GAMMA )) && ARGS+=(--gamma "${GAMMA}")
    (( HAS_EPSILON )) && ARGS+=(--epsilon "${EPSILON}")
    (( HAS_MIN_EPSILON )) && ARGS+=(--min_epsilon "${MIN_EPSILON}")
    (( HAS_EPSILON_DECAY )) && ARGS+=(--epsilon_decay "${EPSILON_DECAY}")
    (( HAS_WEIGHT_DECAY )) && ARGS+=(--weight_decay "${WEIGHT_DECAY}")
    (( HAS_MAX_GRAD_NORM )) && ARGS+=(--max_grad_norm "${MAX_GRAD_NORM}")
    (( HAS_HIDDEN_DIM )) && ARGS+=(--hidden_dim "${HIDDEN_DIM}")
    (( HAS_NUM_HEADS )) && ARGS+=(--num_heads "${NUM_HEADS}")
    (( HAS_NUM_LAYERS )) && ARGS+=(--num_layers "${NUM_LAYERS}")
    (( HAS_DROPOUT )) && ARGS+=(--dropout "${DROPOUT}")
    (( HAS_DEVICE )) && ARGS+=(--device "${DEVICE}")
    (( HAS_ACTOR_DEVICE )) && ARGS+=(--actor_device "${ACTOR_DEVICE}")
    (( HAS_NUM_WORKERS )) && ARGS+=(--num_workers "${NUM_WORKERS}")
    (( HAS_NUM_THREADS )) && ARGS+=(--num_threads "${NUM_THREADS}")
    (( HAS_CPU_THREADS )) && ARGS+=(--cpu_threads "${CPU_THREADS}")
    (( HAS_BUFFER_SIZE )) && ARGS+=(--buffer_size "${BUFFER_SIZE}")
    (( HAS_LEARN_BATCH_SIZE )) && ARGS+=(--learn_batch_size "${LEARN_BATCH_SIZE}")
    (( HAS_LEARN_STEPS )) && ARGS+=(--learn_steps "${LEARN_STEPS}")
    (( HAS_BASELINE_BETA )) && ARGS+=(--baseline_beta "${BASELINE_BETA}")
    (( HAS_LOG_INTERVAL )) && ARGS+=(--log_interval "${LOG_INTERVAL}")
    (( HAS_PROGRESS_INTERVAL )) && ARGS+=(--progress_interval "${PROGRESS_INTERVAL}")
    (( HAS_SAVE_INTERVAL )) && ARGS+=(--save_interval "${SAVE_INTERVAL}")
    (( HAS_MAX_STEPS )) && ARGS+=(--max_steps "${MAX_STEPS}")
    (( HAS_UNROLL_LENGTH )) && ARGS+=(--unroll_length "${UNROLL_LENGTH}")
    (( HAS_NUM_BUFFERS )) && ARGS+=(--num_buffers "${NUM_BUFFERS}")
    (( HAS_RMSPROP_ALPHA )) && ARGS+=(--rmsprop_alpha "${RMSPROP_ALPHA}")
    (( HAS_MOMENTUM )) && ARGS+=(--momentum "${MOMENTUM}")
    (( HAS_OPTIMIZER_EPS )) && ARGS+=(--optimizer_eps "${OPTIMIZER_EPS}")
  else
    ARGS=(
      --episodes "${EPISODES}"
      --name "${TASK_NAME}"
      --savedir "${SAVEDIR}"
      --objective "${OBJECTIVE}"
      --reward_scale "${REWARD_SCALE}"
      --update_mode "${UPDATE_MODE}"
      --learning_rate "${LEARNING_RATE}"
      --gamma "${GAMMA}"
      --epsilon "${EPSILON}"
      --min_epsilon "${MIN_EPSILON}"
      --epsilon_decay "${EPSILON_DECAY}"
      --weight_decay "${WEIGHT_DECAY}"
      --max_grad_norm "${MAX_GRAD_NORM}"
      --hidden_dim "${HIDDEN_DIM}"
      --num_heads "${NUM_HEADS}"
      --num_layers "${NUM_LAYERS}"
      --dropout "${DROPOUT}"
      --device "${DEVICE}"
      --actor_device "${ACTOR_DEVICE}"
      --num_workers "${NUM_WORKERS}"
      --num_threads "${NUM_THREADS}"
      --cpu_threads "${CPU_THREADS}"
      --buffer_size "${BUFFER_SIZE}"
      --learn_batch_size "${LEARN_BATCH_SIZE}"
      --learn_steps "${LEARN_STEPS}"
      --baseline_beta "${BASELINE_BETA}"
      --log_interval "${LOG_INTERVAL}"
      --progress_interval "${PROGRESS_INTERVAL}"
      --save_interval "${SAVE_INTERVAL}"
      --max_steps "${MAX_STEPS}"
      --unroll_length "${UNROLL_LENGTH}"
      --num_buffers "${NUM_BUFFERS}"
      --rmsprop_alpha "${RMSPROP_ALPHA}"
      --momentum "${MOMENTUM}"
      --optimizer_eps "${OPTIMIZER_EPS}"
    )
  fi
  if [[ "${REWARD_SHAPING}" == "1" || "${REWARD_SHAPING}" == "true" || "${REWARD_SHAPING}" == "yes" ]]; then
    ARGS+=(--reward_shaping)
  fi
  [[ -n "${LOAD}" ]] && ARGS+=(--load "${LOAD}")
  [[ -n "${OUTPUT}" ]] && ARGS+=(--output "${OUTPUT}")
  if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
    ARGS+=(--resume)
  fi
  cd "${REPO_ROOT}"
  if [[ "${NICE}" != "0" ]]; then
    /usr/bin/time -v nice -n "${NICE}" python3 src/train_attention_dou.py "${ARGS[@]}"
  else
    /usr/bin/time -v python3 src/train_attention_dou.py "${ARGS[@]}"
  fi
} 2>&1 | tee "${TIME_LOG}"
