#!/usr/bin/env bash
set -euo pipefail

#TODO: DouZero 原始特征版线性 ApproxQ，支持 TD 和 Monte Carlo 两种更新模式。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HAS_TASK_NAME=0; [[ -v TASK_NAME ]] && HAS_TASK_NAME=1
HAS_EPISODES=0; [[ -v EPISODES ]] && HAS_EPISODES=1
HAS_SAVEDIR=0; [[ -v SAVEDIR ]] && HAS_SAVEDIR=1
HAS_OBJECTIVE=0; [[ -v OBJECTIVE ]] && HAS_OBJECTIVE=1
HAS_REWARD_SCALE=0; [[ -v REWARD_SCALE ]] && HAS_REWARD_SCALE=1
HAS_UPDATE_MODE=0; [[ -v UPDATE_MODE ]] && HAS_UPDATE_MODE=1
HAS_ALPHA=0; [[ -v ALPHA ]] && HAS_ALPHA=1
HAS_GAMMA=0; [[ -v GAMMA ]] && HAS_GAMMA=1
HAS_EPSILON=0; [[ -v EPSILON ]] && HAS_EPSILON=1
HAS_MIN_EPSILON=0; [[ -v MIN_EPSILON ]] && HAS_MIN_EPSILON=1
HAS_EPSILON_DECAY=0; [[ -v EPSILON_DECAY ]] && HAS_EPSILON_DECAY=1
HAS_L2=0; [[ -v L2 ]] && HAS_L2=1
HAS_CLIP_TD=0; [[ -v CLIP_TD ]] && HAS_CLIP_TD=1
HAS_DEVICE=0; [[ -v DEVICE ]] && HAS_DEVICE=1
HAS_NUM_WORKERS=0; [[ -v NUM_WORKERS ]] && HAS_NUM_WORKERS=1
HAS_WORKER_EPISODES=0; [[ -v WORKER_EPISODES ]] && HAS_WORKER_EPISODES=1
HAS_CPU_THREADS=0; [[ -v CPU_THREADS ]] && HAS_CPU_THREADS=1
HAS_BUFFER_SIZE=0; [[ -v BUFFER_SIZE ]] && HAS_BUFFER_SIZE=1
HAS_LEARN_BATCH_SIZE=0; [[ -v LEARN_BATCH_SIZE ]] && HAS_LEARN_BATCH_SIZE=1
HAS_LEARN_STEPS=0; [[ -v LEARN_STEPS ]] && HAS_LEARN_STEPS=1
HAS_BASELINE_BETA=0; [[ -v BASELINE_BETA ]] && HAS_BASELINE_BETA=1
HAS_LOG_INTERVAL=0; [[ -v LOG_INTERVAL ]] && HAS_LOG_INTERVAL=1
HAS_PROGRESS_INTERVAL=0; [[ -v PROGRESS_INTERVAL ]] && HAS_PROGRESS_INTERVAL=1
HAS_SAVE_INTERVAL=0; [[ -v SAVE_INTERVAL ]] && HAS_SAVE_INTERVAL=1
HAS_DIAG_TOPK=0; [[ -v DIAG_TOPK ]] && HAS_DIAG_TOPK=1
HAS_MAX_STEPS=0; [[ -v MAX_STEPS ]] && HAS_MAX_STEPS=1

CONFIG="${CONFIG:-}"
TASK_NAME="${TASK_NAME:-approx_doufeature_logadp_td}"
if [[ -n "${CONFIG}" && "${HAS_TASK_NAME}" -eq 0 ]]; then
  TASK_NAME="${CONFIG}"
fi
EPISODES="${EPISODES:-100000}"
SAVEDIR="${SAVEDIR:-approx_qlearning_checkpoints/approx_doufeature}"
GPU_DEVICE="${GPU_DEVICE:-0}"
DEVICE="${DEVICE:-auto}"
OBJECTIVE="${OBJECTIVE:-logadp}"
REWARD_SCALE="${REWARD_SCALE:-1}"
REWARD_SHAPING="${REWARD_SHAPING:-0}"
UPDATE_MODE="${UPDATE_MODE:-td}"
ALPHA="${ALPHA:-0.01}"
GAMMA="${GAMMA:-0.98}"
EPSILON="${EPSILON:-0.1}"
MIN_EPSILON="${MIN_EPSILON:-0.02}"
EPSILON_DECAY="${EPSILON_DECAY:-0.99998}"
L2="${L2:-0.00001}"
CLIP_TD="${CLIP_TD:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WORKER_EPISODES="${WORKER_EPISODES:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50000}"
DIAG_TOPK="${DIAG_TOPK:-20}"
MAX_STEPS="${MAX_STEPS:-1000}"
CPU_THREADS="${CPU_THREADS:-1}"
BUFFER_SIZE="${BUFFER_SIZE:-0}"
LEARN_BATCH_SIZE="${LEARN_BATCH_SIZE:-4096}"
LEARN_STEPS="${LEARN_STEPS:-1}"
BASELINE_BETA="${BASELINE_BETA:-0.01}"
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
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREADS}}"

mkdir -p "${RUN_LOG_DIR}"
{
  if [[ -n "${CONFIG}" ]]; then
    echo "Task=${TASK_NAME} Config=${CONFIG}"
    echo "Config controls training parameters; explicitly provided env values can override it."
  else
    echo "Task=${TASK_NAME} Episodes=${EPISODES} UpdateMode=${UPDATE_MODE} Objective=${OBJECTIVE}"
    echo "Alpha=${ALPHA} Gamma=${GAMMA} Epsilon=${EPSILON} Workers=${NUM_WORKERS} WorkerEpisodes=${WORKER_EPISODES} Device=${DEVICE}"
  fi
  echo "CPU thread caps: OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS} NICE=${NICE}"
  echo "Replay: BUFFER_SIZE=${BUFFER_SIZE} LEARN_BATCH_SIZE=${LEARN_BATCH_SIZE} LEARN_STEPS=${LEARN_STEPS} BASELINE_BETA=${BASELINE_BETA}"
  echo "Diagnostics: ${SAVEDIR}/${TASK_NAME}/feature_diagnostics.csv"
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
    (( HAS_ALPHA )) && ARGS+=(--alpha "${ALPHA}")
    (( HAS_GAMMA )) && ARGS+=(--gamma "${GAMMA}")
    (( HAS_EPSILON )) && ARGS+=(--epsilon "${EPSILON}")
    (( HAS_MIN_EPSILON )) && ARGS+=(--min_epsilon "${MIN_EPSILON}")
    (( HAS_EPSILON_DECAY )) && ARGS+=(--epsilon_decay "${EPSILON_DECAY}")
    (( HAS_L2 )) && ARGS+=(--l2 "${L2}")
    (( HAS_CLIP_TD )) && ARGS+=(--clip_td "${CLIP_TD}")
    (( HAS_DEVICE )) && ARGS+=(--device "${DEVICE}")
    (( HAS_NUM_WORKERS )) && ARGS+=(--num_workers "${NUM_WORKERS}")
    (( HAS_WORKER_EPISODES )) && ARGS+=(--worker_episodes "${WORKER_EPISODES}")
    (( HAS_CPU_THREADS )) && ARGS+=(--cpu_threads "${CPU_THREADS}")
    (( HAS_BUFFER_SIZE )) && ARGS+=(--buffer_size "${BUFFER_SIZE}")
    (( HAS_LEARN_BATCH_SIZE )) && ARGS+=(--learn_batch_size "${LEARN_BATCH_SIZE}")
    (( HAS_LEARN_STEPS )) && ARGS+=(--learn_steps "${LEARN_STEPS}")
    (( HAS_BASELINE_BETA )) && ARGS+=(--baseline_beta "${BASELINE_BETA}")
    (( HAS_LOG_INTERVAL )) && ARGS+=(--log_interval "${LOG_INTERVAL}")
    (( HAS_PROGRESS_INTERVAL )) && ARGS+=(--progress_interval "${PROGRESS_INTERVAL}")
    (( HAS_SAVE_INTERVAL )) && ARGS+=(--save_interval "${SAVE_INTERVAL}")
    (( HAS_DIAG_TOPK )) && ARGS+=(--diag_topk "${DIAG_TOPK}")
    (( HAS_MAX_STEPS )) && ARGS+=(--max_steps "${MAX_STEPS}")
  else
    ARGS=(
      --episodes "${EPISODES}"
      --name "${TASK_NAME}"
      --savedir "${SAVEDIR}"
      --objective "${OBJECTIVE}"
      --reward_scale "${REWARD_SCALE}"
      --update_mode "${UPDATE_MODE}"
      --alpha "${ALPHA}"
      --gamma "${GAMMA}"
      --epsilon "${EPSILON}"
      --min_epsilon "${MIN_EPSILON}"
      --epsilon_decay "${EPSILON_DECAY}"
      --l2 "${L2}"
      --clip_td "${CLIP_TD}"
      --device "${DEVICE}"
      --num_workers "${NUM_WORKERS}"
      --worker_episodes "${WORKER_EPISODES}"
      --cpu_threads "${CPU_THREADS}"
      --buffer_size "${BUFFER_SIZE}"
      --learn_batch_size "${LEARN_BATCH_SIZE}"
      --learn_steps "${LEARN_STEPS}"
      --baseline_beta "${BASELINE_BETA}"
      --log_interval "${LOG_INTERVAL}"
      --progress_interval "${PROGRESS_INTERVAL}"
      --save_interval "${SAVE_INTERVAL}"
      --diag_topk "${DIAG_TOPK}"
      --max_steps "${MAX_STEPS}"
    )
  fi
  if [[ "${REWARD_SHAPING}" == "1" || "${REWARD_SHAPING}" == "true" || "${REWARD_SHAPING}" == "yes" ]]; then
    ARGS+=(--reward_shaping)
  fi
  if [[ -n "${LOAD}" ]]; then
    ARGS+=(--load "${LOAD}")
  fi
  if [[ -n "${OUTPUT}" ]]; then
    ARGS+=(--output "${OUTPUT}")
  fi
  if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
    ARGS+=(--resume)
  fi
  cd "${REPO_ROOT}"
  if [[ "${NICE}" != "0" ]]; then
    /usr/bin/time -v nice -n "${NICE}" python3 src/train_approx_doufeature.py "${ARGS[@]}"
  else
    /usr/bin/time -v python3 src/train_approx_doufeature.py "${ARGS[@]}"
  fi
} 2>&1 | tee "${TIME_LOG}"
