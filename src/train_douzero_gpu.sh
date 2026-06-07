#!/usr/bin/env bash
set -euo pipefail

#note: 这个脚本用于跑 base/ 里的原版 DouZero，默认和 approx_qlearning 的 100w 局做训练量对齐。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="${REPO_ROOT}/base"

#note: DouZero 用 total_frames 控制训练量；这里默认用 EPISODES * STEP_MULTIPLIER 近似对齐 approx 的 step。
GPU_DEVICE="${GPU_DEVICE:-0}"
EPISODES="${EPISODES:-1000000}"
STEP_MULTIPLIER="${STEP_MULTIPLIER:-60}"
TOTAL_FRAMES="${TOTAL_FRAMES:-$((EPISODES * STEP_MULTIPLIER))}"
TASK_NAME="${TASK_NAME:-douzero_logadp_cmp_${TOTAL_FRAMES}}"

#note: reward 默认使用 logadp，方便和 approx_qlearning 的 OBJECTIVE=logadp 对照。
OBJECTIVE="${OBJECTIVE:-logadp}"
SAVEDIR="${SAVEDIR:-${BASE_DIR}/douzero_checkpoints}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/run_logs}"
TIME_LOG="${TIME_LOG:-${RUN_LOG_DIR}/${TASK_NAME}.time.log}"

#note: 原版 DouZero 的 checkpoint interval 单位是分钟，不是 episode/frame；默认设大，避免频繁保存影响速度对照。
SAVE_INTERVAL_MINUTES="${SAVE_INTERVAL_MINUTES:-999999}"

#note: 原版 DouZero 的采样和学习并行参数，可按机器 CPU/GPU 资源调整。
GPU_DEVICES="${GPU_DEVICES:-${GPU_DEVICE}}"
TRAINING_DEVICE="${TRAINING_DEVICE:-0}"
NUM_ACTOR_DEVICES="${NUM_ACTOR_DEVICES:-1}"
NUM_ACTORS="${NUM_ACTORS:-5}"
NUM_THREADS="${NUM_THREADS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
UNROLL_LENGTH="${UNROLL_LENGTH:-100}"
EXP_EPSILON="${EXP_EPSILON:-0.01}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"

#note: 可选开关。LOAD_MODEL=1 从同名 checkpoint 恢复；ACTOR_DEVICE_CPU=1 让 actor 用 CPU。
LOAD_MODEL="${LOAD_MODEL:-0}"
ACTOR_DEVICE_CPU="${ACTOR_DEVICE_CPU:-0}"
DISABLE_CHECKPOINT="${DISABLE_CHECKPOINT:-0}"

mkdir -p "${RUN_LOG_DIR}" "${SAVEDIR}"

ARGS=(
  --xpid "${TASK_NAME}"
  --objective "${OBJECTIVE}"
  --gpu_devices "${GPU_DEVICES}"
  --training_device "${TRAINING_DEVICE}"
  --num_actor_devices "${NUM_ACTOR_DEVICES}"
  --num_actors "${NUM_ACTORS}"
  --num_threads "${NUM_THREADS}"
  --total_frames "${TOTAL_FRAMES}"
  --save_interval "${SAVE_INTERVAL_MINUTES}"
  --savedir "${SAVEDIR}"
  --batch_size "${BATCH_SIZE}"
  --unroll_length "${UNROLL_LENGTH}"
  --exp_epsilon "${EXP_EPSILON}"
  --learning_rate "${LEARNING_RATE}"
)

if [[ "${LOAD_MODEL}" == "1" || "${LOAD_MODEL}" == "true" || "${LOAD_MODEL}" == "yes" ]]; then
  ARGS+=(--load_model)
fi

if [[ "${ACTOR_DEVICE_CPU}" == "1" || "${ACTOR_DEVICE_CPU}" == "true" || "${ACTOR_DEVICE_CPU}" == "yes" ]]; then
  ARGS+=(--actor_device_cpu)
fi

if [[ "${DISABLE_CHECKPOINT}" == "1" || "${DISABLE_CHECKPOINT}" == "true" || "${DISABLE_CHECKPOINT}" == "yes" ]]; then
  ARGS+=(--disable_checkpoint)
fi

{
  echo "DouZero training log: ${TIME_LOG}"
  echo "Task=${TASK_NAME} Objective=${OBJECTIVE} EpisodesRef=${EPISODES} StepMultiplier=${STEP_MULTIPLIER} TotalFrames=${TOTAL_FRAMES}"
  echo "GPUDevices=${GPU_DEVICES} TrainingDevice=${TRAINING_DEVICE} NumActorDevices=${NUM_ACTOR_DEVICES} NumActors=${NUM_ACTORS} NumThreads=${NUM_THREADS}"
  echo "BatchSize=${BATCH_SIZE} UnrollLength=${UNROLL_LENGTH} ExpEpsilon=${EXP_EPSILON} LearningRate=${LEARNING_RATE} SaveIntervalMinutes=${SAVE_INTERVAL_MINUTES}"
  echo "Note: original DouZero logs about every 5 seconds; save_interval is measured in minutes, not frames."
  cd "${BASE_DIR}"
  /usr/bin/time -v python train.py "${ARGS[@]}"
} 2>&1 | tee "${TIME_LOG}"
