# Q-learning 训练与评测说明

当前项目使用 tabular Q-learning 做强化学习实验。
以下命令都建议在项目根目录执行：

```bash
cd /home/ql/Desktop/Homework/CS181/Fight-the-Landlord
```

## 训练

新建一个 Q-learning 训练任务：

```bash
TASK_NAME=qlearning_wp_10k \
EPISODES=10000 \
GPU_DEVICE=0 \
./src/train_qlearning_gpu.sh
```

模型会保存到：

```text
qlearning_checkpoints/qlearning/qlearning_wp_10k/10000.pkl
```

继续训练同一个任务：

```bash
TASK_NAME=qlearning_wp_10k \
EPISODES=5000 \
RESUME=1 \
GPU_DEVICE=0 \
./src/train_qlearning_gpu.sh
```

如果当前最新 checkpoint 是 `10000.pkl`，上面的命令会从它继续训练，并保存为：

```text
qlearning_checkpoints/qlearning/qlearning_wp_10k/15000.pkl
```

也可以直接运行 Python 训练入口：

```bash
PYTHONPATH=src python src/train_qlearning.py \
  --name qlearning_wp_10k \
  --episodes 10000 \
  --alpha 0.1 \
  --gamma 0.95 \
  --epsilon 0.2 \
  --min_epsilon 0.02 \
  --epsilon_decay 0.9995 \
  --objective wp
```

直接用 Python 入口继续训练：

```bash
PYTHONPATH=src python src/train_qlearning.py \
  --name qlearning_wp_10k \
  --episodes 5000 \
  --resume
```

注意：`train_qlearning_gpu.sh` 会设置 CUDA 可见设备并打印 CUDA 状态，但当前 Q-learning 是表格方法，核心更新逻辑主要仍然跑在 CPU 上。

## 评测

固定地主模式：第一个方法固定做地主，另外两个方法做农民。

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp qlearning:qlearning_wp_10k rlcard \
  --eval_mode fixed \
  --evaluate_name fixed_mdp_qlearning_rlcard \
  --eval_data src/eval_data.pkl \
  --num_workers 5
```

结果会保存到：

```text
evaluate_results/fixed_mdp_qlearning_rlcard.json
```

轮流地主模式：三个方法轮流做一次地主，用于排除固定地主身份对结果的影响。

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp qlearning:qlearning_wp_10k rlcard \
  --eval_mode rotate \
  --evaluate_name rotate_mdp_qlearning_rlcard \
  --eval_data src/eval_data.pkl \
  --num_workers 5
```

结果会保存到：

```text
evaluate_results/rotate_mdp_qlearning_rlcard.json
```

保存的 JSON 中会记录每个方法的以下指标：

```text
overall_win_rate   总胜率
landlord_win_rate  做地主时的胜率
farmer_win_rate    做农民时的胜率
games              总局数
wins               总胜局数
landlord_games     做地主局数
landlord_wins      做地主胜局数
farmer_games       做农民局数
farmer_wins        做农民胜局数
```

## Q-learning 模型路径写法

使用某个任务目录下最新的 checkpoint：

```bash
qlearning:qlearning_wp_10k
```

使用某个明确的 checkpoint 文件：

```bash
qlearning:qlearning_checkpoints/qlearning/qlearning_wp_10k/15000.pkl
```

使用默认或最新的 Q-learning checkpoint：

```bash
qlearning
```

## 常用完整流程

先训练：

```bash
TASK_NAME=qlearning_wp_10k EPISODES=10000 GPU_DEVICE=0 ./src/train_qlearning_gpu.sh
```

再轮流地主评测：

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp qlearning:qlearning_wp_10k rlcard \
  --eval_mode rotate \
  --evaluate_name rotate_mdp_qlearning_rlcard \
  --eval_data src/eval_data.pkl \
  --num_workers 5
```
