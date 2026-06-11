# 训练与评测命令速查

建议所有命令都在仓库根目录执行：

```bash
cd /data/sea_disk0/qianlei/Codes/Fight-the-Landlord
```

本项目现在同时保留三类 agent：

- `qlearning`：tabular Q-learning，checkpoint 通常是 `qlearning_checkpoints/.../*.pkl`。
- `approxq`：feature-based approximate Q-learning，checkpoint 通常是 `approx_qlearning_checkpoints/.../*.pkl`。
- `douzero`：原始 DouZero DMC，三种座位分别使用 `landlord_weights_*.ckpt`、`landlord_up_weights_*.ckpt`、`landlord_down_weights_*.ckpt`。

## 训练示例

Tabular Q-learning：

```bash
TASK_NAME=qlearning_logadp_100k \
EPISODES=100000 \
OBJECTIVE=logadp \
REWARD_SCALE=10 \
REWARD_SHAPING=1 \
LOG_INTERVAL=10000 \
SAVE_INTERVAL=50000 \
./src/train_qlearning_gpu.sh
```

Approx Q-learning：

```bash
TASK_NAME=approxq_logadp_cmp_1m_history \
EPISODES=1000000 \
FEATURE_MODE=history \
OBJECTIVE=logadp \
REWARD_SCALE=1 \
REWARD_SHAPING=0 \
DEVICE=cpu \
LOG_INTERVAL=10000 \
PROGRESS_INTERVAL=5000 \
SAVE_INTERVAL=50000 \
./src/train_approx_qlearning_gpu.sh
```

从最新 checkpoint 继续训练：

```bash
TASK_NAME=approxq_logadp_cmp_1m_history \
EPISODES=200000 \
RESUME=1 \
./src/train_approx_qlearning_gpu.sh
```

从明确 checkpoint 继续训练：

```bash
PYTHONPATH=src python src/train_approx_qlearning.py \
  --name approxq_logadp_cmp_1m_history \
  --episodes 200000 \
  --load approx_qlearning_checkpoints/approx_qlearning/approxq_logadp_cmp_1m_history/1000000.pkl \
  --objective logadp \
  --feature_mode history \
  --device cpu
```

注意：训练入口里的 `--episodes` 表示“本次再跑多少局”，不是“最终总局数”。

## 评测方法写法

`src/evaluate.py --methods A B C` 需要传 3 个方法。`fixed` 模式中 `A` 固定做地主，`B/C` 做农民；`rotate` 模式中三者轮流做地主。

常用方法名：

```text
rlcard                         RLCard 规则 agent
random                         随机 agent
mdp                            Bayesian MDP agent
adv                            adversarial search agent
qlearning                      默认或最新 tabular Q-learning checkpoint
qlearning:任务名               任务目录下最新 tabular checkpoint
qlearning:/path/model.pkl      明确 tabular checkpoint
approxq                        默认或最新 approximate Q checkpoint
approxq:任务名                 任务目录下最新 approximate Q checkpoint
approxq:/path/model.pkl        明确 approximate Q checkpoint
douzero:/path/landlord_weights_60000000.ckpt
                               DouZero DMC checkpoint；rotate 时会自动按座位解析同目录下
                               landlord_up_weights_60000000.ckpt 和
                               landlord_down_weights_60000000.ckpt
```

## 评测完整示例

ApproxQ vs rlcard，三组座位轮换：

```bash
mkdir -p run_logs evaluate_results

/bin/time -v python3 src/evaluate.py \
  --eval_mode rotate \
  --methods \
    approxq:/data/sea_disk0/qianlei/Codes/Fight-the-Landlord/approx_qlearning_checkpoints/approx_qlearning/approxq_logadp_cmp_1m_history/1000000.pkl \
    rlcard \
    rlcard \
  --eval_data /data/sea_disk0/qianlei/Codes/Fight-the-Landlord/eval_data.pkl \
  --num_workers 5 \
  --assignment_workers 3 \
  --evaluate_name approxq_logadp_cmp_1m_history_vs_rlcard_rotate \
  --result_dir evaluate_results \
  2>&1 | tee run_logs/approxq_logadp_cmp_1m_history_vs_rlcard_rotate.log
```

DouZero vs rlcard，三组座位轮换：

```bash
mkdir -p run_logs evaluate_results

/bin/time -v python3 src/evaluate.py \
  --eval_mode rotate \
  --methods \
    douzero:/data/sea_disk0/qianlei/Codes/Fight-the-Landlord/base/douzero_checkpoints/douzero_logadp_cmp_60000000/landlord_weights_60000000.ckpt \
    rlcard \
    rlcard \
  --eval_data /data/sea_disk0/qianlei/Codes/Fight-the-Landlord/eval_data.pkl \
  --num_workers 5 \
  --assignment_workers 3 \
  --evaluate_name douzero_logadp_cmp_60m_vs_rlcard_rotate \
  --result_dir evaluate_results \
  2>&1 | tee run_logs/douzero_logadp_cmp_60m_vs_rlcard_rotate.log
```

固定座位评测：

```bash
python3 src/evaluate.py \
  --eval_mode fixed \
  --methods mdp qlearning:qlearning_logadp_100k rlcard \
  --eval_data eval_data.pkl \
  --num_workers 5 \
  --evaluate_name fixed_mdp_qlearning_rlcard \
  --result_dir evaluate_results
```

## Evaluate 每个选项怎么填

`--methods A B C`

必须传 3 个方法，是最推荐的写法。例子：

```bash
--methods approxq:approxq_logadp_cmp_1m_history rlcard rlcard
```

`--eval_mode fixed|rotate`

`fixed` 固定座位；`rotate` 三个方法轮流做地主。例子：

```bash
--eval_mode rotate
```

`--evaluate_name NAME`

结果 JSON 文件名，不要带 `.json`。例子：

```bash
--evaluate_name douzero_logadp_cmp_60m_vs_rlcard_rotate
```

`--result_dir DIR`

结果 JSON 保存目录。例子：

```bash
--result_dir evaluate_results
```

`--eval_data FILE`

评测牌局数据，一般用仓库根目录的 `eval_data.pkl`。例子：

```bash
--eval_data /data/sea_disk0/qianlei/Codes/Fight-the-Landlord/eval_data.pkl
```

`--num_workers N`

每一组座位组合内部并行切分多少份牌局；原版 base eval 也是这一层并行。例子：

```bash
--num_workers 5
```

`--assignment_workers N`

并行跑多少组座位组合。`rotate` 最多 3 组；`--assignment_workers 3 --num_workers 5` 最多会起 15 个评测 worker。例子：

```bash
--assignment_workers 3
```

如果内存压力大，可以用：

```bash
--assignment_workers 2
```

`--gpu_device GPU`

设置 `CUDA_VISIBLE_DEVICES`。空字符串表示不额外限制。例子：

```bash
--gpu_device 0
```

旧式座位参数仍可用，但一般不推荐：

```bash
--landlord mdp --landlord_up rlcard --landlord_down random
```

## 结果文件怎么看

结果保存到：

```text
evaluate_results/<evaluate_name>.json
```

JSON 顶层会有中文 `说明`。`assignments` 里每一项是一组座位组合，并用空行分隔。每组会包含：

```text
轮换编号                 第几组 rotate
中文说明                 谁是地主，谁是两个农民
roles                    规范化后的方法名
resolved_roles           实际加载的 checkpoint，DouZero 会展开成对应座位权重
landlord_win_rate        该组中地主阵营胜率
farmer_win_rate          该组中两个农民阵营合计胜率
landlord_adp             地主阵营平均分差
farmer_adp               农民阵营平均分差，已按两个农民合计口径输出
player_stats             每个方法在该组里的胜率统计
summary                  所有组合汇总后的 overall/landlord/farmer 胜率
```

`evaluate_results/*.json` 已允许进入 git；`run_logs/*.log` 仍然默认忽略。
