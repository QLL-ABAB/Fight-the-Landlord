# History-aware approximate Q-learning 说明

这个版本用于解决 tabular Q-learning 的 Q 表爆炸问题。核心思想是不用字典保存每个 `(state, action)`，而是用固定维度特征近似：

```text
Q(s, a) = w · phi(s, a)
```

权重 `w` 按位置分为三组：`landlord`、`landlord_up`、`landlord_down`。因此模型大小只和特征维度有关，不会随着训练局数线性增长。

## 新增文件

- `src/douzero/rl/approx_qlearning.py`  
  线性 approximate Q-learning 核心实现，包括特征构造、动作剪枝、TD 更新、checkpoint 保存和 resume。

- `src/douzero/rl/approx_arguments.py`  
  approximate Q-learning 的训练参数。

- `src/train_approx_qlearning.py`  
  Python 训练入口。

- `src/train_approx_qlearning_gpu.sh`  
  便捷训练脚本，支持 `FEATURE_MODE`、`TASK_NAME`、`EPISODES`、`RESUME` 等环境变量。

- `src/douzero/evaluation/approx_qlearning_agent.py`  
  approximate Q-learning 评测 agent，支持 `approxq:任务名` 或 `approxq:checkpoint路径`。

## 特征开关

通过 `FEATURE_MODE` 控制特征集：

```text
FEATURE_MODE=history  默认，使用完整固定维度历史特征，230 维
FEATURE_MODE=compact  只使用摘要特征，61 维，速度更快但信息更少
```

`history` 模式不是保存变长历史列表，而是把公共历史编码成固定维度向量，主要包括：

- 自己手牌 counts：15 维。
- 三家已出牌 counts：`landlord`、`landlord_up`、`landlord_down` 各 15 维，共 45 维。
- 总已出牌 counts：15 维。
- 当前视角未见牌 counts：15 维，只表示隐藏牌集合，不记录具体归属。
- `last_move`：15 维。
- `last_two_moves`：2 x 15 维。
- 地主底牌剩余信息：15 维。
- 当前动作 counts：15 维。
- 动作类型、动作长度、是否 pass、是否炸弹、是否出完。
- 手牌结构变化、队友/敌人危险状态、炸弹使用等策略特征。

所有牌数特征都按牌面最大数量归一化，普通牌除以 4，大小王除以 1。

## 训练示例

history-aware 版本：

```bash
cd /home/ql/Desktop/Homework/CS181/Fight-the-Landlord

TASK_NAME=approxq_logadp_100k_history \
EPISODES=100000 \
FEATURE_MODE=history \
OBJECTIVE=logadp \
REWARD_SHAPING=0 \
DEVICE=cpu \
./src/train_approx_qlearning_gpu.sh
```

compact 对照版本：

```bash
TASK_NAME=approxq_logadp_100k_compact \
EPISODES=100000 \
FEATURE_MODE=compact \
OBJECTIVE=logadp \
REWARD_SHAPING=0 \
DEVICE=cpu \
./src/train_approx_qlearning_gpu.sh
```

继续训练：

```bash
TASK_NAME=approxq_logadp_100k_history \
EPISODES=50000 \
RESUME=1 \
./src/train_approx_qlearning_gpu.sh
```

## 评测示例

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp approxq:approxq_logadp_100k_history rlcard \
  --eval_mode rotate \
  --evaluate_name rotate_mdp_approxq_rlcard \
  --eval_data src/eval_data.pkl \
  --num_workers 4
```

也可以直接指定 checkpoint：

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp approxq:approx_qlearning_checkpoints/approx_qlearning/approxq_logadp_100k_history/100000.pkl rlcard \
  --eval_mode rotate \
  --evaluate_name rotate_mdp_approxq_rlcard \
  --eval_data src/eval_data.pkl
```

## 调试特征

默认不会逐步检查每个特征值，以免拖慢训练。如果要检查维度和非有限值，可以临时打开：

```bash
APPROXQ_VALIDATE_FEATURES=1 \
FEATURE_MODE=history \
EPISODES=10 \
./src/train_approx_qlearning_gpu.sh
```

## 和 tabular Q-learning 的区别

tabular Q-learning：

```text
Q[(position, state, action)] = value
```

状态越多，字典越大，内存会持续增长。

approximate Q-learning：

```text
Q(s, a) = w · phi(s, a)
```

只保存固定长度权重。`history` 模式保留了公共牌史，但仍然是固定维度，因此适合长时间训练和做课程项目分析。
