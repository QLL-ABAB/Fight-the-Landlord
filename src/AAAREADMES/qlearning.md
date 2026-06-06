# Q-learning 和后续优化改动说明

`src/` 是我们在原始 DouZero 基础上改出的实验版本。核心方向是移除原 DMC 训练依赖，加入自己的 tabular Q-learning，并优化规则、训练脚本和评测统计。

## 主要变化

- 删除原 DMC 训练模块，不再使用 `douzero/dmc/`。
- 新增 tabular Q-learning 自博弈训练。
- 新增 Q-learning 评测 agent。
- 修改评测逻辑，支持固定地主和三方轮流做地主。
- 修改合法牌规则：取消四带一、四带二、四带两对，只保留纯炸弹。
- 新增训练/评测使用文档和 `.gitignore`，避免提交大量 checkpoint。

## 新增文件

- `src/douzero/rl/__init__.py`  
  导出 Q-learning 的参数解析器和训练函数。

- `src/douzero/rl/arguments.py`  
  Q-learning 训练参数，包括 `--episodes`、`--name`、`--resume`、`--alpha`、`--gamma`、`--epsilon`、`--savedir` 等。

- `src/douzero/rl/qlearning.py`  
  Q-learning 核心实现：
  - Q 表 `QTable`
  - 状态抽象 `make_state_key`
  - 动作 key 编码
  - 自博弈 agent
  - Q-learning 更新公式
  - checkpoint 保存与 resume
  - 按 `qlearning_checkpoints/qlearning/{task_name}/{episodes}.pkl` 保存模型

- `src/train_qlearning.py`  
  Python 训练入口。

- `src/train_qlearning_gpu.sh`  
  训练启动脚本。支持：
  - `TASK_NAME`
  - `EPISODES`
  - `RESUME=1`
  - `GPU_DEVICE`
  - `SAVE_INTERVAL`

  注意：当前 Q-learning 是表格方法，主要仍跑 CPU；脚本只是设置 CUDA 可见性并打印 CUDA 状态。默认中间 checkpoint 每 `50000` 局保存一次，频率是原来 `1000` 局一次的 `1/50`。

- `src/douzero/evaluation/qlearning_agent.py`  
  Q-learning 评测 agent。支持：
  - `qlearning`
  - `qlearning:task_name`
  - `qlearning:path/to/model.pkl`

- `src/AAAtrain_eval.md`  
  训练和评测命令示例文档。

- `.gitignore`  
  忽略 checkpoint、pickle、评测结果、Python 缓存等生成物，防止大文件上传 GitHub。

## 修改文件

- `src/evaluate.py`  
  新增评测参数：
  - `--methods`
  - `--eval_mode fixed|rotate`
  - `--evaluate_name`
  - `--result_dir`

  默认方法改为：

  ```python
  ["mdp", "qlearning", "rlcard"]
  ```

- `src/douzero/evaluation/simulation.py`  
  评测统计升级：
  - `fixed`：第一个方法固定做地主。
  - `rotate`：三个方法轮流做地主。
  - 保存 JSON 结果到 `evaluate_results/{evaluate_name}.json`。
  - 统计每个方法的：
    - 总胜率 `overall_win_rate`
    - 做地主胜率 `landlord_win_rate`
    - 做农民胜率 `farmer_win_rate`

- `src/douzero/env/__init__.py`  
  改成懒加载 `Env`，避免只导入 `GameEnv` 时提前触发完整 `env.py` 依赖。

- `src/douzero/env/move_detector.py`  
  取消四带牌识别；同时修正 `33334` 这类四带一不再被误判成三带二。

- `src/douzero/env/move_generator.py`  
  不再生成四带二、四带两对等动作。

- `src/douzero/env/game.py`  
  删除旧四带牌型的跟牌分支。

- `src/douzero/env/move_selector.py`  
  删除四带牌型的筛选函数。

- `src/douzero/env/utils.py`  
  删除四带牌型常量。

## 常用命令

新开训练：

```bash
TASK_NAME=qlearning_wp_10k EPISODES=10000 ./src/train_qlearning_gpu.sh
```

继续训练：

```bash
TASK_NAME=qlearning_wp_10k EPISODES=5000 RESUME=1 ./src/train_qlearning_gpu.sh
```

固定地主评测：

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp qlearning:qlearning_wp_10k rlcard \
  --eval_mode fixed \
  --evaluate_name fixed_mdp_qlearning_rlcard \
  --eval_data src/eval_data.pkl
```

轮流地主评测：

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods mdp qlearning:qlearning_wp_10k rlcard \
  --eval_mode rotate \
  --evaluate_name rotate_mdp_qlearning_rlcard \
  --eval_data src/eval_data.pkl
```
