# base 项目结构速览

`base/` 是原始 DouZero 版本，主要包含斗地主环境、DMC 训练、评测脚本和预训练模型接口。

## 顶层文件

- `README.md` / `README.zh-CN.md`：项目说明、安装、训练、评测方法。
- `requirements.txt`：依赖，主要包括 `torch`、`rlcard`、`GitPython`。
- `setup.py`：Python 包安装配置，包名为 `douzero`。
- `train.py`：DMC 训练入口，解析参数后调用 `douzero.dmc.train`。
- `evaluate.py`：评测入口，指定地主、地主上家、地主下家的 agent。
- `generate_eval_data.py`：随机发牌，生成 `eval_data.pkl` 评测数据。
- `get_most_recent.sh`：从 checkpoint 目录中取最新三份模型权重。
- `baselines/`：预训练模型放置目录。
- `.github/workflows/python-package.yml`：GitHub Actions CI 配置。

## `douzero/env/`：斗地主环境和规则

- `game.py`：底层游戏引擎，负责手牌、轮转、胜负、得分和 `InfoSet`。
- `env.py`：RL 风格环境封装，提供 `reset/step/get_obs` 和特征编码。
- `move_generator.py`：根据手牌枚举所有可能动作。
- `move_detector.py`：判断一手牌的牌型。
- `move_selector.py`：筛选能压过上一手牌的动作。
- `utils.py`：牌型常量和组合工具。

## `douzero/dmc/`：原始 DMC 训练

- `arguments.py`：DMC 训练参数。
- `dmc.py`：主训练循环，启动 actor、learner、保存 checkpoint。
- `models.py`：地主/农民 LSTM 模型。
- `utils.py`：actor 采样、共享 buffer、optimizer 等训练工具。
- `env_utils.py`：训练环境包装，把观测转成 tensor。
- `file_writer.py`：日志和元数据保存。

## `douzero/evaluation/`：评测 agent

- `simulation.py`：多进程评测，统计地主/农民阵营胜率和 ADP。
- `deep_agent.py`：加载 `.ckpt` 深度模型进行动作选择。
- `random_agent.py`：随机合法出牌。
- `rlcard_agent.py`：基于 RLCard 规则启发式出牌。

注意：原始 `base/` 目录中没有 `mdp_agent.py`。`mdp_agent.py` 是后续在 `src/` 中新增的规则/启发式 agent。

