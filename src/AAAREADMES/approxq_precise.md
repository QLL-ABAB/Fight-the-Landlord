# ApproxQ Precise 说明

`approxq_precise` 是基于 `better_history_full` 的精细特征版本。

主要变化：

- 基于 full history，不再使用裁剪版。
- 删除粗粒度高抖特征：`action_type_0`、`action_is_pass`、`action_preserves_control`。
- 新增 pass 精细上下文：上一手是谁、上一手牌型、下一家是谁、地主/队友/自己是否危险、是否有非控制牌响应、是否只能用控制牌响应、是否有炸弹响应。
- 新增 control preserve 精细上下文：地主/农民身份、主动/跟牌、上一手关系、地主/队友/敌人危险、是否有非控制牌替代、是否必须用控制牌、是否用控制牌收尾或阻断。
- 新增手牌残余监督：去掉最佳顺子后的剩余低单牌、出顺子后的剩余低单牌、去掉最佳连对后的剩余低对子、出连对后的剩余低对子。

当前特征维度：

```text
precise_history_full = 378 维
```

它来自：

```text
better_history_full 312 维
- action_type_0
- action_is_pass
- action_preserves_control
+ 69 个 precise 新特征
```

## Warm Start

推荐先从 `history_full` 当前最好 checkpoint 继承：

```text
approx_qlearning_checkpoints/better_approxq/better_approxq_full_50k_lr_1e-1/50000.pkl
```

加载时会按特征名迁移：

- 老特征名相同：直接复制原权重。
- 被删除的粗特征：丢弃。
- 新 precise 特征：初始化为 0。

这比从头训练更适合作为第一轮实验，因为 full_history 已经学到了大量稳定基础权重，新特征只需要学习“pass/control/残余手牌”的细分修正。

## 训练

标准 warm-start，带诊断：

```bash
cd /data/sea_disk0/qianlei/Codes/Fight-the-Landlord

PYTHONPATH=src python src/train_approxq_precise.py \
  --config approxq_precise_full_50k_warm
```

快速版，不开诊断，候选动作降到 32：

```bash
PYTHONPATH=src python src/train_approxq_precise.py \
  --config approxq_precise_full_50k_warm_fast
```

从头训练对照：

```bash
PYTHONPATH=src python src/train_approxq_precise.py \
  --config approxq_precise_full_50k_scratch
```

## 评测

单点评测：

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods approxq_precise:approxq_precise_full_50k_warm rlcard rlcard \
  --eval_mode rotate \
  --evaluate_name approxq_precise_full_50k_warm_vs_rlcard_rotate
```

画 checkpoint 趋势：

```bash
PYTHONPATH=src python src/plot_eval_trends.py \
  --approxq_series precise=approxq_precise:approx_qlearning_checkpoints/approxq_precise/approxq_precise_full_50k_warm \
  --test_role landlord \
  --num_games 500 \
  --num_workers 5 \
  --skip_douzero \
  --output_dir visualization/approxq_precise_eval_trends \
  --output_prefix approxq_precise_full_50k_warm_landlord
```

农民两家同时使用 precise：

```bash
PYTHONPATH=src python src/plot_eval_trends.py \
  --approxq_series precise=approxq_precise:approx_qlearning_checkpoints/approxq_precise/approxq_precise_full_50k_warm \
  --test_role two_farmers \
  --num_games 500 \
  --num_workers 5 \
  --skip_douzero \
  --output_dir visualization/approxq_precise_eval_trends \
  --output_prefix approxq_precise_full_50k_warm_two_farmers
```

## 速度与 Buffer

当前版本保持在线 Q-learning，不引入并行 actor 或 replay buffer。

原因：

- 在线自博弈的策略分布一直在变，直接加 replay buffer 容易把旧策略样本混进当前策略，可能降低稳定性。
- 多进程 actor 需要用权重快照采样，再集中更新，训练会变成半离策略；速度会提升，但学习曲线和当前 approxq/better_approxq 不再完全可比。
- 对线性模型，最大瓶颈常常是 Python 环境和逐动作特征构造，第一优先级仍是关闭诊断、减少候选动作、减少保存频率。

建议实验顺序：

1. 先跑 `approxq_precise_full_50k_warm_fast` 看方向。
2. 如果明显优于 full/better，再做并行 actor + 小 replay buffer。
3. replay buffer 只建议先用很小容量，并按最近样本优先，避免旧策略污染。

