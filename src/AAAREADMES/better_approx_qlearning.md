# Better ApproxQ 说明

`better_approxq` 是在原 `approxq` 基础上的手工特征增强版，原实现不变。

主要变化：

- 默认特征模式是 `better_history`。它保留原 `compact` 和关键状态特征，但删除旧 `history` 中已被新摘要覆盖且诊断中抖动很大的逐牌面历史特征。
- 新增低/中/高/王/控制牌的分组计数摘要，降低 `total_played_*` 逐牌面特征的高方差。
- 新增地主特化：敌方低牌压力、炸弹是否收尾、底牌控制牌、地主控制牌优势。
- 新增农民特化：队友危险、地主危险、是否压队友、是否压地主、是否把控制牌留给地主。
- 新增上家/下家特化：地主上家阻断地主、地主下家接地主后是否放行/抢先手等上下文。
- 推荐更保守的默认训练参数：`alpha=0.006`、`clip_td=5`、`l2=3e-5`。

被删除的旧高抖动特征包括：

- 逐牌面 `my_hand_*`、`played_*`、`total_played_*`、`unseen_*`。
- 逐牌面 `last_move_*`、`last_two_move_*`、`landlord_bottom_*`、旧 `action_*` 牌面计数。
- 旧的 `unseen_possible_bombs`、`unseen_control_cards`、`played_control_cards`、`played_bomb_like_ranks` 汇总项。

这些信息由新增的分组计数、控制牌优势、pass/压制上下文和地主/农民特化特征覆盖。

训练示例：

```bash
cd /data/sea_disk0/qianlei/Codes/Fight-the-Landlord

PYTHONPATH=src python src/train_better_approx_qlearning.py \
  --config better_approxq_logadp_10k
```

50k 训练：

```bash
PYTHONPATH=src python src/train_better_approx_qlearning.py \
  --config better_approxq_logadp_finetune_50k_time_equal
```

学习率对照实验：

```bash
PYTHONPATH=src python src/train_better_approx_qlearning.py \
  --config better_approxq_logadp_finetune_50k_time_equal_lr_1e-1

PYTHONPATH=src python src/train_better_approx_qlearning.py \
  --config better_approxq_logadp_finetune_50k_time_equal_lr_1e-2
```

其中 `lr_1e-1` 是默认 `alpha=0.006` 的 1/10，`lr_1e-2` 是 1/100。

评测示例：

```bash
PYTHONPATH=src python src/evaluate.py \
  --methods better_approxq:better_approxq_logadp_10k rlcard rlcard \
  --eval_mode rotate \
  --evaluate_name better_approxq_10k_vs_rlcard_rotate
```

诊断分析：

```bash
PYTHONPATH=src python src/analyze_approxq_feature_diagnostics.py \
  --diag_csv approx_qlearning_checkpoints/better_approxq/better_approxq_logadp_10k/feature_diagnostics.csv \
  --output_dir visualization/approxq_feature_diagnostics \
  --prefix better_approxq_logadp_10k
```
