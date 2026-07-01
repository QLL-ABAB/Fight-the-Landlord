# ApproxQ 实验思路总结

## 1. 初步尝试：人工特征 ApproxQ，训练震荡明显

第一版 ApproxQ 使用线性状态动作价值函数：

```text
Q_position(s, a) = w_position · feature(s, a)
```

每个位置各有一套权重：`landlord`、`landlord_up`、`landlord_down`。动作选择使用 epsilon-greedy，训练时用 TD target 更新对应位置的线性权重。

人工特征主要分为两类：

- `compact`：使用较少的摘要特征，例如手牌数量、剩余牌数量、炸弹数、候选动作类型、动作大小、是否过牌等。
- `history`：在 `compact` 基础上加入公共出牌历史，让模型能看到局势变化，但仍然是线性模型。

观察到的问题：

- 胜率曲线大幅震荡，checkpoint 之间表现不稳定。
- 地主位置偶尔能达到较高胜率，但难以保持，微调后也容易掉下来。
- 农民位置明显弱，说明模型学到的策略有较强的地主偏置。
- 说明人工摘要特征表达能力不足，线性模型很难稳定刻画斗地主中的组合牌型、协作关系和长程收益。

## 2. 尝试 DouZero 特征：对齐强基线的输入表示

第二版 `approx_doufeature` 保留线性 ApproxQ 框架，但把特征设计改成严格复用 DouZero 的输入：

```text
feature(s, a) = concat(x_batch, flatten(z_batch))
```

其中：

- `x_batch`：当前手牌、其他玩家牌信息、上一次动作、各玩家已出牌、剩余牌数、炸弹数、候选动作等。
- `z_batch`：最近若干轮公共出牌历史，用 DouZero 的历史编码方式展开。
- 地主特征维度为 `1183`，农民特征维度为 `1294`。

具体维度构成：

```text
一组牌面编码 = 54 维
= 13 个普通点数 * 4 张 + 大小王 2 张
= 52 + 2
```

地主 `landlord` 的特征：

```text
x_batch = 373 维
z_batch = 5 * 162 = 810 维
总维度 = 373 + 810 = 1183
```

其中地主的 `x_batch=373` 来自：

```text
my_hand                 54  当前地主手牌
other_hand              54  两个农民合并后的未知/对手牌信息
last_action             54  上一个动作
landlord_up_played      54  地主上家已经出过的牌
landlord_down_played    54  地主下家已经出过的牌
landlord_up_left        17  地主上家剩余牌数 one-hot
landlord_down_left      17  地主下家剩余牌数 one-hot
bomb_num                15  当前炸弹数量 one-hot
action                  54  当前候选动作
合计                   373
```

农民 `landlord_up / landlord_down` 的特征：

```text
x_batch = 484 维
z_batch = 5 * 162 = 810 维
总维度 = 484 + 810 = 1294
```

其中农民的 `x_batch=484` 来自：

```text
my_hand                 54  当前农民手牌
other_hand              54  其他玩家牌信息
landlord_played         54  地主已经出过的牌
teammate_played         54  队友农民已经出过的牌
last_action             54  上一个动作
last_landlord_action    54  地主上一次动作
last_teammate_action    54  队友农民上一次动作
landlord_left           20  地主剩余牌数 one-hot
teammate_left           17  队友剩余牌数 one-hot
bomb_num                15  当前炸弹数量 one-hot
action                  54  当前候选动作
合计                   484
```

`z_batch=810` 来自最近公共出牌历史：

```text
5 组历史块 * 3 个玩家动作 * 54 维牌面编码
= 5 * 3 * 54
= 810
```

直观理解：

- `x_batch` 描述当前局面和当前候选动作。
- `z_batch` 描述最近出牌历史。
- `approx_doufeature` 把这两部分直接拼成一个大向量，然后用线性权重 `w_position` 计算 `Q(s,a)`。

效果和现象：

- 使用 DouZero 特征后，TD 版本明显强于旧人工特征版本，说明特征表达是原 ApproxQ 失败的重要原因。
- 但模型仍然是线性的，只能学习每个特征的加权和，无法像 DouZero 神经网络一样建模复杂组合关系。
- MC 版本直接用终局回报回填整局动作时，容易出现策略塌缩，说明单纯终局信号对线性模型太粗糙、方差太大。
- 加入 `feature_diagnostics` 后，可以记录 TD error、`abs_delta_w` 和 top-k 权重变化，用来分析哪些 DouZero 特征导致训练抖动。

## 3. 尝试 Buffer 和 Baseline：进一步提高训练稳定性

为了降低在线更新带来的非平稳和高方差，当前版本加入 replay buffer：

```text
actor workers 并行采样 transitions
main process 写入 replay buffer
learner 从 buffer 随机抽 mini-batch
对线性 w 做多次 batch update
新 w 再同步给下一轮 actor workers
```

buffer 的作用：

- 打散连续对局中的强相关样本。
- 复用历史 transitions，提高每次采样的学习效率。
- 支持更大的 batch 更新，使 GPU/矩阵计算更充分。
- 对 TD 模式尤其有用，因为 TD target 本身仍然是合法的 Bellman target。

`learn_batch_size=4096` 和 `learn_steps=10` 的具体含义：

- `learn_batch_size=4096`：learner 每次从 replay buffer 中随机抽 4096 条 transition 组成一个 mini-batch。
- `learn_steps=10`：每完成一轮 actor 采样后，learner 连续做 10 次 mini-batch 更新。
- 所以每轮采样之后，实际用于更新的 transition 数约为：

```text
4096 * 10 = 40960
```

举例说明：

```text
num_workers = 4
worker_episodes = 8
learn_batch_size = 4096
learn_steps = 10
```

这表示一轮训练中：

```text
4 个 worker 各打 8 局
=> 新采样 32 局
=> 新 transitions 被放入 replay buffer
=> learner 从 buffer 中随机抽 4096 条 transition，更新一次 w
=> 这个抽样更新过程重复 10 次
=> 一轮总共用 40960 条 transition 做学习
```

和没有 buffer 的在线 TD 相比：

```text
无 buffer：一局平均约 40 步，每个 episode 约更新 40 条 transition
有 buffer：32 局之后学习 40960 条 transition，平均每局约 1280 条
```

因此当前 buffer 配置的学习强度约为：

```text
1280 / 40 ≈ 32 倍
```

直观理解：

- `learn_batch_size` 控制每次学习“看多少条样本”。
- `learn_steps` 控制每轮采样后“重复学多少次”。
- 两者相乘决定 replay buffer 版本每轮 learner 的总学习量。
- 这会提高样本复用和训练稳定性，但也会显著增加训练时间。

baseline 的作用：

- baseline 只用于 `mc_adv` 模式，不直接用于 TD。
- MC advantage target 为：

```text
target = final_reward - baseline[position]
```

- 这样可以减小终局回报的整体偏移，避免某个位置长期负回报时把该位置所有动作都整体压低。
- 对 TD 来说，不建议直接减 baseline，因为 TD target 是：

```text
target = r + gamma * max_a Q(s', a)
```

直接减 baseline 会改变 Bellman 方程本身，可能把 Q 值学偏。

当前新增配置：

- `approx_doufeature_logadp_td_buffer_1m`：TD + DouZero 特征 + replay buffer。
- `approx_doufeature_logadp_mc_adv_buffer_1m`：MC advantage + DouZero 特征 + replay buffer + position baseline。

后续重点：

- 比较 `td_1m`、`td_buffer_1m`、`mc_1m`、`mc_adv_buffer_1m` 的地主和农民固定角色胜率曲线。
- 观察 buffer 是否降低 checkpoint 间震荡。
- 观察 baseline 是否能缓解 MC 塌缩。
- 和老师讨论线性 ApproxQ 的能力边界：继续增强稳定化技巧，还是转向非线性函数逼近。



## 4. AttentionDou 中 TD 和 MC 的区别

AttentionDou 和 DouZero 一样使用自博弈、actor/learner、buffer 和 batch learning，但把 DouZero 的 LSTM 换成了 multi-head attention。TD 和 MC 的区别主要在训练 target：

```text
TD:
  y_t = r_t + gamma * max_a Q(s_{t+1}, a)
```

TD 是边估计边学习。它不必等整局结束才能给中间动作信号，因此反馈更密集；但 target 依赖当前网络自己的估计，可能有 bootstrapping bias。

```text
MC:
  y_t = G
```

MC 是整局结束后，把最终输赢回报 `G` 回填给这一局中的动作。它的 target 更真实，不依赖下一步 Q 估计；但斗地主回合长、方差大，容易把一局中好动作和坏动作都归为同一个结果。

当前 `mc_adv` 是 MC 的改进版：

```text
y_t = G - baseline(position)
```

也就是用“这局结果比该位置平均水平好多少”来更新，能降低 MC 的方差。简单说：TD 更快给信号但可能偏，MC 更接近真实输赢但更抖，MC+baseline 是为了让终局回报训练更稳定。


---

这张图主要说明三件事：

1. **大权重特征和高抖动特征不是完全同一批东西**
   
   纵轴 Top 权重主要是：
   `action_finish`、`singles_delta`、`hand_singles`、`badness_delta`、`hand_badness`、`teammate_danger_pass`

   横轴 Top 抖动主要是：
   `played_control_cards`、`played_group_control_a_2_joker`、`played_mid_cards_ratio`、`played_group_mid_8_to_k`、`played_high_cards_ratio`、以及 `landlord_down:action_finish`

   这说明当前模型里有些特征权重大、真正在决策里被依赖；另一些特征虽然权重不一定最大，但训练中反复被更新、方向不稳。

2. **历史出牌分组特征的抖动和手牌结构价值特征呈明显负相关**

   最强的一批相关关系是负相关，例如：

   - `singles_delta` vs `played_mid_cards_ratio`: `-0.797`
   - `badness_delta` vs `played_mid_cards_ratio`: `-0.790`
   - `hand_singles` vs `played_mid_cards_ratio`: `-0.779`
   - `hand_badness` vs `played_mid_cards_ratio`: `-0.764`

   直觉上可以理解为：模型在“自己手牌结构好坏”和“历史牌面消耗情况”之间来回分摊解释权。也就是说，某些训练窗口里模型更相信手牌结构特征，另一些窗口里又把变化归因到历史出牌比例上，于是历史特征就会抖。

3. **`action_finish` 有角色间联动**

   `landlord:action_finish` 和 `landlord_down:action_finish` 相关为 `0.653`，这是图里少数明显正相关。

   这说明“能否打完/收尾”这个动作特征在不同位置上具有共同训练方向，属于比较稳定、有实际策略含义的信号。它不像历史比例特征那样只是被动解释环境变化。

**对特征改造的启示**

`played_mid_cards_ratio`、`played_group_mid_8_to_k`、`played_control_cards` 这类历史统计特征很可能不是“没用”，而是太粗，容易和 `hand_singles`、`badness_delta`、`singles_delta` 争解释权。

所以后面更适合做的是：
- 不要简单全删历史特征；
- 把历史特征改成更条件化的版本，例如按角色、上家/下家、是否地主快没牌、上一手类型、当前动作类型拆开；
- 对和手牌结构高度负相关的历史特征加约束或裁剪，避免它们反复替代手牌价值特征；
- `action_finish` 这种跨角色一致的特征可以保留，甚至细分收尾类型。


