# Value Iteration 思路记录

Value Iteration 的基本目标是：先在离线阶段计算每个状态的价值 `V(s)`，之后实际对局时直接查表，根据后继状态价值选择动作。

标准公式是：

```text
V_{k+1}(s) = max_a sum_{s'} P(s' | s, a) [ R(s, a, s') + gamma V_k(s') ]
```

如果转移是确定的，可以写成：

```text
V_{k+1}(s) = max_a [ R(s, a, T(s, a)) + gamma V_k(T(s, a)) ]
```

实际决策时，对每个合法动作计算：

```text
Q(s, a) = R(s, a, s') + gamma V(s')
pi(s) = argmax_a Q(s, a)
```

也就是说，Value Iteration 理论上不是通过对局经验一点点更新 Q 表，而是先定义状态、动作、奖励和转移模型，然后离线反复更新 `V(s)`，直到收敛。

## 1. 目前思路：不能直接用真实手牌当状态

最初的想法是把斗地主看成一个 MDP，然后离线计算 `V(s)`。但是这里的 `s` 不能简单写成“真实手牌”，否则会有两个问题：

第一，如果 `s = 自己当前真实手牌`，那么它只描述了自己能怎么拆牌，完全没有描述当前对局环境。这样算出来的 `V(s)` 更像“这副牌自己出完要几步”，不是“这个局面赢的概率或期望分数”。

第二，如果 `s` 包含其他玩家的真实手牌，那又不符合实际对局。因为对局时我们看不到对手和队友的完整手牌，agent 不能用不可观测信息做决策。

所以更合理的状态应该是“可观测信息 + 必要的抽象/估计”，例如：

- 自己的手牌。
- 其他玩家剩余手牌数量。
- 已经出过的牌。
- 上一手牌和上一手出牌玩家。
- 当前轮到谁行动。
- 当前身份是地主、地主上家还是地主下家。
- 当前倍数和真实积分奖励。
- 对其他玩家可能手牌的概率估计，而不是真实隐藏手牌。

但是目前实现中，为了先跑通流程，我们把状态简化成：

```text
s = 当前玩家自己的手牌
```

动作就是从手牌里选出一组合法牌，后继状态是：

```text
s' = s - a
```

所以当前 `V(s)` 更接近“这副手牌自己拆起来有多快”，而不是真实对局中的胜率价值。例如在设置：

```text
terminal_reward = 100
step_reward = -1
gamma = 1
```

时，很多状态的值会接近：

```text
V(s) ~= 100 - 最少出完步数
```

这也解释了为什么 `values.json` 中很多值都在 90 多：它主要反映“离出完还差几步”，并没有真正考虑对手、队友、上一手牌、出牌权和真实积分。

结合胜率来看，当前问题是：同一副手牌在不同对局环境下价值完全不同。比如对手只剩一张牌、队友快赢、上一手需要压制、已经出过哪些大牌，这些都会改变最优动作。但当前状态只有自己的手牌，所以表里的 `V(my_hand)` 不能代表完整局面的真实 value。

因此当前版本更准确地说是一个“手牌拆解 Value 表”，可以作为辅助信息，但还不是完整斗地主意义上的 Value Iteration agent。接下来如果继续做 Value Iteration，重点应该不是继续扩大 `my_hand -> V` 这张表，而是重新设计状态表示，让 `s` 至少包含可观测的对局上下文。

## 2. Feature-state 尝试与失败原因

在发现 `V(my_hand)` 不能表达真实局面后，我们尝试把 state 从单纯手牌扩展成 feature-state。目标是：

```text
s = feature(infoset)
Q(s, a) = R(s, a, s') + gamma V(s')
pi(s) = argmax_a Q(s, a)
```

也就是说，agent 的决策仍然是纯粹选最大 `V`，不是改成启发式；只是 `V` 表里的 state key 不再是单纯手牌，而是由可观测信息构造出来的 feature。

### 2.1 第一次 feature 尝试：过于精确

最开始的 feature key 设计得比较完整，大致包含：

```text
position
my_hand_counts
last_move
last_player
num_cards_left
played_card_counts
leading_flag
```

这样从定义上更接近真实对局状态，但问题是状态空间太大。`my_hand_counts` 和 `played_card_counts` 都是 15 维计数，再加上 `last_move`、`last_player`、三家剩余牌数，组合数量会迅速爆炸。

实际评测时，大量真实对局中的 feature-state 在 `values.json` 里查不到。查不到时如果默认 `V(s') = 0`，那么：

```text
pass:      Q = pass_reward
普通出牌:   Q = step_reward
直接出完:   Q = step_reward + terminal_reward
```

当 `pass_reward = step_reward = -1` 时，pass 和普通出牌经常同分；而 tie-break 中 pass 的动作字符串为空，排序靠前，所以 agent 会表现得像“能不出就不出”。这导致胜率非常低。

### 2.2 第二次 feature 尝试：过于粗糙

为了提高命中率，我们又把 feature 大幅粗化，只保留：

```text
position
hand_structure
```

其中 `hand_structure` 包含剩余牌数分桶、单牌数量、对子数量、三张数量、炸弹数量、火箭、控制牌强度、最长顺子长度等。

这个版本命中率确实更高，但问题是它又退回到了“主要只看自己的手牌结构”。它没有真正表达：

- 当前是否需要压别人。
- 上一手牌是谁出的。
- 敌人是否只剩 1-2 张。
- 队友是否快赢。
- 已经出过哪些大牌。

所以它虽然比纯手牌字符串更抽象，但本质上仍然不是完整对局 value，胜率依然很低。

### 2.3 第三次 feature 尝试：粗上下文 + fallback

之后我们尝试了折中版本 `feat_v3`。它保留一些粗粒度公共上下文：

```text
pos      身份
hand     手牌结构摘要
lead     主动出牌 / 跟牌
last     上一手牌的粗牌型、长度分桶、大小分桶
actor    上一手来自 self / team / enemy / none
enemy    敌人最少剩余牌数分桶
team     队友剩余牌数分桶
played   已出控制牌、炸弹数、总出牌数的粗摘要
```

同时为了避免大量 miss，又额外保存了：

```text
feat_v3_base = position + hand_structure
```

agent 查表时先查完整 `feat_v3`，如果没有命中，再退到 `feat_v3_base`，最后才用 `-len(hand)` 作为 fallback。

这个设计从形式上更合理，但实际效果仍然不好。原因是：离线构建 feature 表时，我们主要是从初始发牌的 root hand 展开手牌拆解状态，再用 fake infoset 生成 feature key。这样生成出来的完整 `feat_v3` 多数类似：

```text
lead=1
last=none
actor=none
played=0,0,0
```

也就是初始主动出牌语境。真实对局中大量状态是中后期局面，例如：

```text
lead=0
last=pair/high_single/chain
actor=enemy/team
played=...
```

这些完整 `feat_v3` 在表里覆盖很少，所以实际评测时经常退到 `feat_v3_base`。一旦退到 base，就又主要是在看手牌结构。

### 2.4 根本失败原因

这次尝试失败的根本原因不是 key 格式不够漂亮，而是 `V` 的来源不对。

当前流程本质上是：

```text
hand -> reduced hand-decomposition VI -> V(hand)
feature_key -> 存 V(hand)
```

也就是说，虽然 key 里加入了 feature，但 value 本身仍然来自“这副剩余手牌多久能出完”。它没有真的根据完整 feature-state 重新计算真实对局价值。

真正想要的是：

```text
feature_state -> 根据真实对局转移和终局 reward 估计 V(feature_state)
```

但斗地主是不完全信息游戏，真实 state 不能包含对手隐藏手牌，只能使用 infoset 或 belief state。这样状态空间非常大，纯表格 value iteration 很难覆盖。

因此目前结论是：

```text
feature key 可以缓解 my_hand 过窄的问题，
但如果 V 的训练来源仍然是 hand-decomposition，
那么它不能真正学到胜率价值。
```

这也是为什么 `vi_3k_feature_v3` 的胜率仍然很低。

## 3. 放弃该方法

我们可以得知qlearning在million级别的训练轮次中表现尚且不够理想

所以更为具体需要计算V值的方法对于feature要求太高，如果用更加模糊的方法feature，去拟合，会大大提升feature数量，增大计算开销，也未必带来提升。且每一步没有及时奖励，其最后可能V值过于接近。如果说在决策时加上许多启发式的东西去拟合，那么该方法与heuristic方法过于接近时间开销却远远大于，没有明显优势，属于负优化。所以我们决定放弃了值迭代方法
