
# 代码逻辑说明

本文档解释 `adersarial_agent.py` 的功能、核心原理与代码逻辑。该代码是在原始 `bayesian_sampled_search_agent.py` 的基础上修改得到的，保留了原来的"隐藏牌采样 + 浅层对抗搜索"框架，但将敌人节点从单一 softmax 或单一 min 改成了 **按角色区分的 softmax/min 混合模型**，并且把队友节点从纯 `max` 改成了"合作 softmax + 少量 max"的模型。

---

## 1. 方法整体定位

这个 Agent 不是纯 greedy，也不是完整的 POMDP 求解器，而是一个实用的近似搜索 agent。

它的核心思想可以概括为：

> **隐藏牌采样 + 浅层队伍对抗搜索 + 角色条件敌人模型**

更具体地说：

1. 当前玩家只能看到自己的手牌和已经打出的牌；
2. 根据这些信息推断未知牌集合；
3. 将未知牌随机分配给其他玩家，生成多个可能的完整牌局；
4. 对每个候选动作，在这些采样牌局中做有限深度搜索；
5. 搜索树中：
   - 自己节点取 `max`；
   - 队友节点使用 cooperative softmax + 少量 max；
   - 敌人节点使用 softmax/min 混合模型；
6. 最后对多个采样世界的搜索结果求平均，选择估计价值最高的动作。

整体公式可以写成：

$$
a^* = \arg\max_{a\in C(s)} \left[ R_{\text{root}}(s,a) + \frac{1}{N} \sum_{i=1}^{N} Search(T(s_i,a), d) \right]
$$

其中：

- $C(s)$：剪枝后的根节点候选动作集合；
- $R_{\text{root}}(s,a)$：根动作即时评分；
- $s_i$：第 $i$ 个隐藏牌采样世界；
- $T(s_i,a)$：在采样世界 $s_i$ 中执行动作 $a$ 后的新状态；
- $d$：搜索深度；
- $N$：采样数。

---

## 2. 为什么要做 role-hybrid 修改？

原版敌人节点采用 softmax opponent model：

$$
V(s) = \sum_a \pi(a|s)V(s,a)
$$

也就是说，敌人不是总选对我方最坏的动作，而是根据启发式分数，以更高概率选择看起来较好的动作。

后来测试发现：

```text
softmax：地主胜率更高，农民胜率更低
min：地主胜率更低，农民胜率更高
```

这个现象非常有意义。它说明：

> **地主和农民面对的对手结构不同，敌人模型不能一刀切。**

因此当前版本采用按角色区分的混合模型：

$$
V(s) = (1-\lambda)V_1(s) + \lambda V_2(s)
$$

其中：

$$
V_1(s) = \sum_a \pi(a|s)V(s,a)
$$

$$
V_2(s) = \min_a V(s,a)
$$

$\lambda$ 控制敌人节点有多接近 minimax。

---

## 3. 为什么地主更适合偏 softmax？

当当前 agent 是地主时：

```text
我方：地主
敌方：两个农民
```

如果敌人节点使用纯 min：

$$
V(s) = \min_a V(s,a)
$$

就等价于假设两个农民在每一步都能选择对地主最不利的动作。

这会产生过度悲观的问题。

真实斗地主里，两个农民虽然同队，但并不是完美联合决策：

- 一个农民不知道另一个农民的完整手牌；
- 农民可能误判该不该接牌；
- 农民可能为了省牌选择 pass；
- 农民之间可能配合不完美；
- 农民不一定每步都能做出对地主最坏的响应。

如果地主把两个农民都建模成完美防守者，就会导致地主过于保守，不敢主动走牌。

而地主本身往往需要主动权和推进速度。因此地主视角下更适合：

> **偏 softmax 的有限理性敌人模型**

在代码中：

```python
"enemy_min_mix_landlord_root": 0.15
```

也就是说，当 root agent 是地主时：

$$
\lambda = 0.15
$$

敌人节点大约是：

$$
V(s) = 0.85V_{\text{softmax}} + 0.15V_{\min}
$$

这表示地主主要相信农民按较合理策略行动，但仍保留少量最坏情况风险。

---

## 4. 为什么农民更适合偏 min？

当当前 agent 是农民时：

```text
我方：两个农民
敌方：地主
```

这里敌人只有一个地主。地主是单人决策，不存在队友配合误差。

如果农民把地主建模成 softmax，就可能低估地主的压制能力：

- 地主通常会主动争夺牌权；
- 地主的行为更集中；
- 地主只为自己服务，不需要协调；
- 地主一旦快出完，威胁非常大；
- 农民如果过于乐观，容易错过拦截时机。

另外，原版搜索中队友节点是 `max`，这意味着农民会假设队友总能选择最好的合作动作。如果敌人地主也用 softmax，那么农民搜索会变成：

```text
队友：完美合作
地主：有限理性
```

这会让农民过度乐观。

因此农民视角下，地主敌人节点更适合偏 min：

> **把地主建模成更强、更会反击的对手**

在代码中：

```python
"enemy_min_mix_farmer_root": 0.65
```

也就是说，当 root agent 是农民时：

$$
\lambda = 0.65
$$

敌人节点大约是：

$$
V(s) = 0.35V_{\text{softmax}} + 0.65V_{\min}
$$

这会让农民更谨慎，更重视地主的最坏反击。

---

## 5. 为什么不直接全局用 min？

纯 min 的问题是过度悲观。

敌人节点如果写成：

$$
V(s) = \min_a V(s,a)
$$

就表示：

> 只要敌人有一个动作能让我方最难受，就假设敌人一定会选择它。

在完全信息、零和、双人游戏中，这种假设比较自然。但斗地主有几个特殊点：

1. 这是不完全信息游戏；
2. 搜索前的完整手牌是采样出来的；
3. 玩家实际上不知道其他人的全部手牌；
4. 农民之间不是一个完美统一体；
5. 搜索深度有限；
6. 叶子评价函数是手写启发式。

如果在这些近似基础上再使用纯 min，就容易把敌人建模成：

> **全知、完美、永远选择最坏动作的对手**

这会放大采样误差和评价误差，使 agent 变得过度保守。

例如某个采样世界里敌人刚好有炸弹，纯 min 可能总是假设敌人会炸。但真实对局中敌人未必愿意交炸弹，也未必判断出这是最佳时机。

因此，纯 min 更适合做风险检测，而不适合全局固定使用。

---

## 6. 为什么不直接全局用 softmax？

纯 softmax 的问题是可能过于宽容。

softmax 模型是：

$$
V(s) = \sum_a \pi(a|s)V(s,a)
$$

它假设敌人会倾向于选择较好的动作，但不一定选择最坏动作。

这在地主视角下比较合理，因为两个农民确实不一定完美联合防守。但在农民视角下，敌人是地主，地主的压制能力更集中。如果农民仍然用 softmax 建模地主，就容易低估地主威胁。

尤其当地主只剩 1-2 张牌时，农民更应该考虑最坏情况，而不是平均情况。

因此：

> **softmax 适合地主视角，min 更适合农民视角或敌人危险局面。**

---

## 7. 角色条件 hybrid 的核心公式

当前版本采用：

$$
V(s) = (1-\lambda)V_{\text{softmax}} + \lambda V_{\min}
$$

其中：

```python
"enemy_min_mix_landlord_root": 0.15
"enemy_min_mix_farmer_root": 0.65
"enemy_min_mix_danger_bonus": 0.20
"enemy_min_mix_very_danger_bonus": 0.15
"enemy_min_mix_cap": 0.95
```

含义如下：

| 参数                              | 含义                                 |
| --------------------------------- | ------------------------------------ |
| `enemy_min_mix_landlord_root`     | root 是地主时，敌人节点中 min 的权重 |
| `enemy_min_mix_farmer_root`       | root 是农民时，敌人节点中 min 的权重 |
| `enemy_min_mix_danger_bonus`      | 敌人危险时额外增加 min 权重          |
| `enemy_min_mix_very_danger_bonus` | 敌人极危险时继续增加 min 权重        |
| `enemy_min_mix_cap`               | min 权重上限                         |

如果 root 是地主：

$$
\lambda = 0.15
$$

如果 root 是农民：

$$
\lambda = 0.65
$$

如果敌人只剩 1-2 张，会进一步增加 $\lambda$。

---

## 8. enemy_min_mix_weight() 函数

代码中计算 $\lambda$ 的函数是：

```python
def enemy_min_mix_weight(self, sim):
```

其逻辑是：

```text
如果自己是地主：
    lambda = enemy_min_mix_landlord_root
否则：
    lambda = enemy_min_mix_farmer_root

计算 root 敌方阵营中最少剩余牌数 enemy_min

如果 enemy_min <= very_danger_cards:
    lambda += danger_bonus + very_danger_bonus
elif enemy_min <= danger_cards:
    lambda += danger_bonus

lambda 被限制在 [0, enemy_min_mix_cap]
```

对应含义是：

- 地主默认更乐观；
- 农民默认更谨慎；
- 敌人快出完时，不管身份如何，都要更重视最坏情况。

---

## 9. search() 中的三类节点

当前版本最重要的修改在：

```python
def search(self, sim, depth):
```

搜索树节点分为三类：

```text
1. 自己节点
2. 队友节点
3. 敌人节点
```

---

### 9.1 自己节点：max

如果当前行动者是 root agent 自己：

```python
if current == self.position:
```

则使用：

$$
V(s) = \max_a V(s,a)
$$

代码逻辑：

```python
values = []
for a in pruned[: self.cfg["ally_max_width"]]:
    ns = self.apply_action(sim, a)
    values.append(self.search(ns, depth - 1))
val = max(values)
```

这是合理的，因为自己当然会选择当前搜索下最有利的动作。

---

### 9.2 队友节点：cooperative softmax + max mix

如果当前行动者和 root agent 同队，但不是自己：

```python
elif self.same_team(current, self.position):
```

原版使用纯 max：

$$
V_{\text{teammate}}(s) = \max_a V(s,a)
$$

这表示队友总能选出对我方最好的动作，容易过度乐观。

当前版本改成：

$$
V(s) = (1-\mu)V_{\text{softmax}} + \mu V_{\max}
$$

其中：

```python
"teammate_max_mix": 0.20
"teammate_softmax_temp": 7.0
```

也就是说，队友主要按 cooperative softmax 行动，但保留少量理想合作成分。

具体计算：

```python
weights = self.softmax(scores, temp=self.cfg["teammate_softmax_temp"])
soft_val = sum(w * v for w, v in zip(weights, values))
max_val = max(values)
mix = self.cfg["teammate_max_mix"]
val = (1.0 - mix) * soft_val + mix * max_val
```

这样做的意义是：

> **队友是合作的，但不是全知完美的。**

这尤其能减少农民 agent 过度依赖队友的问题。

---

### 9.3 敌人节点：role-conditioned hybrid

如果当前行动者是敌人：

```python
else:
```

则计算：

```python
soft_val = sum(w * v for w, v in zip(weights, values))
min_val = min(values)
mix = self.enemy_min_mix_weight(sim)
val = (1.0 - mix) * soft_val + mix * min_val
```

也就是：

$$
V(s) = (1-\lambda)V_{\text{softmax}} + \lambda V_{\min}
$$

这是当前版本的核心改动。

---

## 10. act() 主流程

主函数是：

```python
def act(self, infoset):
```

整体流程：

```text
1. 读取 legal_actions
2. 如果没有动作，返回 []
3. 如果只有一个动作，直接返回
4. 提取当前状态 root_state
5. 构建 belief
6. 如果有动作能直接出完，直接出
7. 剪枝得到候选动作 candidates
8. 采样若干隐藏手牌 worlds
9. 对每个候选动作做 evaluate_root_action
10. 选择估值最高的动作
11. 如果异常，fallback 到 greedy_fallback_action
```

直接出完优先级最高：

```python
finish_actions = [
    a for a in legal_actions
    if a != [] and len(self.env_cards_to_real_str(a)) == root_state["my_count"]
]
if finish_actions:
    return self.choose_lowest_cost_action(finish_actions, root_state)
```

这表示，只要能赢当前局部牌局，就不再搜索。

---

## 11. evaluate_root_action() 根动作评估

对一个候选动作 $a$，代码计算：

$$
Q(a) = R_{\text{root}}(s,a) + \frac{1}{N} \sum_i Search(s_i', d-1)
$$

对应代码：

```python
immediate = self.root_immediate_score(action, root_state, belief)

for hands in samples:
    sim = self.build_initial_sim_state(root_state, hands)
    next_sim = self.apply_action(sim, action_str)
    val = self.search(next_sim, self.cfg["search_depth"] - 1)
    total += val

return immediate + total / used
```

其中：

- `root_immediate_score()` 是当前动作的即时战术评分；
- `search()` 是采样世界里的后续搜索；
- 多个 sample 的结果取平均。

---

## 12. root_immediate_score() 即时评分

这个函数给根动作加入一些局部牌理。

如果动作是 pass：

- 队友刚出牌：pass 加分；
- 敌人刚出牌：pass 扣分；
- 敌人危险：pass 大扣分；
- 主动出牌时 pass 极大扣分。

如果动作不是 pass：

- 能出完：加分；
- 压敌人：加分；
- 敌人危险时压敌人：额外加分；
- 压队友且不能出完：大扣分；
- 主动出高单张：扣分；
- 非终局使用炸弹：扣分；
- 被压风险高：扣分。

这个函数的目的不是代替搜索，而是给搜索加入强牌理约束，避免明显不合理动作。

---

## 13. 隐藏手牌采样

函数：

```python
sample_determinizations(infoset, state, n)
```

负责生成多个可能世界。

步骤：

1. 从完整牌堆中减去自己手牌；
2. 减去已经打出的牌；
3. 得到未知牌集合；
4. 根据 `num_cards_left` 确定其他玩家各自还剩几张；
5. 随机打乱未知牌；
6. 按剩余牌数分给其他玩家；
7. 重复 $n$ 次。

这就是 determinization sampling。

它把不完全信息问题转化成多个完全信息近似问题。

---

## 14. evaluate_sim_state() 叶子评价

当搜索深度耗尽或超时时，代码调用：

```python
evaluate_sim_state(sim)
```

该函数用手写启发式估计局面好坏。

它主要考虑：

1. 手牌越少越好；
2. 手牌结构越好越好；
3. 我方有人快出完，加分；
4. 敌方有人快出完，扣分；
5. 我方有牌权，加分；
6. 上一手是我方出的，加分；
7. 我方有炸弹和控制牌，加分；
8. 敌方有炸弹和控制牌，扣分。

简化形式：

$$
Eval(s) = TeamGood(s) - EnemyGood(s) + DangerBonus + InitiativeBonus + ControlBonus
$$

注意，它仍然是手写评价函数，不是训练出来的真实胜率。

---

## 15. 动作生成与剪枝

根节点动作来自环境：

```python
infoset.legal_actions
```

这是最安全的，因为它保证动作合法。

搜索树内部动作需要自己生成：

```python
generate_actions_from_hand(hand)
```

支持：

- 单张；
- 对子；
- 三张；
- 三带一；
- 三带二；
- 炸弹；
- 王炸；
- 四带二；
- 顺子；
- 连对；
- 飞机；
- 飞机带翅膀。

由于动作空间可能很大，代码用参数限制：

```python
"max_generated_actions_per_state": 120
"max_wing_combinations": 12
```

根节点剪枝参数：

```python
"root_topk_leading": 7
"root_topk_following": 5
```

模拟节点剪枝参数：

```python
"sim_topk_leading": 5
"sim_topk_following": 4
```

这样可以避免搜索树爆炸。

---

## 16. 风险估计 estimate_beaten_risk()

该函数估计当前动作被压住的风险。

如果当前动作是炸弹：

- 看是否可能有更大的炸弹；
- 看是否可能有王炸。

如果当前动作是普通牌：

- 看未知牌中是否可能有更大的同牌型；
- 看是否可能有炸弹；
- 看是否可能有王炸。

风险越高，主动出牌得分越低。

这能减少一些明显冒险动作，例如主动出容易被压的小单张。

---

## 17. 关键参数说明

### 搜索预算

```python
"num_samples": 400
"search_depth": 3
"time_budget_sec": 0.20
```

注意：`num_samples=400` 和 `time_budget_sec=0.20` 有冲突。虽然代码尝试采样 400 个世界，但每步只有 0.20 秒软时间限制，所以实际可能只用到一部分 sample。

可以通过 `stats["samples_used"]` 和 `stats["time_cutoffs"]` 观察实际使用情况。

---

### 敌人 hybrid 参数

```python
"enemy_min_mix_landlord_root": 0.15
"enemy_min_mix_farmer_root": 0.65
"enemy_min_mix_danger_bonus": 0.20
"enemy_min_mix_very_danger_bonus": 0.15
"enemy_min_mix_cap": 0.95
```

调参建议：

如果地主太保守：

```python
"enemy_min_mix_landlord_root": 0.05
```

如果地主太冒险：

```python
"enemy_min_mix_landlord_root": 0.25
```

如果农民太乐观、拦不住地主：

```python
"enemy_min_mix_farmer_root": 0.75
```

如果农民太保守：

```python
"enemy_min_mix_farmer_root": 0.50
```

---

### 队友模型参数

```python
"teammate_softmax_temp": 7.0
"teammate_max_mix": 0.20
```

如果农民过度依赖队友，可以降低：

```python
"teammate_max_mix": 0.10
```

如果希望队友更理想化，可以提高：

```python
"teammate_max_mix": 0.35
```

---

## 18. 与 softmax 版本和 min 版本的区别

### 原 softmax 版本

敌人节点：

$$
V_{\text{enemy}} = V_{\text{softmax}}
$$

优点：

- 不会过度悲观；
- 地主更敢进攻；
- 对双农民敌人建模更合理。

缺点：

- 农民可能低估地主；
- 敌人危险时可能不够谨慎。

---

### min 版本

敌人节点：

$$
V_{\text{enemy}} = V_{\min}
$$

优点：

- 更谨慎；
- 农民更重视地主最坏反击；
- 敌人危险时更安全。

缺点：

- 地主面对两个农民时过度悲观；
- 可能不敢推进；
- 容易放大小概率坏情况。

---

### role-hybrid 版本

敌人节点：

$$
V(s) = (1-\lambda)V_{\text{softmax}} + \lambda V_{\min}
$$

优点：

- 地主偏 softmax，保持主动性；
- 农民偏 min，更重视地主威胁；
- 敌人危险时自动提高 min 权重；
- 比单一 softmax/min 更符合斗地主角色结构。

---

## 19. 当前方法的局限

### 19.1 仍然依赖手写评价函数

叶子节点的 `evaluate_sim_state()` 不是训练出来的胜率模型，而是手写启发式。所以搜索不一定稳定超过 RLCard greedy。

### 19.2 采样存在 determinization 偏差

每个 sample 假设隐藏牌确定，但真实玩家并不知道这些牌。因此搜索内部可能出现"玩家像知道完整牌局一样行动"的问题。

### 19.3 队友和敌人模型仍然近似

虽然 role-hybrid 比纯 softmax/min 更合理，但它仍然是人工设定参数，不是从数据中学习出来的真实策略模型。

### 19.4 搜索深度有限

`search_depth=3` 只能看到很浅的后续局面，很多长期拆牌和牌权交换仍然无法准确判断。

### 19.5 动作生成和环境可能有细节差异

根节点动作来自 `infoset.legal_actions`，较可靠；但搜索内部动作由 `generate_actions_from_hand()` 自己生成，可能与环境完整规则存在细微差异。

---

## 20. 实验建议

建议不要只看总胜率，而要分角色统计：

```text
地主胜率
农民胜率
总体胜率
time_cutoffs
samples_used
fallbacks
```

特别是对比三个版本：

```text
1. softmax enemy
2. min enemy
3. role-hybrid enemy
```

如果 role-hybrid 正常，预期应该是：

- 地主胜率不会像纯 min 那样下降太多；
- 农民胜率不会像纯 softmax 那样偏低；
- 总体表现更平衡。

如果地主仍然弱，可以降低：

```python
enemy_min_mix_landlord_root
```

如果农民仍然弱，可以提高：

```python
enemy_min_mix_farmer_root
```

如果整体太慢，可以降低：

```python
num_samples
search_depth
root_topk
sim_topk
```

---

## 21. 总结

该代码实现的是一个 **角色条件混合敌人模型的贝叶斯采样浅层对抗搜索 Agent**。

它的核心贡献是：

> **不再把敌人节点固定为 softmax 或 min，而是根据当前 agent 是地主还是农民，动态调整敌人节点的悲观程度。**

核心直觉是：

> **地主面对两个农民时，不应该过度悲观，因此偏 softmax。**

> **农民面对单个地主时，不能低估地主压制，因此偏 min。**

同时，队友节点也不再假设完美合作，而是使用 cooperative softmax + 少量 max，减少农民过度依赖队友的问题。

一句话概括：

> **这是一个在原采样对抗搜索基础上加入角色条件 opponent model 的改进版本。**
>
