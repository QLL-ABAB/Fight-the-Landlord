# `probabilistic_response_agent.py` 代码说明

## 1. 这份代码想解决什么问题？

这份代码实现的是一个 **RLCard Greedy + 概率推理覆盖（balanced override）** 的斗地主 Agent。

之前几种方法都有一个共同问题：

- 纯值迭代 / DP：主要优化自己的手牌拆分，容易忽略牌权、队友、敌人响应。
- 采样对抗搜索：样本少时方差大，样本多时速度慢，而且叶子评价函数不准。
- 概率推理：比采样稳定，但如果直接用手写评分覆盖 greedy，容易误判。

所以这份代码采用折中策略：

```text
默认使用 RLCardAgent 的 greedy 动作。
只有当概率推理动作在特定场景下明显更好时，才允许覆盖 RLCard。
```

换句话说，它不是一个完全替代 RLCard 的新策略，而是一个：

```text
RLCard baseline + selective reasoning override
```

也可以称为：

```text
Greedy++ Agent
```

---

## 2. 总体决策公式

对于一个候选动作 `a`，代码估计：

```text
Q(a) = immediate_score(a)
       + response_weight * sum_r P(r | a, history) * response_value(a, r)
```

其中：

- `immediate_score(a)`：我现在出这手牌本身好不好。
- `P(r | a, history)`：我出了 `a` 以后，下一个玩家以多大概率响应 `r`。
- `response_value(a, r)`：下一个玩家响应 `r` 后，对我方局面是好是坏。
- `response_weight`：响应期望在总评分中的权重。

但最终并不是简单选择 `Q(a)` 最大的动作。真正逻辑是：

```text
1. 先用 RLCardAgent 得到 greedy_action。
2. 再用概率推理找出 best_action。
3. 如果 best_action 没有通过 should_override_greedy() 的门槛，就继续使用 greedy_action。
4. 只有少数情况下才返回 best_action。
```

这就是 balanced 版本的核心。

---

## 3. 文件依赖关系

代码主要依赖两个外部部分：

### 3.1 `probability_inference.py`

从中导入：

```python
from .probability_inference import (
    EnvCard2RealCard,
    RealCard2EnvCard,
    INDEX,
    CARD_ORDER,
    NORMAL_CHAIN_ORDER,
    ResponseOption,
    sort_card_str,
    main_rank_value,
    parse_action,
    can_beat,
    contains_probability,
    estimate_response_distribution,
)
```

其中最关键的是：

```python
estimate_response_distribution(...)
```

它用于估计下一个玩家的响应概率分布。

例如我出 `99` 后，它可能估计：

```text
pass: 0.45
TT:   0.20
JJ:   0.12
QQ:   0.08
bomb: 0.15
```

这个概率不是通过大量采样得到的，而是根据未知牌集合、对方剩余手牌数量、行为权重估计得到的。

### 3.2 `rlcard_agent.py`

代码中导入：

```python
try:
    from .rlcard_agent import RLCardAgent
except Exception:
    from rlcard_agent import RLCardAgent
```

并在初始化中创建 greedy 底座：

```python
self.greedy_policy = RLCardAgent(position)
```

也就是说，这个 agent 的基础行为来自 RLCardAgent。

---

## 4. 初始化部分：`__init__()`

核心初始化如下：

```python
self.name = "ProbabilisticResponseAgent_RLCardBalanced"
self.position = position
self.debug = debug
self.greedy_policy = RLCardAgent(position)
```

其中：

- `position`：当前玩家身份，可能是：
  - `landlord`
  - `landlord_down`
  - `landlord_up`
- `debug`：是否保存调试信息。
- `greedy_policy`：RLCard greedy 底座。

此外还有一个统计字典：

```python
self.stats = {
    "acts": 0,
    "overrides": 0,
    "override_leading": 0,
    "override_enemy_danger": 0,
    "override_finish": 0,
    "blocked_overrides": 0,
    "fallbacks": 0,
}
```

这些统计量很重要，用来观察这个 agent 是否真的在覆盖 RLCard。

含义如下：

| 字段                      | 含义                         |
| ------------------------- | ---------------------------- |
| `acts`                  | 总决策次数                   |
| `overrides`             | 成功覆盖 RLCard 的次数       |
| `override_leading`      | 主动出牌时覆盖 RLCard 的次数 |
| `override_enemy_danger` | 敌人危险时覆盖 RLCard 的次数 |
| `override_finish`       | 直接出完导致覆盖的次数       |
| `blocked_overrides`     | 推理动作被门槛拦住的次数     |
| `fallbacks`             | 代码异常后 fallback 的次数   |

建议实验时重点看：

```text
overrides / acts
```

如果这个比例太低，比如低于 1%，说明它几乎等于 RLCard。

如果这个比例太高，比如超过 10%，说明它可能覆盖太频繁，容易变差。

---

## 5. 重要参数解释

参数都在 `self.cfg` 中。

### 5.1 候选动作数量

```python
"root_topk_leading": 8,
"root_topk_following": 6,
```

含义：

- 主动出牌时，只评估前 8 个候选动作。
- 跟牌时，只评估前 6 个候选动作。

这么做是为了降低计算量。

---

### 5.2 是否启用推理覆盖

```python
"enable_reasoning_override": True,
```

如果设成：

```python
"enable_reasoning_override": False,
```

那么 agent 会退化成 RLCard wrapper：

```text
只调用 RLCardAgent，不使用概率推理覆盖。
```

这个开关主要用于检查接入是否正确。

如果关闭覆盖后仍然和 RLCard 表现不同，说明问题不在算法，而在导入、评测或 position 传递。

---

### 5.3 覆盖阈值 margin

```python
"override_margin_leading": 22.0,
"override_margin_following": 38.0,
"override_margin_enemy_danger": 8.0,
```

含义是：

```text
新动作的估计分数必须比 RLCard 动作高出一定 margin，才有资格覆盖。
```

也就是：

```text
best_q - greedy_q >= margin
```

不同场景阈值不同：

| 场景     | 参数                             | 解释                                     |
| -------- | -------------------------------- | ---------------------------------------- |
| 主动出牌 | `override_margin_leading`      | 主动拆牌时 RLCard 有时短视，因此阈值中等 |
| 普通跟牌 | `override_margin_following`    | RLCard 跟牌很强，所以阈值较高            |
| 敌人危险 | `override_margin_enemy_danger` | 敌人快出完时需要更积极拦截，因此阈值较低 |

如果你发现它几乎不覆盖，可以降低这些值。

如果你发现覆盖后胜率下降，可以提高这些值。

---

### 5.4 概率响应模型参数

```python
"max_responses": 18,
"response_temperature": 8.0,
"response_weight": 1.00,
"strategic_pass_enemy": 0.22,
"strategic_pass_teammate": 0.80,
"strategic_pass_unknown": 0.42,
```

含义：

| 参数                        | 含义                               |
| --------------------------- | ---------------------------------- |
| `max_responses`           | 最多保留多少个可能响应动作         |
| `response_temperature`    | softmax 温度，越大响应分布越平滑   |
| `response_weight`         | 概率响应期望在 Q 值中的权重        |
| `strategic_pass_enemy`    | 敌人即使能压也选择 pass 的基础概率 |
| `strategic_pass_teammate` | 队友即使能压也选择 pass 的基础概率 |
| `strategic_pass_unknown`  | 关系未知时的战略性 pass 概率       |

其中 `strategic_pass_teammate` 很高，是因为农民队友一般不应该乱压队友。

---

## 6. 主流程：`act(infoset)`

这是整份代码的主入口。

简化流程如下：

```text
act(infoset):
    1. 读取 legal_actions。
    2. 如果只有一个合法动作，直接返回。
    3. 提取 state。
    4. 推断 belief。
    5. 如果能直接出完，直接出。
    6. 调用 RLCardAgent 得到 greedy_action。
    7. 如果关闭推理覆盖，直接返回 greedy_action。
    8. 生成候选动作 candidates。
    9. 对每个候选动作计算 root_action_value。
    10. 找出评分最高的 best_action。
    11. 调用 should_override_greedy 判断是否允许覆盖。
    12. 如果允许，返回 best_action；否则返回 greedy_action。
```

其中最关键的是第 6 步和第 11 步。

---

### 6.1 为什么调用 RLCard 时要 `deepcopy`？

代码中有：

```python
greedy_action = self.greedy_policy.act(copy.deepcopy(infoset))
```

原因是原版 RLCardAgent 内部可能会修改 `infoset` 里的列表，例如把数字牌改成字符串牌。

如果不 `deepcopy`，当前 agent 后面继续使用同一个 `infoset` 时可能出问题。

所以这里用：

```python
copy.deepcopy(infoset)
```

让 RLCard 在副本上运行，避免污染当前状态。

---

### 6.2 直接出完优先

代码中：

```python
finish_actions = [
    a for a in legal_actions
    if a != [] and len(self.env_cards_to_real_str(a)) == state["my_count"]
]
if finish_actions:
    return self.choose_lowest_cost_action(finish_actions, state)
```

如果某个合法动作可以一次性打光自己的手牌，就直接出。

这个规则优先于 RLCard 和概率推理。

---

### 6.3 Debug / ablation 模式

代码中：

```python
if not self.cfg["enable_reasoning_override"]:
    return greedy_action
```

如果关闭该参数，agent 就只返回 RLCard 动作。

这个模式用于排查：

```text
如果纯 wrapper 都不能复现 RLCard，说明不是算法问题，而是接入问题。
```

---

## 7. 候选动作生成：`prune_root_actions()`

`legal_actions` 可能很多，不可能每个都仔细概率推理，所以先筛选 top-K。

主动出牌：

```python
candidates = [a for a in legal_actions if a != []]
```

跟牌时：

- 如果敌人危险，只考虑非 pass 动作。
- 否则保留所有合法动作。

然后用 `fast_action_rank_score()` 快速打分，保留前几个。

这个函数不是最终评分，只是粗筛。

---

## 8. 候选动作快速评分：`fast_action_rank_score()`

它主要看：

```text
1. 出完加很大分。
2. 手牌 badness 降低越多越好。
3. 出牌张数越多略好。
4. 能带走最小牌加分。
5. 顺子/连牌加分。
6. 单张高牌扣分。
7. 非终局炸弹扣分。
```

对应代码逻辑：

```python
score += 5.0 * (hand_badness(hand) - hand_badness(next_hand))
score += 0.8 * len(action_str)
if next_hand == "":
    score += finish_bonus
if min_card(hand) in action_str:
    score += min_card_bonus
if is_chain_type(t):
    score += chain_bonus
if len(action_str) == 1:
    score -= 0.25 * main_rank_value(action_str)
if is_bomb_or_rocket(t) and next_hand != "":
    score -= 45.0
```

它的目标是快速找出值得深评估的动作，而不是精确判断胜率。

---

## 9. 核心评分：`root_action_value()`

这是概率推理评分的核心。

如果动作是 pass：

```python
return self.pass_value(state)
```

如果是出牌动作：

```python
immediate = self.immediate_action_score(action_str, state, belief)
```

然后估计下一个玩家怎么响应：

```python
dist = estimate_response_distribution(...)
```

最后计算响应期望：

```python
expected_response = 0.0
for opt in dist:
    expected_response += opt.prob * self.response_value(...)

return immediate + response_weight * expected_response
```

数学形式就是：

```text
Q(a) = immediate_score(a)
       + response_weight * E[response_value]
```

注意：这个 Q 仍然是启发式评分，不是真实胜率。

所以后面必须通过 `should_override_greedy()` 做严格过滤。

---

## 10. 当前动作即时评分：`immediate_action_score()`

这个函数评估我现在出 `action_str` 的直接价值。

主要因素有：

### 10.1 出完直接最大化

```python
if next_hand == "":
    return finish_bonus
```

能出完直接给极大分数。

---

### 10.2 手牌结构改善

```python
score += hand_improve_weight * (hand_badness(hand) - hand_badness(next_hand))
```

如果出牌后手牌更顺，分数更高。

这里 `hand_badness()` 越低越好，所以：

```text
hand_badness(before) - hand_badness(after)
```

越大，说明改善越明显。

---

### 10.3 出牌数量、最小牌、牌型奖励

```python
score += action_len_weight * len(action_str)
```

出更多牌略加分。

```python
if min_card(hand) in action_str:
    score += min_card_bonus
```

带走最小牌加分。

```python
if is_chain_type(type_str):
    score += chain_bonus + chain_len_bonus * len(action_str)
```

顺子、连对、飞机等结构牌加分。

三张、对子也有小幅奖励。

---

### 10.4 高单张和控制牌惩罚

主动出单张时：

```python
if state["leading_round"] and len(action_str) == 1:
    score -= lead_high_single_penalty * main_rank_value(action_str)
```

这避免主动打出 A、2、王这种控制牌。

控制牌消耗也会扣分：

```python
score -= control_cost_weight * control_cost(action_str, state)
```

`control_cost()` 中，A、2、小王、大王依次有更高成本。

---

### 10.5 炸弹惩罚

如果不是直接出完，炸弹和王炸通常不该随便用：

```python
if is_bomb_or_rocket(type_str):
    if leading_round:
        score -= lead_bomb_penalty
    else:
        score -= nonfinish_bomb_penalty
```

---

### 10.6 队友 / 敌人修正

如果不是主动出牌：

- 压队友：扣分。
- 压敌人：加分。
- 敌人危险时压敌人：额外加分。

对应逻辑：

```python
if is_teammate_last_player(state):
    score += beat_teammate_penalty
elif is_enemy_last_player(state):
    score += beat_enemy_bonus
    if state["dangerous"]:
        score += enemy_danger_beat_bonus
```

---

## 11. 响应价值：`response_value()`

这个函数评估：

```text
我出完某动作后，下一个玩家如果做出某响应，对我方好不好。
```

### 11.1 下家 pass

如果下一个玩家 pass：

- 如果他是敌人，对我方通常好，因为我的动作暂时安全。
- 如果他是队友，也通常不错。

```python
if opt.action == "":
    if relation == "enemy":
        return next_pass_bonus_enemy + keep_initiative_bonus
    if relation == "teammate":
        return next_pass_bonus_teammate
```

---

### 11.2 敌人响应

如果下一个玩家是敌人且出牌压我：

```python
val += enemy_response_penalty
val += enemy_response_len_penalty * len(opt.action)
val += lose_initiative_penalty
```

也就是敌人能压我，扣分。

如果敌人直接出完：

```python
val += enemy_response_finish_penalty
```

极大扣分。

如果敌人用炸弹：

```python
val += enemy_bomb_response_extra_penalty
```

额外扣分，但也给一点“敌人消耗控制牌”的补偿。

---

### 11.3 队友响应

如果队友直接出完：

```python
val += teammate_response_finish_bonus
```

极大加分。

如果队友没出完却压我：

```python
val += teammate_response_nonfinish_penalty
```

扣分。

---

## 12. Pass 评分：`pass_value()`

如果当前是主动出牌，却选择 pass：

```python
return leading_pass_penalty
```

直接极大扣分，因为主动轮不能无意义 pass。

如果是跟牌：

- 队友刚出牌：pass 加分。
- 队友快出完：pass 大加分。
- 敌人刚出牌：pass 扣分。
- 敌人危险：pass 大扣分。

这对应斗地主合作逻辑：

```text
农民不要乱压队友；敌人快走时不能轻易放。
```

---

## 13. 覆盖 RLCard 的核心：`should_override_greedy()`

这是 balanced 版最重要的函数。

它决定：

```text
best_action 是否真的允许替代 greedy_action。
```

这个函数返回：

```python
(allowed: bool, reason: str)
```

`reason` 用来统计覆盖类型。

---

### 13.1 相同动作：直接允许

```python
if best_action == greedy_action:
    return True, "same"
```

如果推理动作和 RLCard 动作一样，就没有覆盖问题。

---

### 13.2 直接出完：永远允许

```python
if best_str and len(best_str) == state["my_count"]:
    return True, "finish"
```

能出完就是最优先。

---

### 13.3 禁止用 pass 覆盖 RLCard 的出牌动作

```python
if best_action == [] and greedy_action != []:
    return False, "no_pass_over_nonpass"
```

这是为了防止概率推理误判导致“该压不压”。

这是之前很多 heuristic agent 比 RLCard 差的常见原因。

---

### 13.4 禁止阻塞队友

```python
if is_teammate_last_player(state):
    if best_str and len(best_str) < state["my_count"]:
        return False, "no_block_teammate"
```

如果上一手是队友出的，除非我能直接出完，否则不压队友。

---

### 13.5 必须超过 margin

```python
advantage = best_q - greedy_q
if advantage < margin:
    return False, "low_advantage"
```

推理动作必须明显优于 RLCard 动作，才有资格覆盖。

---

## 14. 三类允许覆盖的主要场景

### 14.1 主动出牌覆盖

主动出牌时，RLCard 的弱点是：

```text
它主要选择包含最小牌的组合，但不显式比较所有拆牌方案。
```

所以 balanced 版允许在主动出牌时覆盖，但限制很严格。

不允许：

```python
if best_is_bomb and best_next != "":
    return False, "no_lead_bomb"
```

也就是不允许主动用非终局炸弹覆盖。

也不允许高单张：

```python
if is_high_single(best_str) and best_next != "":
    return False, "no_high_single"
```

允许覆盖的情况主要是：

1. RLCard 动作可疑，比如非终局炸弹或高单张。
2. 新动作是结构牌，比如顺子、连对、三张、较长组合。
3. 新动作让 `hand_badness` 改善明显优于 RLCard。

判断逻辑：

```python
best_improve = hand_badness(hand) - hand_badness(best_next)
greedy_improve = hand_badness(hand) - hand_badness(greedy_next)
if best_structure and best_improve >= greedy_improve + 1.5:
    return True, "leading"
```

---

### 14.2 敌人危险时覆盖

如果上一手是敌人出的，而且敌人只剩 1-2 张：

```python
state["dangerous"] == True
```

这时 RLCard 的 pass 可能太保守。

所以如果：

```python
greedy_action == [] and best_action != []
```

说明 RLCard 想 pass，但推理模块认为应该压。

这时允许覆盖：

```python
return True, "enemy_danger"
```

但炸弹仍有限制：

```python
if best_is_bomb and not state.get("very_dangerous", False):
    return False, "bomb_only_very_danger"
```

也就是说，只有敌人只剩 1 张时，才更允许用炸弹类动作。

---

### 14.3 普通敌人跟牌时覆盖

普通情况下，RLCard 的“最低成本同牌型压制”非常强，所以这里非常保守。

只有当：

- 新动作不是炸弹；
- 新动作不是 pass；
- 新动作和 RLCard 都是出牌动作；
- 新动作显著改善手牌结构；
- 新动作没有明显多消耗控制牌；

才允许覆盖：

```python
if best_improve >= greedy_improve + 3.0 and control_cost(best_str) <= control_cost(greedy_str) + 1.0:
    return True, "ordinary_enemy_shape"
```

否则继续使用 RLCard。

---

## 15. 状态提取：`extract_state()`

该函数把 `infoset` 转换成更方便的字典。

主要字段包括：

| 字段                  | 含义                |
| --------------------- | ------------------- |
| `my_hand`           | 当前玩家手牌字符串  |
| `my_count`          | 当前玩家剩余牌数    |
| `last_move`         | 上一手非 pass 动作  |
| `last_two_moves`    | 最近两手动作        |
| `last_pid`          | 上一手出牌玩家      |
| `leading_round`     | 当前是否主动出牌    |
| `num_cards_left`    | 三个玩家剩余牌数    |
| `enemy_positions`   | 敌人位置            |
| `teammate_position` | 队友位置            |
| `enemy_min_cards`   | 敌人中最少剩牌数    |
| `teammate_cards`    | 队友剩余牌数        |
| `dangerous`         | 敌人是否只剩 1-2 张 |
| `very_dangerous`    | 敌人是否只剩 1 张   |
| `is_landlord`       | 自己是否地主        |

其中：

```python
"dangerous": enemy_min_cards <= danger_cards
"very_dangerous": enemy_min_cards <= very_danger_cards
```

默认：

```python
"danger_cards": 2
"very_danger_cards": 1
```

---

## 16. Belief 推断：`infer_belief()`

这个函数目前比较简单：

```python
unknown_counter = self.get_unknown_cards(infoset, state["my_hand"])
return {"unknown_counter": unknown_counter}
```

也就是根据：

```text
完整牌堆 - 我的手牌 - 已经出过的牌
```

得到未知牌集合。

这个未知牌集合会传给：

```python
estimate_response_distribution(...)
```

用于估计下一个玩家可能有什么牌、会怎么响应。

---

## 17. 手牌坏度：`hand_badness()`

这是很多评分函数的基础。

它认为一手牌越难打出去，badness 越高。

主要考虑：

```text
单张数量
对子数量
三张数量
炸弹数量
手牌总长度
顺子 / 连对 / 飞机结构
控制牌数量
```

大致形式是：

```text
badness = turn_weight * 出牌轮次数估计
        + single_weight * 单张数
        + len_weight * 手牌长度
        - chain_discount_weight * 连牌结构奖励
        - control_discount * 控制牌数量
        - bomb_discount * 炸弹数量
```

注意：

```text
badness 越低越好。
```

所以很多地方会用：

```python
hand_badness(before) - hand_badness(after)
```

来衡量出牌后的结构改善。

---

## 18. 和 RLCard 的关系

这份代码不是放弃 RLCard，而是把 RLCard 当作默认策略。

它的基本逻辑是：

```text
RLCard 动作 = 保底动作
概率推理动作 = 候选增强动作
should_override_greedy = 安全门槛
```

只有当推理动作满足以下条件时，才会覆盖 RLCard：

```text
1. 估计分数明显高于 RLCard。
2. 不会把 RLCard 的出牌动作替换成 pass。
3. 不会乱压队友。
4. 不会在普通局面乱用炸弹。
5. 主动出牌时必须是更好的结构牌。
6. 敌人危险时可以更积极拦截。
```

---

## 19. 为什么这个版本叫 balanced？

因为之前有两个极端：

### 19.1 原始概率推理版

特点：

```text
覆盖太频繁。
```

问题：

```text
手写评分不准确，容易把 RLCard 的稳健动作替换掉，导致胜率下降。
```

### 19.2 safe 版

特点：

```text
覆盖太少。
```

问题：

```text
几乎等于 RLCard，看不出增强效果。
```

### 19.3 balanced 版

目标：

```text
在 RLCard 明显可能有弱点的地方开放覆盖，其他地方保持 RLCard。
```

主要开放场景：

```text
1. 主动出牌拆牌更优。
2. 敌人危险时 RLCard pass，但我们能压。
3. 普通跟牌中存在明显更好且不多消耗控制牌的动作。
4. 直接出完。
```

---

## 20. 如何调参？

### 20.1 如果它太像 RLCard

观察：

```text
overrides / acts 很低，比如 < 1%
```

可以降低 margin：

```python
"override_margin_leading": 14.0,
"override_margin_following": 28.0,
"override_margin_enemy_danger": 4.0,
```

或者增大概率响应权重：

```python
"response_weight": 1.20
```

---

### 20.2 如果它比 RLCard 差

观察：

```text
overrides / acts 很高，或者 override 后胜率低
```

可以提高 margin：

```python
"override_margin_leading": 32.0,
"override_margin_following": 55.0,
"override_margin_enemy_danger": 12.0,
```

或者降低响应权重：

```python
"response_weight": 0.75
```

---

### 20.3 如果敌人危险时还是不够积极

可以降低：

```python
"override_margin_enemy_danger": 4.0
```

或者提高：

```python
"enemy_danger_beat_bonus": 70.0
```

---

### 20.4 如果它乱用炸弹

提高：

```python
"lead_bomb_penalty": 80.0
"nonfinish_bomb_penalty": 60.0
"cost_bomb_weight": 75.0
```

同时保持：

```python
bomb_only_very_danger
no_lead_bomb
```

这些 gate 不要轻易删除。

---

## 21. 建议实验记录

不要只看总胜率，建议输出或记录：

```python
agent.stats
```

重点看：

```text
acts
overrides
override_leading
override_enemy_danger
override_finish
blocked_overrides
fallbacks
```

推荐分析：

```text
override_rate = overrides / acts
```

大概目标：

```text
2% ~ 6% 左右比较合理。
```

如果覆盖率接近 0%，说明它只是 RLCard。

如果覆盖率过高，说明它又变成“手写评分主导”，容易劣化。

---

## 22. 当前版本的局限

这份代码仍然不是完美方法。

主要局限：

1. `root_action_value()` 仍然是手写评分，不是真实胜率。
2. 概率推理只看下一个玩家响应，不是完整多轮 POMDP。
3. pass 之后没有严格做 belief update。
4. `hand_badness()` 是人工设计的，不一定和胜率完全一致。
5. RLCard 本身已经很强，能稳定超过它很难。

所以这份代码的合理定位是：

```text
一个基于 RLCard 的安全增强版，而不是能大幅超越 RLCard 的强 AI。
```

---

## 23. 一句话总结

这份代码的核心思想是：

```text
RLCardAgent 负责给出稳定保底动作；概率推理模块负责发现少数可能更优的动作；should_override_greedy() 负责防止不可靠的手写评分频繁覆盖 RLCard。
```

它的决策结构可以总结为：

```text
greedy_action = RLCardAgent(infoset)
best_action = argmax_a Q_prob(a)

if should_override_greedy(best_action, greedy_action):
    return best_action
else:
    return greedy_action
```

也就是：

```text
默认相信 RLCard，只在高置信度场景下进行推理覆盖。
```
