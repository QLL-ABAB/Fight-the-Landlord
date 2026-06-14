ValueDPAgent 深度解析

## 一、整体架构概览

`ValueDPAgent` 是一个基于**简化 MDP（马尔可夫决策过程）**的斗地主智能体，采用**记忆化动态规划（Memoized DP）**替代传统的在线价值迭代，结合**贪婪锚点安全层（Greedy-Anchor Safety Layer）**实现高效且稳定的决策。

### 核心设计理念

| 设计要点              | 说明                                                    |
| --------------------- | ------------------------------------------------------- |
| **简化 MDP**          | 状态仅包含玩家手牌，不考虑对手手牌和游戏上下文          |
| **有向无环图（DAG）** | 每次出牌严格减少手牌数量，状态空间形成 DAG，适合递归 DP |
| **记忆化**            | 缓存状态价值和最佳动作，避免重复计算                    |
| **贪婪锚点**          | DP 仅在有明显优势时覆盖贪婪基线，防止战术失误           |

### 决策流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         act(infoset)                           │
├─────────────────────────────────────────────────────────────────┤
│  1. 提取状态 (extract_state)                                    │
│  2. 推断信念 (infer_belief)                                      │
│  3. 直接出牌检测 (finish_actions)                                │
│  4. 队友保护规则                                                │
│  5. 计算贪婪基线动作 (greedy_baseline_action)                    │
│  6. 候选动作集筛选                                              │
│  7. 计算各动作 Q 值 (online_q_value)                             │
│  8. DP 覆盖决策 (should_override_greedy)                        │
│  9. 返回最终动作                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、配置参数详解

### 2.1 DP / 简化 MDP 参数

```python
"gamma": 0.94,                      # 折扣因子，权衡当前与未来奖励
"max_dp_states": 3500,              # DP 状态缓存上限（防止内存爆炸）
"max_generated_actions_per_state": 90,  # 每个状态生成的最大动作数
"max_wing_combinations": 16,        # 三带两/四带两等翅膀组合的最大数量
"cutoff_value_badness_weight": 0.85,    # 截断值的手牌坏度权重
"cutoff_value_len_weight": 0.35,    # 截断值的手牌长度权重
```

### 2.2 启发式奖励参数（R_heur）

```python
"terminal_reward": 90.0,            # 出牌完毕奖励
"turn_penalty": -4.0,               # 每轮出牌惩罚（鼓励快速出牌）
"card_reward": 0.35,                # 出牌数量奖励（鼓励出多张牌）
"structure_reward": 1.35,           # 手牌结构改善奖励
"min_card_bonus": 2.2,              # 打出最小牌奖励
"chain_bonus": 4.0,                 # 顺子奖励
"chain_len_bonus": 0.45,            # 顺子长度奖励
"trio_bonus": 2.2,                  # 三张奖励
"pair_bonus": 1.0,                  # 对子奖励
"four_with_penalty": -4.0,          # 四带两惩罚（减少拆弹）
"bomb_use_penalty": -24.0,          # 使用炸弹惩罚
"rocket_use_penalty": -28.0,        # 使用火箭惩罚
"control_use_penalty_weight": 0.55, # 大牌使用权重
"finish_bonus_online": 70.0,        # 在线模式出牌完毕额外奖励
```

### 2.3 在线上下文调整参数

```python
"pass_penalty": -7.0,               # 过牌惩罚
"leading_pass_penalty": -9999.0,    # 首轮过牌惩罚（禁止）
"danger_pass_penalty": -45.0,       # 危险情况下过牌惩罚
"teammate_release_cards": 2,        # 队友剩余牌数阈值
"teammate_release_bonus": 80.0,     # 让队友出牌奖励
"beat_teammate_penalty": 45.0,      # 压制队友惩罚
"enemy_danger_beat_bonus": 35.0,    # 危险情况下击败敌人奖励
"unnecessary_bomb_follow_penalty": 48.0,  # 不必要时用炸弹跟随惩罚
"risk_penalty_weight": 5.0,         # 风险惩罚权重
"enemy_follow_beat_bonus": 13.0,    # 跟随击败敌人奖励
"same_type_follow_bonus": 7.0,      # 同类型跟随奖励
"ordinary_enemy_pass_penalty": -12.0,  # 普通情况下放过敌人惩罚
"leading_single_high_penalty": 0.18,   # 首轮出高单牌惩罚
"leading_bomb_extra_penalty": 18.0,    # 首轮出炸弹额外惩罚
```

### 2.4 贪婪锚点安全层参数

```python
"dp_override_margin_leading": 10.0,     # 首轮 DP 覆盖阈值
"dp_override_margin_following": 16.0,   # 跟牌时 DP 覆盖阈值
"dp_override_margin_enemy_danger": 3.0, # 敌人危险时 DP 覆盖阈值
"max_override_risk": 0.72,              # 最大覆盖风险
"allow_non_finish_bomb_override": False, # 是否允许非结束时炸弹覆盖
"greedy_lead_bomb_penalty": 80.0,       # 贪婪策略首轮出炸弹惩罚
"greedy_lead_high_single_penalty": 0.40, # 贪婪策略首轮出高单牌惩罚
"greedy_lead_hand_improve_weight": 7.0, # 手牌改善权重
"greedy_lead_len_weight": 1.0,          # 出牌长度权重
"greedy_lead_chain_bonus": 10.0,        # 顺子奖励
"greedy_lead_trio_bonus": 5.0,          # 三张奖励
"greedy_lead_pair_bonus": 2.0,          # 对子奖励
```

### 2.5 状态危险阈值

```python
"danger_cards": 2,           # 危险阈值（敌人剩余牌数）
"very_danger_cards": 1,      # 极度危险阈值
```

### 2.6 贝叶斯信念参数

```python
"rocket_prob": 0.25,         # 火箭概率先验
"bomb_prob_per_possible": 0.13,  # 每个可能炸弹的概率
"bomb_prob_cap": 0.65,       # 炸弹概率上限
"higher_bomb_risk_weight": 0.12, # 更高炸弹风险权重
"risk_cap": 0.9,             # 风险上限
"bomb_risk_weight": 0.25,    # 炸弹风险权重
"rocket_risk_weight": 0.15,  # 火箭风险权重
"same_type_risk_cap": 0.8,   # 同类型风险上限
"same_type_exp_weight": 0.35, # 同类型风险指数权重
```

### 2.7 手牌坏度参数

```python
"badness_turn_weight": 7.0,          # 出牌轮数权重
"badness_single_weight": 3.0,        # 单牌权重
"badness_len_weight": 1.2,           # 手牌长度权重
"badness_chain_discount_weight": 5.0, # 顺子折扣权重
"badness_control_discount": 2.2,     # 大牌折扣
"badness_bomb_discount": 3.0,        # 炸弹折扣
"solo_chain_coef": 1.0,              # 单顺系数
"pair_chain_coef": 1.5,              # 双顺系数
"trio_chain_coef": 2.0,              # 三顺系数
```

### 2.8 动作成本参数

```python
"cost_len_weight": 0.5,      # 长度成本权重
"cost_rank_weight": 0.08,    # 牌面成本权重
"cost_bomb_weight": 50.0,    # 炸弹成本
"A_cost": 1.5,               # A 的成本
"2_cost": 4.5,               # 2 的成本
"B_cost": 7.0,               # 小王成本
"R_cost": 8.0,               # 大王成本
"danger_control_scale": 0.45, # 危险时大牌成本缩放
```

---

## 三、核心函数详解

### 3.1 状态提取：`extract_state(infoset)`

**功能**：从游戏信息集中提取关键状态特征

**输入**：`infoset` - 游戏信息集，包含手牌、历史动作、剩余牌数等

**输出**：状态字典，包含以下字段：

| 字段                | 类型     | 说明                   |
| ------------------- | -------- | ---------------------- |
| `my_hand`           | str      | 当前手牌（字符串形式） |
| `my_count`          | int      | 手牌数量               |
| `last_move`         | str      | 上一轮动作             |
| `last_two_moves`    | list     | 上两轮动作             |
| `last_pid`          | int/str  | 上一个出牌玩家         |
| `leading_round`     | bool     | 是否为首轮             |
| `num_cards_left`    | dict     | 各玩家剩余牌数         |
| `enemy_positions`   | list     | 敌人位置               |
| `teammate_position` | str/None | 队友位置               |
| `enemy_min_cards`   | int      | 敌人最少剩余牌数       |
| `teammate_cards`    | int/None | 队友剩余牌数           |
| `dangerous`         | bool     | 是否危险状态           |
| `very_dangerous`    | bool     | 是否极度危险           |
| `is_landlord`       | bool     | 是否为地主             |

### 3.2 贝叶斯信念推断：`infer_belief(infoset, state)`

**功能**：基于已出牌推断未知牌的分布

**核心逻辑**：

```python
def infer_belief(self, infoset, state):
    unknown_counter = self.get_unknown_cards(infoset, state["my_hand"])
    return {
        "unknown_counter": unknown_counter,
        "bomb_prob": self.estimate_bomb_prob(unknown_counter),
        "rocket_prob": self.estimate_rocket_prob(unknown_counter),
    }
```

**关键子函数**：

| 函数                     | 作用               |
| ------------------------ | ------------------ |
| `get_unknown_cards()`    | 计算未知牌的计数器 |
| `estimate_rocket_prob()` | 估算火箭存在概率   |
| `estimate_bomb_prob()`   | 估算炸弹存在概率   |

### 3.3 价值函数：`V(hand)`

**功能**：记忆化 Bellman 价值计算

**数学公式**：

```
V(s) = max_a [ R(s,a,s') + gamma * V(s') ]
```

**实现逻辑**：

```python
def V(self, hand):
    hand = self.sort_card_str(hand)
  
    # 缓存命中
    if hand in self.value_cache:
        self.stats["value_cache_hits"] += 1
        return self.value_cache[hand]
  
    # 状态数量上限检查
    if len(self.value_cache) >= self.cfg["max_dp_states"]:
        return self.cutoff_value(hand)
  
    # 生成所有可能动作
    actions = self.generate_actions_from_hand(hand)
  
    # 递归计算 Q 值
    best_val = -float("inf")
    for action_str in actions:
        next_hand = self.remove_action_from_hand(hand, action_str)
        r = self.mdp_reward(hand, action_str, next_hand)
        q = r + self.cfg["gamma"] * self.V(next_hand)
        best_val = max(best_val, q)
  
    # 缓存结果
    self.value_cache[hand] = best_val
    return best_val
```

### 3.4 奖励函数：`mdp_reward(hand, action_str, next_hand)`

**功能**：计算简化 MDP 中的启发式奖励

**奖励组成**：

| 奖励项        | 计算方式                                       |
| ------------- | ---------------------------------------------- |
| 回合惩罚      | `turn_penalty`                                 |
| 出牌奖励      | `card_reward * 出牌数`                         |
| 结构改善      | `structure_reward * (当前坏度 - 下一状态坏度)` |
| 最小牌奖励    | 若包含最小牌则加 `min_card_bonus`              |
| 顺子奖励      | `chain_bonus + chain_len_bonus * 长度`         |
| 三张奖励      | `trio_bonus`                                   |
| 对子奖励      | `pair_bonus`                                   |
| 四带惩罚      | `four_with_penalty`                            |
| 炸弹/火箭惩罚 | `bomb_use_penalty` 或 `rocket_use_penalty`     |
| 大牌成本      | `-control_use_penalty_weight * control_cost`   |
| 终局奖励      | 若 `next_hand == ""` 则加 `terminal_reward`    |

### 3.5 在线 Q 值：`online_q_value(action, state, belief)`

**功能**：在真实游戏中计算动作的 Q 值（考虑战术上下文）

**关键调整项**：

| 调整项         | 条件                    | 影响                                  |
| -------------- | ----------------------- | ------------------------------------- |
| 首轮过牌惩罚   | `leading_round`         | 返回 `-9999.0`                        |
| 队友保护       | 队友即将出牌且牌数 <= 2 | 过牌奖励 +80                          |
| 危险过牌惩罚   | 敌人危险                | 额外 `-45.0`                          |
| 直接出牌奖励   | `next_hand == ""`       | 加 `finish_bonus_online`              |
| 风险惩罚       | 非终局                  | `-risk_penalty_weight * 被击败风险`   |
| 首轮高单牌惩罚 | 首轮且出单牌            | `-leading_single_high_penalty * 牌值` |
| 首轮炸弹惩罚   | 首轮且出炸弹            | `-leading_bomb_extra_penalty`         |
| 压制队友惩罚   | 跟牌且击败队友          | `-beat_teammate_penalty`              |
| 击败敌人奖励   | 跟牌且击败敌人          | `+enemy_follow_beat_bonus`            |
| 同类型跟随奖励 | 同类型压制              | `+same_type_follow_bonus`             |
| 危险击败奖励   | 危险时击败敌人          | `+enemy_danger_beat_bonus`            |

### 3.6 贪婪基线动作：`greedy_baseline_action(legal_actions, state, belief)`

**功能**：生成稳定的贪婪基线动作，作为 DP 覆盖的基准

**决策逻辑**：

```
1. 优先直接出牌（清空手牌）
2. 首轮：选择改善手牌结构、去除小牌、保留炸弹的动作
3. 跟随队友：过牌（除非能直接出牌）
4. 跟随敌人：优先非炸弹压制；敌人危险时可使用炸弹
5. 未知情况：保守策略（优先过牌）
```

### 3.7 DP 覆盖决策：`should_override_greedy(dp_action, greedy_action, dp_q, state, belief)`

**功能**：决定是否用 DP 动作覆盖贪婪基线

**覆盖条件**：

| 条件                            | 是否覆盖 |
| ------------------------------- | -------- |
| DP 动作 == 贪婪动作             | 是       |
| DP 动作为直接出牌               | 是       |
| DP 优势 < 覆盖阈值              | 否       |
| DP 动作为炸弹且非终局且非危险   | 否       |
| DP 动作风险 > max_override_risk | 否       |
| DP 动作击败队友且非终局         | 否       |

### 3.8 动作生成：`generate_actions_from_hand(hand)`

**功能**：从手牌生成所有可能的分解动作

**生成的动作类型**：

| 类型              | 说明                                  | 示例                        |
| ----------------- | ------------------------------------- | --------------------------- |
| 基础牌型          | 单张、对子、三张、炸弹                | "3", "33", "333", "3333"    |
| 火箭              | 大小王                                | "BR"                        |
| 三带一/三带二     | 三张 + 单张/对子                      | "3334", "33344"             |
| 四带两单/四带两对 | 四张 + 两个单张/两个对子              | "333345", "33334455"        |
| 顺子              | 单顺（>=5）、双顺（>=3）、三顺（>=2） | "34567", "334455", "333444" |
| 飞机带翅膀        | 三顺 + 对应数量的单张/对子            | "33344456", "3334445566"    |

### 3.9 手牌坏度评估：`hand_badness(hand_str)`

**功能**：评估手牌的结构质量（值越低越好）

**计算公式**：

```
badness = turn_weight * 回合数 + 
          single_weight * 单牌数 + 
          len_weight * 手牌长度 - 
          chain_discount_weight * 顺子折扣 - 
          control_discount * 大牌数 - 
          bomb_discount * 炸弹数
```

### 3.10 风险估计：`estimate_beaten_risk(action_str, action_type, belief)`

**功能**：估算动作被对手击败的概率

**风险组成**：

- **同类型风险**：被更大同类型牌型击败的概率
- **炸弹风险**：被炸弹击败的概率
- **火箭风险**：被火箭击败的概率

---

## 四、数据结构与常量

### 4.1 卡牌映射

```python
EnvCard2RealCard = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A", 17: "2",
    20: "B", 30: "R"
}
```

- `T` 代表 10
- `B` 代表小王（Black Joker）
- `R` 代表大王（Red Joker）

### 4.2 卡牌顺序

```python
CARD_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2", "B", "R"]
NORMAL_CHAIN_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
```

### 4.3 位置定义

```python
ALL_POSITIONS = ["landlord", "landlord_down", "landlord_up"]
```

---

## 五、整体决策流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        act(infoset)                                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  extract_state()    │
              │  infer_belief()     │
              └───────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  直接出牌检测？      │──── 是 ────► 返回出牌动作
              └───────────┬─────────┘
                          │ 否
                          ▼
              ┌─────────────────────┐
              │  队友保护规则？      │──── 是 ────► 返回过牌
              └───────────┬─────────┘
                          │ 否
                          ▼
              ┌─────────────────────┐
              │  greedy_baseline_   │
              │  action()           │
              └───────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  构建候选动作集      │
              └───────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  遍历候选动作       │
              │  计算 online_q_value│
              └───────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  should_override_   │
              │  greedy()           │
              └───────────┬─────────┘
                    │           │
                   是           否
                    ▼           ▼
              返回 DP 动作   返回贪婪动作
```

---

## 六、技术亮点总结

### 6.1 简化 MDP 设计

- 将复杂的斗地主 POMDP 简化为仅考虑手牌的 MDP
- 利用手牌减少的特性构建 DAG，避免循环状态
- 记忆化 DP 避免重复计算，提升效率

### 6.2 Greedy-Anchor 安全层

- DP 仅在有显著优势时覆盖贪婪基线
- 防止简化 MDP 做出战术失误（如随意使用炸弹、压制队友）
- 不同场景采用不同覆盖阈值（首轮、跟牌、危险状态）

### 6.3 贝叶斯信念估计

- 基于已出牌推断对手可能的炸弹/火箭
- 风险感知融入决策过程
- 动态调整动作选择策略

### 6.4 多维度奖励设计

- 综合考虑手牌结构、出牌效率、战术价值
- 区分在线和离线奖励，适应不同决策场景
- 通过惩罚机制保护关键牌型（炸弹、火箭、大牌）

注：为了保持胜率，这个值迭代的方法是在greedy的基础上增加，且reward函数也是启发性的，因为用真实值确实难以收敛
