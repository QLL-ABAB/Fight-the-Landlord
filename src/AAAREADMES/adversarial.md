# Adversarial Agent 探究路径

这个 agent 的核心目标不是学习一个神经网络，而是用搜索来处理斗地主中的对抗决策。整体思路可以概括为：

```text
隐藏牌采样 + 多次确定化搜索 + baseline 保护 + 有限理性对手模型
```

它要解决的问题是：当前玩家只能看到自己的手牌和公共出牌历史，但如果要做对抗搜索，就必须假设另外两家的手牌是什么。因此我们需要在不完全信息下近似地做 minimax。

## 1. 基础逻辑：隐藏牌采样下的 Minimax 搜索

最直接的想法是：以当前决策者为根节点，向下展开未来动作。

如果是完全信息游戏，可以直接写成 minimax：

```text
V(s) = max_a V(T(s, a))                 我方节点
V(s) = min_a V(T(s, a))                 对手节点
```

或者把队友也看成我方：

```text
我方 / 队友节点：max
敌方节点：min
```

但是斗地主是不完全信息游戏。当前玩家不知道另外两家的具体手牌，所以不能直接对真实状态 `s` 做 minimax。我们能看到的只是信息集：

```text
I = (自己的手牌, 已出牌历史, 上一手牌, 剩余牌数, 当前身份)
```

因此我们使用 **determinization**：把未知牌随机分配给另外两名玩家，得到一个“可能的完整牌局”：

```text
I -> s_i
```

在这个采样出来的完整状态 `s_i` 上，就可以暂时当作完全信息游戏做搜索。

单次随机分配误差很大，因为一次采样出来的对手手牌可能并不接近真实情况。因此我们重复采样多次，对每个候选动作求平均价值：

```text
Q(a) = 1/N * sum_i Search(T(s_i, a), d)
a* = argmax_a Q(a)
```

其中：

- `N` 是隐藏牌采样次数。
- `s_i` 是第 `i` 次随机补全出的完整牌局。
- `d` 是搜索深度。
- `Search` 是之后的对抗搜索或 rollout 估值。

这个方法本质上很像 **Monte Carlo Determinization**，也就是在不完全信息游戏中，把隐藏信息多次采样成完整状态，再在这些完整状态上搜索，最后对结果取平均。

在代码中，对应逻辑大致是：

```text
determinization()
return_adversarial_depth_action()
_adversarial_value()
```

## 2. Baseline 保护：避免搜索把结果变差

实现基础对抗搜索后，我们发现一个问题：搜索并不一定稳定优于原来的 greedy / high-rank 策略。

原因是搜索里有很多近似：

- 隐藏牌是随机分配的，不一定真实。
- 搜索深度有限，看不到很远的后果。
- 每个节点只保留 top-k 动作，动作空间被剪枝。
- 叶子节点仍然依赖启发式或 rollout 估值。
- 敌方是否真的会按搜索假设行动并不确定。

因此，如果完全相信搜索结果，可能会出现：

```text
搜索觉得某个动作好，但其实是采样误差或叶子估值误差导致的。
```

所以我们加入了 baseline 保护。baseline 就是原高排名 bot 给出的第一候选动作。对抗搜索只有在“明显更好”时才覆盖 baseline。

可以写成：

```text
a_search = argmax_a Q_search(a)
a_base   = high_rank_policy(I)

if Q_search(a_search) - Q_search(a_base) >= margin:
    return a_search
else:
    return a_base
```

同时也会比较第一名和第二名的差距：

```text
if Q_best - Q_second >= margin_second:
    accept best
else:
    fallback to baseline
```

这个设计的作用是：搜索只在证据足够强时介入，否则保持原策略。这样可以避免不稳定的浅层搜索把本来还可以的 greedy/high-rank 决策带偏。

在代码中，对应的是：

```text
use_root_gate
accept_margin_vs_second
accept_margin_vs_baseline
rootActions[0]
```

实验上，这种 baseline gate 能让胜率更稳定。它不是让搜索每次都“更激进”，而是让搜索成为一个有保护的改进层。

## 3. 对手节点策略：从纯 Minimax 到 Softmin

在 baseline 后，胜率有一定改善。接下来我们继续分析搜索树内部的决策假设。

最初直觉是：

```text
自己和队友：max
敌人：min
```

也就是：

```text
V(s) = max_a V(T(s, a))       我方/队友
V(s) = min_a V(T(s, a))       敌方
```

但是这个假设在斗地主中并不完全合理。原因是：对手和队友也不知道完整牌局。我们在搜索中使用的完整状态 `s_i` 是当前 agent 采样出来的，但其他玩家真实决策时并不知道这个采样状态。

如果敌方节点直接取 `min`，就等于假设敌人：

```text
知道所有人的手牌
每一步都完美选择最克制我们的动作
```

这会过于悲观。尤其在斗地主中，两个农民虽然同队，但他们并不是一个共享全部信息的完美联合体；地主也不知道农民的具体牌。因此，纯 minimax 会把对手建模得过强。

另一种极端是平均：

```text
V(s) = average_a V(T(s, a))
```

但这又太乐观，因为敌人不会随机乱走。

所以我们采用折中策略：**softmin**。它表示敌人倾向于选择对我们不利的动作，但不是永远选最坏动作。

公式可以写成：

```text
w_a = exp(-V(T(s,a)) / tau) / sum_b exp(-V(T(s,b)) / tau)
V(s) = sum_a w_a V(T(s,a))
```

其中：

- `tau` 是温度参数。
- `tau` 越小，越接近 `min`。
- `tau` 越大，越接近平均。

这个模型可以理解为 **有限理性对手模型**：对手大概率会选对我们不利的动作，但不假设对手全知全能。

在代码中，对应参数是：

```text
enemy_node_mode = "softmin"
enemy_softmin_temp
```

如果需要做对比，也可以切换成：

```text
enemy_node_mode = "min"
enemy_node_mode = "policy_expectation"
```

当前选择 softmin 的原因是：它在纯 min 的过度悲观和平均策略的过度乐观之间做了折中，更符合不完全信息斗地主中的实际对抗关系。

## 总结

这个 adversarial agent 的探究路径是：

```text
1. 先尝试 minimax，但斗地主隐藏信息导致无法直接搜索真实状态。
2. 用 determinization 随机补全隐藏牌，并多次采样求平均。
3. 发现搜索不稳定，于是加入 high-rank baseline gate 防止变差。
4. 发现纯 min 假设敌人过强，于是用 softmin 表示有限理性对手。
```

因此，它不是一个标准完全信息 minimax，也不是纯 Monte Carlo rollout，而是一个针对斗地主不完全信息特点做出的近似搜索方法。
