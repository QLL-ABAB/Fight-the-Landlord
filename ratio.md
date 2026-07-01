# DouZero 与 Approx 系列 Episode 换算比例

## 统一口径

为了和 ApproxQ / ApproxDouFeature 的 `episode` 做对比，这里把 DouZero checkpoint 文件名中的训练 `frames` 换算成项目里用于对齐的 episode。

需要注意两个不同概念：

```text
真实平均出牌步数：avg_steps_per_episode ≈ 40
DouZero 对齐用 frame multiplier：STEP_MULTIPLIER = 60
```

DouZero 训练脚本 `src/train_douzero_gpu.sh` 使用的是：

```text
TOTAL_FRAMES = EPISODES * STEP_MULTIPLIER
STEP_MULTIPLIER = 60
```

所以画图对齐时应该使用 `60`，不是 `40`。

因此：

```text
DouZero 等效 episode = DouZero checkpoint frame / 60
Approx 等效 episode = checkpoint 文件名中的 episode
```

反过来：

```text
1 个 Approx episode ≈ 60 个 DouZero frame
```

## DouZero 换算

| DouZero checkpoint step | 等效 episode | 说明 |
|---:|---:|---|
| 3,001,600 | 50,027 | 早期 checkpoint |
| 6,003,200 | 100,053 | 早期 checkpoint |
| 12,000,000 | 200,000 | 中早期 checkpoint |
| 24,000,000 | 400,000 | 中期 checkpoint |
| 30,000,000 | 500,000 | 中期 checkpoint |
| 40,000,000 | 666,667 | 中后期 checkpoint |
| 60,000,000 | 1,000,000 | 和 Approx 1M episode 对齐 |

## 与几个 Approx 版本的 episode 比例

| 模型版本 | checkpoint / 训练规模 | 等效 episode | 相对 DouZero 60M |
|---|---:|---:|---:|
| `approxq_logadp_cmp_1m_history` | 1,000,000 episode | 1,000,000 | 1.00x |
| `approxq_logadp_best_landlord_finetune_50k` | 750,000 + 50,000 | 800,000 | 0.80x |
| `approxq_logadp_best_landlord_finetune_50k_time_equal` | 750,000 + 50,000 | 800,000 | 0.80x |
| `approx_doufeature_logadp_td_1m` | 1,000,000 episode | 1,000,000 | 1.00x |
| `approx_doufeature_logadp_td_1m_gamma` | 1,000,000 episode | 1,000,000 | 1.00x |
| `approx_doufeature_logadp_mc_1m` | 1,000,000 episode | 1,000,000 | 1.00x |
| `approx_doufeature_logadp_td_buffer_1m` | 1,000,000 episode | 1,000,000 | 1.00x |
| `approx_doufeature_logadp_mc_adv_buffer_1m` | 1,000,000 episode | 1,000,000 | 1.00x |

关键结论：

```text
DouZero 60M frame ≈ 1M episode
Approx 1M episode ≈ DouZero 60M frame
DouZero 60M : Approx 1M ≈ 1 : 1
```

所以如果横轴想按环境交互量公平比较：

- Approx 1,000,000 episode 应该和 DouZero 60,000,000 frame 对齐。
- 图里 DouZero 横轴应该使用 `frame / 60`。
- 如果图里使用 `frame / 40`，会把 DouZero 60M 画到 1.5M episode，导致 DouZero 横轴偏大。

## Buffer 版本的额外学习强度

上面的比例只比较环境交互量，也就是实际打了多少局。

buffer 版本虽然还是 `1,000,000 episode`，但每局之后会从 replay buffer 中重复抽样学习，因此 learner 更新量更大。

当前 buffer 配置为：

```text
num_workers = 4
worker_episodes = 8
learn_batch_size = 4096
learn_steps = 10
```

每轮采样：

```text
采样 episode = 4 * 8 = 32
学习 transition 数 = 4096 * 10 = 40960
平均每个 episode 的 learner 更新数 ≈ 40960 / 32 = 1280
```

普通非 buffer 版本平均每局约：

```text
avg_steps_per_episode ≈ 40
每个 episode 约 40 次 transition 更新
```

因此 buffer 版本的学习更新强度约为：

```text
1280 / 40 ≈ 32x
```

这解释了为什么：

- `td_1m` / `mc_1m` 环境交互是 1M episode。
- `td_buffer_1m` / `mc_adv_buffer_1m` 环境交互仍是 1M episode。
- 但 buffer 版本的 learner 更新量大约是非 buffer 版本的 32 倍，所以训练时间会显著变长。

## 汇报时建议使用的说法

如果比较“打了多少局”：

```text
DouZero 60M frame ≈ 1M episode，Approx 1M episode 约等于 DouZero 60M frame。
```

如果比较“学习更新强度”：

```text
非 buffer Approx 每局约 40 次 transition 更新；buffer Approx 每局约 1280 次 transition 更新，学习强度约为 32 倍。
```

如果画训练曲线：

```text
DouZero 横轴 = checkpoint_frame / 60
Approx 横轴 = checkpoint_episode
```
