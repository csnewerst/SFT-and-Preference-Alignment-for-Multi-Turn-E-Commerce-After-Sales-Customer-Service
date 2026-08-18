# DPO v1.4 质量优先数据规范

## 结论

正式 DPO 不再以 5,000～10,000 对为硬目标。结合 v1.3.2 在 1.5B 上的显著回退，v1.4 采用：

- 方向筛选集：800 对（720 train / 80 validation）；
- 通过方向门禁后的正式候选：目标 2,500 对（2,250 / 250）；
- 可接受总量：2,000～3,500 对；
- 不允许用随机、模板化或 preference accuracy 已饱和的容易 pair 补足数量。

## 行为分桶

| 分桶 | 比例 | 目的 |
|---|---:|---|
| must_continue | 25% | observation 后仍需继续调用工具，反对提前回复 |
| must_stop | 10% | 已有充分结论或禁止操作时停止，反对多余调用 |
| wrong_action | 10% | 在相同状态下选择正确工具而不是近似错误工具 |
| parameter | 25% | 保持工具不变，只比较参数与跨轮实体 |
| response | 30% | 比较事实、政策和下一步，不承担工具决策教学 |

`must_continue` 与 `must_stop` 分开配额，不能再用全局 Action/Final 50:50 代替条件分布。

## 难度筛选

冻结入选 SFT adapter，对每对 completion 计算长度归一化 log-prob：

```text
mean_logp_margin = chosen_mean_logp - rejected_mean_logp
```

每个行为分桶内优先选择 margin 最低的 pair，即 SFT 当前真正难分、甚至错误偏好 rejected 的近失误。构建器拒绝缺少 `metadata.sft_hardness` 的输入，避免退回随机抽样。

每个父样本在一个 split 最多保留 2 对，防止单条公开对话扩展出的多个工具位置支配训练分布。

## 实验门禁

1. 先在 800 对方向集上运行 10/25/50 steps checkpoint。
2. 优先用较小分层开发集排除明显回退，只让存活 checkpoint 跑完整 200 screen。
3. 相对同一 SFT 的 paired task-success 区间显著为负时立即停止。
4. create、事实忠实、禁止工具和 premature-stop 任一关键回退都阻止扩量。
5. 只有方向通过后才构建约 2,500 对正式训练候选；formal test v2 仍只在 7B 配置冻结后打开。

## 2026-08-10 冻结 SFT 难度审计结论

对 v1.3.2 的 4,436 train / 564 validation 候选逐对计算冻结 SFT r4 的长度归一化
`chosen - rejected` log-prob margin。审计发现：

| train 行为桶 | 候选数 | margin <= 0 | margin <= 0.5 | margin p50 |
|---|---:|---:|---:|---:|
| must_continue | 739 | 0 | 0 | 1.570 |
| must_stop | 879 | 310 | 713 | 0.208 |
| wrong_action | 138 | 0 | 0 | 1.468 |
| parameter | 1,109 | 0 | 0 | 1.073 |
| response | 1,571 | 0 | 0 | 2.303 |

全池只有 310 / 4,436（6.99%）是 margin <= 0 的难例，且全部属于 `must_stop`。
按固定比例选出的 720 条训练 screen 中，72 条 margin <= 0，也同样全部属于 `must_stop`；
其余四桶即使取各桶最低 margin，也仍是 SFT 已明显偏好 chosen 的容易 pair。

因此当前 800 对只能称为“分桶平衡、按难度排序的合成候选”，不能称为高质量 hard-negative 集。
直接训练会让有效梯度偏向停止工具调用，存在复现此前 action suppression 的风险。本轮阻止训练与 2,500 对扩量，
改为从 CSDS、DCH-2 的 train split 构造独立 rollout mining 场景，让冻结 SFT 实际生成错误，再在首次分歧位置构造
`must_continue / must_stop / wrong_action / parameter / response` 偏好对。screen 一轮 720 train 对应约 45 个有效步，
checkpoint 改为 10 / 25 / 45。

## 2026-08-10 真实 rollout hard-negative 结果

从 CSDS、DCH-2 的 train split 表达骨架构造 2,000 条独立 mining 场景，冻结 1.5B SFT r4
以 greedy、最多 6 步执行真实工具环境：

- task success 975 / 2,000（48.75%）；失败 1,025 条；
- 首分歧成功构造 1,024 对，只有 1 条多工具同轮复杂分歧被保守跳过；
- train / validation = 921 / 103，父样本全部唯一；
- train 中 608 / 921（66.02%）为 `mean_logp_margin <= 0`，显著高于旧合成池的 6.99%。

| train 行为桶 | 数量 | margin <= 0 | margin <= 0.5 | margin p50 |
|---|---:|---:|---:|---:|
| must_continue | 311 | 0 | 72 | 0.653 |
| must_stop | 401 | 401 | 401 | -2.625 |
| wrong_action | 0 | 0 | 0 | - |
| parameter | 33 | 32 | 33 | -0.159 |
| response | 176 | 175 | 176 | -0.935 |

`wrong_tool` 是完整序列评测错误，不等价于“第一步选择了另一工具”；真实首分歧中没有发现可用的
wrong-action near miss。因此 1.5B screen 不用容易模板硬凑该桶，临时比例修订为：
`must_continue 36% / must_stop 36% / wrong_action 0% / parameter 4% / response 24%`。
这既保持 continue/stop 对称，也恰好能从现有池选择 720 train / 80 validation。

自然语言 chosen 使用 10 类事实受限模板，每个模板均通过 simulator/evaluator oracle replay。
人工复核单位从 641 条 response pair 降为最多 10 个唯一模板，只检查语言自然度；动作、参数、事实和状态由程序验证。
