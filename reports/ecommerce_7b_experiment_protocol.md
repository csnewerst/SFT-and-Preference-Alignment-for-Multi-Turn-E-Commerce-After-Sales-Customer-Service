# Qwen2.5-7B 主实验与短程校准协议 v1

> 状态：**待人工评审，禁止启动实验**。当前仅下载模型文件并准备脚本；以下数值是预注册配置，
> 不是实验结果。只有本方案经用户确认后，才能执行 Initial、SFT 或 DPO。

执行授权已于2026-08-10确认。formal test v2已构建并保持未打开状态：600条，自动oracle replay 600/600，
与200 screen、已使用的600 gate及SFT/DPO训练文件无身份或精确消息重叠；manifest SHA256为
`76285f06b871fb40f8bcad69986c9ea99c203df60fe23be567e3d7318d0ba74a`。

7B BF16 LoRA前检已通过：模型聚合SHA256为
`4b0ff1363802cb9711c38f3f511bd813c46f174da0e04cc1d142276ada48223a`，batch 2、长度1024、
rank 8 all-linear真实前向/反向峰值显存38,644.64MiB，无OOM。

## 0. 本阶段要回答的问题

1. 7B Initial 相比 1.5B Initial 是否已有更强的零样本工具调用能力？
2. 7B SFT 是否稳定学习三工具、多轮状态与政策约束，而不是固定调用某个工具？
3. 1.5B 胜出的 rollout-quality DPO 数据方向能否迁移到 7B？最佳更新强度是否仍位于很早的 checkpoint？
4. SFT+DPO 是否在任务成功和自动解决率上超过 SFT，同时不损害事实忠实和政策合规？

## 1. 为什么先做短程校准

1.5B 已证明真实 rollout 首分歧偏好数据在短程 DPO 上方向正确，但 step 25/45 出现显著回退。
该结论用于排除失败方向和限定搜索范围，不能证明同一 LoRA rank、学习率和最佳 step 可直接迁移到 7B。
因此 7B 只比较两个 SFT 容量候选，并在胜出 SFT 上比较三个早期 DPO checkpoint。

## 2. SFT 校准与主训练

- 模型：Qwen2.5-7B-Instruct，BF16 LoRA，不量化，不使用 DeepSpeed；每个候选使用一张 A800。
- 固定项：all-linear、alpha/rank=2、dropout 0.05、有效 batch 32、seed/data seed 42、最大长度 1024。
- 候选：rank 8 与 rank 16；统一 `lr=1e-5`、warmup 3 steps，各训练 100 steps。
- 选择：相同 200 条开发 screen 上比较总体与 IID/Compositional/Challenge、create、参数正确、事实忠实和禁止调用。
- 胜出 rank 从原始 7B 重新执行完整 1 epoch，避免把校准 checkpoint 当正式模型。
- 第 2 epoch 不是默认项：只有 validation loss 与开发错误切片均继续改善、且无阻断性回退时才执行；不计划第 3 epoch。

## 3. DPO 校准

- 从选定的完整 SFT adapter 初始化 policy，并复制冻结 adapter 作为 reference；首 step 前必须通过 log-prob 等价检查。
- 使用 720/80 的 rollout-quality DPO v1.4，不扩大数量。
- 主候选：`beta=0.1`、`lr=2e-6`，保存并评测 step 5/10/20。
- 若出现过度调用或策略漂移，依次选择更早 checkpoint、将学习率降至 `1e-6`、必要时增大 beta 至 `0.2`。
  `beta=0.05` 只作为探索性对照，不能称为“更保守”的回退方案。
- 选择依据是逐样本 rollout 配对结果，不以 DPO preference accuracy 或训练 loss 选模。

## 4. 阻断条件

任一候选出现以下情况时不得晋级：事实忠实或禁止工具明显下降；create 三步链退化；提前停止增加；
不必要的三工具链增加；或重复触发 1.5B checkpoint-10 的 9 条已知回退案例。开发 screen 可反复用于选模，
但 formal test v2 在 Initial/SFT/SFT+DPO 配置和随机种子全部冻结前保持密封。

## 5. 正式结果矩阵

配置冻结后，在同一 formal test v2 上一次性比较 Initial、SFT、SFT+DPO，并报告：任务成功率、
自动解决率、工具选择与参数正确率、事实忠实、政策合规、延迟、峰值显存、逐案例结果、paired bootstrap
95% CI 以及“问题→原因→尝试→是否解决”的失败案例闭环。若预算允许，再用第二随机种子复验 SFT 与 SFT+DPO。

## 6. 执行顺序与预计时间

时间均为四张 A800-80GB 当前环境下的保守预估，首个实测 run 后必须用真实吞吐更新，不能写成实测结果。

| 阶段 | GPU 使用 | 工作内容 | 预计墙钟时间 | 产出/决策 |
|---|---:|---|---:|---|
| A. 前检 | 1 卡短时 | 权重完整性、tokenizer、单 batch 前后向、OOM 与 manifest 检查 | 10～20 分钟 | 通过才允许训练 |
| B. Initial screen | 1 卡 | 7B Initial 在固定 200 条开发 screen 上 rollout | 20～60 分钟 | 零样本基线，不参与调参 |
| C. SFT 校准 | 2 卡并行 | rank 8/16，各 100 steps | 20～60 分钟 | 选一个 rank；不把校准 adapter 当正式模型 |
| D. SFT 校准评测 | 2 卡并行 | 两个候选在相同 200 条 screen 上 rollout、配对比较 | 30～90 分钟 | 冻结 SFT rank |
| E. 正式 SFT | 1 卡 | 从原始 7B 训练完整 1 epoch | 1～3 小时 | 生成正式 SFT；按门禁决定是否延长到 2 epoch |
| F. DPO 校准 | 1 卡 | beta 0.1、step 5/10/20；必要时才补 beta 0.05 | 20～60 分钟 | 冻结 DPO checkpoint/beta |
| G. 开发集闭环 | 1～3 卡 | Initial/SFT/SFT+DPO 同屏比较与失败迁移分析 | 1～3 小时 | 冻结全部正式配置 |
| H. Formal test v2 | 1～3 卡 | 密封集一次性统一评测 | 1～3 小时 | 项目最终主结果 |
| I. 复验（可选） | 2 卡并行 | seed `20260809` 的 SFT 与 SFT+DPO | 2～6 小时 | 验证结果不是单一种子偶然性 |

不使用多卡 DeepSpeed：7B BF16 LoRA 和当前 DPO reference 方案可在单张 80GB A800 内完成；四卡用于并行独立候选，
避免把分布式差异混入模型效果。若单 batch 前检意外 OOM，再单独评估 gradient checkpointing、micro batch 或 ZeRO，
不能静默改变实验定义。

## 7. 预注册参数

### 7.1 SFT 校准

| 参数 | 固定值 |
|---|---|
| base model | `Qwen/Qwen2.5-7B-Instruct` |
| precision / quantization | BF16 / none |
| LoRA | all-linear；rank 8 vs 16；alpha 16 vs 32；dropout 0.05 |
| train budget | 100 steps |
| optimizer schedule | lr `1e-5`；cosine；warmup 3；weight decay 0.01 |
| batch | micro batch 2；gradient accumulation 16；effective batch 32 |
| sequence | max length 1024；训练集审计最大 851，预期无截断 |
| randomness | seed 42；data seed 42 |

### 7.2 正式 SFT

沿用胜出 rank 和上述固定项，从原始 7B 重新训练 1 epoch。若考虑第 2 epoch，必须先记录 epoch 1 的完整结果，
并满足：validation loss 未恶化、screen 总体或关键错误切片继续改善、事实忠实/禁止调用/create 无阻断回退。
不执行 3 epoch，因为当前 10,800 条数据下缺乏支持该预算的证据。

### 7.3 DPO 校准

| 参数 | 固定值/候选 |
|---|---|
| initialization | 正式 SFT LoRA adapter |
| reference | SFT adapter 冻结副本；首 step 前 policy/reference log-prob 最大差需 `<=1e-4` |
| data | v1.4 rollout-quality，720 train / 80 validation |
| precision | BF16 LoRA，继承 SFT rank/targets |
| primary | beta 0.1；lr `2e-6`；step 5/10/20 |
| drift mitigation | 更早 checkpoint → lr `1e-6` → 必要时 beta `0.2` |
| exploratory only | beta `0.05`，不解释为保守回退 |
| batch / randomness | effective batch 16；seed 42 |

`lr=2e-6` 是 7B 的保守校准起点，不宣称由 1.5B 得到最优；它必须由短程 rollout 结果验证。

## 8. 指标定义与选模规则

### 8.1 主指标

- `task_success_rate`：模拟器状态、工具序列、参数、回答要求、事实和步数约束全部通过。
- `eligible_auto_resolution_rate`：仅在业务规则允许自动处理的样本中，完整任务成功且无需人工介入的比例。
  当前项目没有真实线上人工转接日志，因此不能把该离线指标包装成线上“自动解决率提升”。
- 相同样本上的 paired bootstrap 差值，10,000 次重采样，seed `20260809`，报告 95% CI。

### 8.2 必报诊断指标

工具选择、参数正确、observation 结果一致、事实忠实、禁止工具不触发、回答要求、状态断言、步数限制；
同时按 IID/Compositional/Challenge 和 status/policy/create/missing-order/duplicate/identity/not-delivered/
timeout/expired/anti-hallucination 分层报告。

### 8.3 晋级规则

- SFT：相对 Initial 的任务成功率明确提高；rank 选择优先看配对差值和关键切片，不以 train loss 最低选模。
- DPO：相对 SFT 的配对差值不得显著为负；优先选择最早达到收益的平台 checkpoint。
- 若总体提高但事实忠实、禁止调用、create 三步链或 9 条已知回退案例出现关键退化，候选不得晋级。
- 200 条 screen 是开发集，可以用于选择；formal test v2 只在配置冻结后打开一次。

## 9. 可复现证据清单

每个 run 必须保留并汇总：

- Git commit/dirty 状态、完整命令、配置 JSON、模型 revision 与权重聚合 SHA256；
- SFT/DPO 数据目录聚合 SHA256、样本数、token 长度与截断统计；
- Python、PyTorch、CUDA、Transformers、TRL、PEFT 版本和四卡型号；
- seed/data seed、LoRA 参数、优化器、学习率、warmup、batch、训练步数/epoch；
- train/eval loss、DPO reward chosen/rejected/margin/accuracy，但明确它们不是业务效果；
- 每 5 秒 GPU 利用率、显存、功耗、温度，峰值显存与墙钟时间；
- 固定推理参数、逐样本 trace、逐样本评测、聚合指标、paired CI 与失败迁移表；
- 失败案例记录：`出了什么问题 → 根因证据 → 尝试的修改 → 是否解决 → 是否引入新回退`。

## 10. 人工确认点

执行被明确暂停在本方案评审处。用户确认本 Markdown 后，才依次执行 A～D；SFT rank 的结果和失败案例
会再次汇报，得到方向结论后再进入 E～G。formal test v2 不因开发集失败而提前打开。
