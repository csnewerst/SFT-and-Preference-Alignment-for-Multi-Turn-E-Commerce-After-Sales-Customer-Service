# 1.5B DPO 后审计筛选与失败闭环

> 日期：2026-08-09
>
> 代码提交：训练 `a225cadd3455a1e72402e684f71eb36c8d05ff85`；失败分析 `bb64859`；checkpoint 评测 `e879b8f`
>
> 状态：v1.3.2 DPO 未通过 1.5B 方向门禁，不进入 beta 矩阵或 7B

## 1. 受控设置

- Initial：Qwen2.5-1.5B-Instruct；SFT parent：`sft-r4-all-full-seed42-v1`。
- 三组均从同一个 r4 SFT adapter 初始化，reference 为其冻结副本；首步前 policy/reference log-prob 最大绝对差均为 0.0。
- BF16 LoRA，`lr=5e-6`，weight decay 0.01，有效 batch 16，beta 0.1，seed 42。
- response-only matched：1,571 train / 179 validation，99 steps。
- multigranularity matched：1,571 train / 179 validation，99 steps。
- multigranularity full：4,436 train / 564 validation，278 steps。
- 业务评测固定使用 200 条 screen，IID / Compositional / Challenge = 100 / 50 / 50；贪心解码，最多 6 个工具轮次。
- 所有差值在相同 case 上进行 10,000 次 paired bootstrap，seed 20260809。

## 2. 训练指标不能代表业务效果

| DPO 组成 | train runtime | train loss | eval loss | eval preference accuracy | eval reward margin | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| response-only matched | 426.68 s | 0.0646 | 0.0033 | 1.000 | 7.0018 | 48,317 MiB |
| multigranularity matched | 383.16 s | 0.2111 | 0.1164 | 1.000 | 3.1080 | 45,195 MiB |
| multigranularity full | 1,328.33 s | 0.0799 | 0.0074 | 1.000 | 6.5898 | 51,225 MiB |

三组验证 preference accuracy 均达到 1.0，但 rollout 全部退化。该指标只证明模型区分了当前构造的 chosen/rejected，不能证明它学到了正确业务策略。

## 3. 固定 200 条 screen 结果

| 模型 | task success（95% CI） | 相对 SFT 配对差值（95% CI） | 工具选择 | 参数正确 | 事实忠实 | 禁止工具不触发 |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 13.5% [9.0%, 18.5%] | - | 30.0% | 29.5% | 84.0% | - |
| SFT r4 | 49.5% [42.5%, 56.5%] | 基线 | 61.5% | 61.0% | 92.5% | 82.0% |
| DPO response-only matched | 38.0% [31.5%, 44.5%] | -11.5pp [-18.5, -4.0] | 44.0% | 44.0% | 96.0% | 84.5% |
| DPO multigranularity matched | 28.5% [22.5%, 35.0%] | -21.0pp [-29.0, -12.5] | 33.5% | 33.5% | 96.0% | 93.0% |
| DPO multigranularity full | 25.0% [19.0%, 31.0%] | -24.5pp [-33.0, -15.5] | 32.5% | 32.5% | 97.0% | 96.0% |

三组相对 SFT 的置信区间均完全低于 0。多粒度 full 增加数据和训练步数后没有改善，反而出现 37 条 parse/missing-final failure。

## 4. “出了什么问题 → 为什么 → 试了什么 → 解没解决”

### 4.1 SFT 成功能力被 DPO 破坏

| 组成 | SFT 成功→DPO 失败 | SFT 失败→DPO 成功 | 净变化 | 工具序列发生变化 |
|---|---:|---:|---:|---:|
| response-only | 40 | 17 | -23 cases | 57 / 200 |
| multigranularity matched | 62 | 20 | -42 cases | 108 / 200 |
| multigranularity full | 72 | 23 | -49 cases | 129 / 200 |

多粒度 full 的 72 条回退中，33 条来自 create；回退后的轨迹有 35 条不调用工具、37 条只调用 `query_order_status`。matched 的 62 条回退中同样有 33 条 create，37 条无工具、25 条只查订单。这是明确的动作抑制和多步流程坍缩，不是措辞变化。

### 4.2 是否只是训练过久

复用三个现有 `checkpoint-50`，不重新训练，在相同 200 条 screen 上评测：

| checkpoint | task success | 相对 SFT 配对差值（95% CI） |
|---|---:|---:|
| response-only step 50 | 40.5% | -9.0pp [-16.0, -2.0] |
| multigranularity matched step 50 | 26.5% | -23.0pp [-31.5, -14.5] |
| multigranularity full step 50 | 24.0% | -25.5pp [-33.5, -17.5] |

早停没有消除回退，两个多粒度 checkpoint 在前 50 steps 就已经失去全部 create 成功案例。因此“仅仅 epoch 太多”未被证据支持。

### 4.3 当前根因判断

已确认：

1. 不是 reference 错用 Base；数值前检已通过。
2. 不是 weight decay 漏传；实际落盘为 0.01。
3. 不是只因 full 数据更多；matched 等量实验仍显著回退。
4. 不是只因训练后半程过拟合；checkpoint-50 已显著回退。

最可能但仍需下一版消融验证的根因：

- 静态“chosen 比 rejected 好”门禁没有衡量 pair 对 SFT 当前策略是否过易、是否产生全局停止偏置；
- continue/stop 被全局配平，而不是按环境状态分别约束 `must_continue` 与 `must_stop`，大量 final-response 对可能压制了多步工具策略；
- 合成 rejected 与真实 SFT rollout 的边界错误分布不一致，preference accuracy 很快饱和；
- response 层更新通过共享 LoRA 参数影响了更早的工具决策。

## 5. 决策与下一版数据门禁

本轮停止 beta 0.05/0.3 矩阵，不把失败的数据方向放大到 7B，也不打开 600 development gate 或 formal test v2。

v1.4 优先建设较小但更难的行为保持型偏好集：

1. hard negative 优先来自 SFT 在训练来源池上的真实 rollout 失败，不以明显错误模板为主；
2. `must_continue` 和 `must_stop` 分开建池、分别平衡，不能用全局 Action/Final 50:50 代替条件分布；
3. decision/parameter pair 优先保持 chosen/rejected 目标类型和长度相近，每对只引入一个可验证行为错误；
4. 增加 create 三步链路、identity、not-delivered 和跨轮参数保持的最低覆盖门禁；
5. 对 chosen 用模拟器 replay，且检查它与 SFT 原正确轨迹一致；
6. 先构造 500～1,000 对高质量 v1.4，在 10/25/50 steps 保存 checkpoint；先用开发 screen 的小分层子集判断方向，只有候选不回退才跑完整 200 条；
7. 只有相对 SFT 的 task success 配对区间不再显著为负，且 create、事实忠实、禁止调用没有关键退化，才进一步比较 beta 或扩大数据量。

## 6. 远端证据

- 统一比较：`experiments/local/1p5b/dpo_postaudit_screen_comparison_v2.json`
- 失败迁移：`experiments/local/1p5b/dpo_postaudit_failure_analysis_v2.json`
- checkpoint-50 比较：`experiments/local/1p5b/dpo_postaudit_checkpoint50_comparison_v2.json`
- 训练资源聚合：`experiments/local/1p5b/dpo_postaudit_training_summary_v2.json`
- 每个 run 内保留 manifest、command、environment、GPU 采样、reference preflight、训练日志、adapter、trace 和逐样本评测。

这些产物位于 AutoDL 数据盘并被 Git 忽略；Git 跟踪训练/评测/失败分析脚本和本报告。

## 7. v1.4 候选池审计（2026-08-10）

新增冻结 SFT 难度打分后，4,436 条训练候选仅有 310 条 `mean_logp_margin <= 0`，占 6.99%，
且全部集中于 `must_stop`。`must_continue`、`wrong_action`、`parameter`、`response` 四桶的
`margin <= 0.5` 数量均为 0。原计划筛出的 720 train / 80 validation 虽满足行为配额和父样本上限，
但 train 中仅 72 条真正难例，仍全部为 `must_stop`。

这解释了为什么“减少到 2,000--3,500 对”不能只做数量截断：若直接从当前 5,000 对扩到 2,500 对，
只会补入更多已被 SFT 轻易区分的合成负例。本轮停止 800 对训练，下一版改用训练来源上的冻结 SFT
真实 rollout 失败挖掘 hard negative；只有五个行为桶均具备足够边界错误，才恢复 1.5B 方向实验。

## 8. 真实 rollout 挖掘闭环（2026-08-10）

冻结 SFT r4 在 CSDS/DCH-2 train-only 的 2,000 条 mining 场景上得到 48.75% task success，
并从 1,025 个失败中提取 1,024 个首分歧 pair。逐对重新打分后，train 中 66.02% 的 pair
满足 `mean_logp_margin <= 0`；must-stop、parameter、response 三桶分别有 401/401、32/33、
175/176 条满足该条件。相比旧池仅 6.99% 且全部集中于 must-stop，真实 rollout 明显提供了更有效的边界错误。

本次没有观察到可用的首分歧 wrong-action pair，因此不为原定 10% 配额加入模型已经轻易区分的合成错误。
1.5B 方向集改为 continue/stop 各 36%、parameter 4%、response 24%，共 720/80 对，
在 10/25/45 step checkpoint 上检查是否保住 create、事实忠实和多步工具链。

## 9. v1.4 方向门结果与 checkpoint 选择（2026-08-10）

使用修复否定句事实识别后的 evaluator v2，对相同 200 条 screen 重新评测并进行
10,000 次配对 bootstrap（seed `20260809`）：

| 模型 | task success | 相对 SFT 配对差值（95% CI） | 工具选择 | 参数正确 | 事实忠实 | 禁止工具不触发 |
|---|---:|---:|---:|---:|---:|---:|
| SFT r4 | 49.5%（99/200） | 基线 | 61.5% | 61.0% | 92.5% | 82.0% |
| DPO step 10 | **58.0%（116/200）** | **+8.5pp [+3.0, +14.0]** | 71.0% | 69.0% | 97.0% | 83.5% |
| DPO step 25 | 49.0%（98/200） | -0.5pp [-7.5, +6.5] | 61.0% | 59.0% | 97.5% | 73.5% |
| DPO step 45 | 46.5%（93/200） | -3.0pp [-10.0, +4.0] | 58.5% | 56.5% | 97.0% | 70.0% |

step 10 相对 SFT 的配对区间完全高于 0，并同时提高工具选择、参数正确和事实忠实，
因此通过 1.5B DPO 方向门。step 25 相对 step 10 回退 9.0pp（95% CI
[-15.5, -3.0]），step 45 回退 11.5pp（95% CI [-18.5, -4.5]），表明继续训练会放大
不必要的三工具链和错误参数调用，不能用最终 checkpoint 代替业务指标选模。

step 10 共发生 26 条失败转成功和 9 条成功转失败，净增加 17 条成功案例。9 条回退主要是
5 条提前停止无工具、3 条仅查询订单、1 条查询后检查政策；错误集中在未完成解决（7 次），
另有错误工具、错误参数、意外工具结果和无依据状态断言。它们不是总体方向失败，但必须作为
7B 短程校准的阻断性回归清单：若 7B 候选再次触发同类案例，不得仅凭总体均值晋级。

决策：冻结 1.5B 胜出点为 `checkpoint-10`，不继续做 1.5B beta/epoch 扩展，也不增加 3B。
下一阶段只在 7B 上做 1～2 个短程校准候选；配置冻结前仍不打开 formal test v2。

远端证据位于：

- `experiments/local/1p5b/dpo-v1p4-rollout-quality-screen800-beta0p1-seed42-from-r4-v1/analysis_negation_v2/paired_comparison.json`
- `experiments/local/1p5b/dpo-v1p4-rollout-quality-screen800-beta0p1-seed42-from-r4-v1/analysis_negation_v2_detailed/failure_transitions.json`
- 三个 checkpoint 的逐样本 trace 与 evaluator v2 汇总保留在同一 run 目录中。
