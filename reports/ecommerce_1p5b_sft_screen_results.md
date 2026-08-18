# 1.5B SFT 筛选实验结果

## 1. 结论

固定 200 条 screen 集上的 Initial 基座任务成功率为 13.5%。三组 all-linear LoRA 均带来显著提升，其中 r4 与 r16 的总体成功率相当，因此二者进入独立 600 条 gate；r64 没有表现出与四倍适配器成本相匹配的收益，q/v-only 则明显欠拟合。

本阶段只用于选择 1.5B 的训练方案和验证业务链路，不把结果外推为 7B 正式模型效果。最终 SFT 方案由 600 条 gate 决定，screen 不重复用于最终报告。

## 2. 可复现设置

- 基座模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 本地模型目录：`models/base/Qwen2.5-1.5B-Instruct`
- `model.safetensors` SHA256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`
- SFT 数据：10,800 train / 1,200 validation
- 数据根目录：`data/ecommerce/domain_train_v1_3_2_zh`
- 主随机种子：42；`data_seed=42`
- 精度：BF16
- 学习率：`2e-5`，cosine scheduler，warmup 10 steps，weight decay 0.01
- epoch：1；有效 batch size：32
- 最大长度：1,024；实测 SFT 最大 851 tokens，截断率 0
- LoRA dropout：0.05
- 并行方式：4 个单卡独立实验并行；未使用 DeepSpeed
- screen：200 条，IID / Compositional / Challenge = 100 / 50 / 50
- screen cases SHA256：`66c06ab29ffcabd6dbad53e7a3fa306549338de0eaab9cb3644efb05d7d30439`
- 解码：greedy，`do_sample=false`，`max_new_tokens=512`，最多 6 个工具轮次
- bootstrap：10,000 次，种子 20260809；模型差异使用相同 case 的配对重采样
- 训练代码提交：`a8da40d`

每个训练目录保存 `manifest.json`、`command.sh`、`environment.txt`、`git_status.txt`、训练日志、GPU 采样、adapter 和评测逐样本结果。

## 3. 训练与验证集结果

| 配置 | LoRA targets | 可训练参数 | adapter 大小 | 训练耗时 | 吞吐 samples/s | train loss | validation loss | PPL | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r4 all | all-linear | 4,616,192 | 18,516,064 B | 1,124.1 s | 9.608 | 0.8144 | 0.4480 | 1.5652 | 39,019 MiB |
| r16 all | all-linear | 18,464,768 | 73,911,112 B | 1,116.6 s | 9.673 | 0.3469 | 0.0487 | 1.0499 | 39,219 MiB |
| r64 all | all-linear | 73,859,072 | 295,488,936 B | 1,117.7 s | 9.663 | 0.1841 | 0.0400 | 1.0408 | 39,539 MiB |
| r16 q/v | q_proj,v_proj | 2,179,072 | 8,731,128 B | 742.8 s | 14.539 | 1.1585 | 0.9463 | 2.5762 | 38,867 MiB |

验证 loss 随 rank 增大而下降，但该排序没有转化为 screen 业务成功率。因此模型选择不能只依赖 token-level loss。

## 4. 固定 200 条 screen 业务结果

| 配置 | 任务成功率（95% CI） | 相对 Initial 配对提升（95% CI） | 工具选择 | 参数正确 | 事实忠实 | 状态断言 |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 13.5% | 基线 | 30.0% | 29.5% | 84.0% | 81.5% |
| r4 all | 49.5% [42.5%, 56.5%] | +36.0pp [+27.0, +45.0] | 61.5% | 61.0% | 92.5% | 97.5% |
| r16 all | 49.0% [42.0%, 56.0%] | +35.5pp [+27.0, +44.0] | 63.5% | 61.5% | 100.0% | 92.0% |
| r64 all | 47.0% [40.0%, 54.0%] | +33.5pp [+24.5, +42.0] | 57.0% | 56.5% | 100.0% | 100.0% |
| r16 q/v | 17.5% [12.5%, 23.0%] | +4.0pp [-3.5, +11.0] | 33.0% | 33.0% | 65.0% | 81.0% |

r16 与 r4 的配对成功率差值为 -0.5pp，95% CI 为 [-9.0, +8.0]，screen 样本不足以判定二者优劣。

分层成功率：

| 配置 | IID（100） | Compositional（50） | Challenge（50） |
|---|---:|---:|---:|
| r4 all | 46.0% | 46.0% | 60.0% |
| r16 all | 44.0% | 68.0% | 40.0% |
| r64 all | 39.0% | 72.0% | 38.0% |
| r16 q/v | 12.0% | 6.0% | 40.0% |

## 5. 失败结构与筛选决策

r4 与 r16 的总分接近但擅长场景不同：

- r4 在 create、timeout、not_delivered 和 Challenge 上更强。
- r16 在 identity、duplicate、status、Compositional 和事实忠实上更强。
- r64 在 create、duplicate 上强，但 anti-hallucination、status 和 policy 明显退化；在更大 adapter 成本下总体也未提升。
- q/v-only 出现 parse error、错误工具/参数和无依据状态，说明只训练 attention 的 q/v 投影不足以学习当前工具协议。

选择：

1. r4 all 与 r16 all 进入 600 条 gate。
2. r64 all 和 r16 q/v 在本轮停止，不进入 gate。
3. gate 仍使用冻结的 IID / Compositional / Challenge = 300 / 150 / 150，不能根据 screen 失败案例修改。
4. gate 结束后根据任务成功率、关键安全指标、分层退化和 adapter 成本选择一个 SFT 初始化点，再进行 DPO 组合与 beta 筛选。

## 6. 600 条 SFT development gate 与 SFT 选择

| 配置 | 任务成功率（95% CI） | 工具选择 | 参数正确 | 事实忠实 | 禁止工具不触发 | 状态断言 |
|---|---:|---:|---:|---:|---:|---:|
| r4 all | 48.83% [44.83%, 52.83%] | 61.0% | 60.0% | 93.33% | 81.0% | 97.17% |
| r16 all | 48.50% [44.50%, 52.50%] | 61.67% | 58.17% | 99.67% | 75.0% | 91.0% |

r16-r4 的配对成功率差值为 -0.33pp，95% CI 为 [-5.33, +4.50]，两者总体成功率统计等价。分层成功率如下：

| 配置 | IID（300） | Compositional（150） | Challenge（150） |
|---|---:|---:|---:|
| r4 all | 43.67% | 52.67% | 55.33% |
| r16 all | 42.33% | 67.33% | 42.00% |

最终选择 r4 all 作为 DPO 初始化点。它的 adapter 只有 r16 的四分之一，在总体成功率等价的前提下，Challenge、禁止工具不触发和状态断言更好。r16 的事实忠实优势必须保留为重要证据：r4 后续 DPO 若不能维持或提高 93.33% 的事实忠实，或出现政策/禁止工具回退，就不能宣称偏好对齐有效。

该 600 条集合已经用于选择 SFT parent，因此只能作为 development gate。它不得再用于修改规则后的最终系统结论；7B 正式 Initial/SFT/SFT+DPO 对比需要另建并一次性打开 formal test v2。

## 7. DPO 5-step 冒烟与预算校准

从 r4 和 r16 各自 adapter 继续训练同一个 response-only matched 数据，固定 beta 0.1、seed 42、BF16、有效 batch 16、5 steps。该实验只验证链路和资源，不用于模型选择。

| SFT 初始化 | train runtime | train steps/s | eval runtime（179 对） | eval reward accuracy | eval reward margin | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| r4 all | 41.25 s | 0.121 | 20.73 s | 1.000 | 0.5956 | 42,451 MiB |
| r16 all | 40.78 s | 0.123 | 20.40 s | 1.000 | 2.2041 | 42,779 MiB |

冒烟确认 DPO 会复用并继续训练 SFT adapter，没有创建第二套 LoRA。1.5B 单卡显存充足，不需要 DeepSpeed。验证 preference accuracy 很快达到 1.0，说明该指标不足以证明业务增益，完整 DPO 仍必须进行相同 rollout 评测。

审计更正：上述历史 DPO run 的命令虽然包含 weight decay 0.01，但旧版 `dpo_training.py` 未把它传入 `DPOConfig`；保存的 `training_args.bin` 证明实际值为 0.0。因此这些 run 只能作为“当前偏好数据/训练强度会导致动作抑制”的负面诊断，不能当作 weight decay 0.01 配方的结果。修复版要求在训练前保存 policy/reference log-prob 等价前检，并需要用新的不可变 run ID 做有界复验。

修复后的完整复验已经完成：response-only、等量多粒度、全量多粒度相对 SFT 分别回退 11.5、21.0、24.5 个百分点，且 checkpoint-50 仍显著回退。v1.3.2 DPO 不进入 beta 或 7B；完整结果与失败闭环见 `reports/ecommerce_1p5b_dpo_screen_results.md`。

## 8. 远端结果位置

- 训练聚合：`experiments/local/1p5b/sft_matrix_summary_v1.json`
- SFT screen 聚合：`experiments/local/1p5b/sft_screen_comparison_v1.json`
- Initial + SFT 完整聚合：`experiments/local/1p5b/screen_comparison_v1.json`
- SFT gate 聚合：`experiments/local/1p5b/sft_gate_comparison_v1.json`
- DPO 5-step 聚合：`experiments/local/1p5b/dpo_bench5_summary_v1.json`
- 单次逐样本结果：各 run 目录下的 `screen_eval/per_sample.jsonl`
- 单次 trace：各 run 目录下的 `screen_traces.jsonl`
- 600 条 gate：各入选 run 目录下的 `gate_traces.jsonl` 与 `gate_eval/`

这些实验产物位于 AutoDL 数据盘并被 Git 忽略；Git 仅跟踪可复现代码、配置和本报告。
