# Qwen2.5-7B 电商售后主实验结果

##实验设置

- 模型：Qwen2.5-7B-Instruct
- 训练：BF16 LoRA SFT → DPO
- SFT数据：10,800 train / 1,200 validation
- DPO数据：720 train / 80 validation
- 正式测试：600条冻结case，包含300 IID、150 Compositional与150 Challenge
- 解码：greedy，最多6轮工具交互
- 统计：10,000次paired bootstrap，seed `20260809`
- 硬件：单张NVIDIA A800-SXM4-80GB

SFT使用all-linear LoRA、rank 8、alpha 16、学习率`1e-5`、effective batch 32、seed/data seed 42；DPO从SFT适配器继续训练，使用`beta=0.1`、学习率`2e-6`、effective batch 16、seed 42。

##Formal test v2

| 模型 | Task success | Eligible auto-resolution | 工具/参数正确 | 事实忠实 | 禁用工具未调用 | Parse success |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 36.67% | 34.35% | 48.67% / 48.67% | 89.33% | 83.00% | 85.17% |
| SFT | 54.50% | 56.49% | 71.83% / 71.83% | 95.33% | 79.17% | 97.83% |
| SFT+DPO | **57.50%** | **60.11%** | **74.67% / 74.67%** | **95.67%** | **82.17%** | **97.83%** |

配对统计：

- SFT - Initial：`+17.83pp`，95% CI `[+13.00, +22.67]`
- SFT+DPO - SFT：`+3.00pp`，95% CI `[+0.67, +5.33]`
- SFT+DPO - Initial：`+20.83pp`，95% CI `[+16.16, +25.50]`

DPO相对SFT的工具选择与参数正确率均提升`2.83pp`，禁止工具未调用提升`3.00pp`，answer requirements提升`3.67pp`。

##第二训练seed复现

固定相同配置，仅将训练seed/data seed改为43：

| 模型 | Task success | Eligible auto-resolution | 工具/参数正确 | 事实忠实 | 禁用工具未调用 |
|---|---:|---:|---:|---:|---:|
| SFT | 63.00% | 65.46% | 76.00% / 76.00% | 97.33% | 84.67% |
| SFT+DPO | **65.67%** | **67.75%** | **80.00% / 80.00%** | **97.67%** | **86.67%** |

DPO - SFT的task success为`+2.67pp`，95% CI `[+0.33, +5.00]`，与seed 42保持同方向收益。

##复现证据

- 数据：`data/ecommerce/domain_train_v1_3_2_zh/`、`data/ecommerce/dpo_v1_4_rollout_quality_screen_800_v2/`
- 冻结测试：`data/ecommerce/formal_test_v2/`
- 最终适配器：`artifacts/checkpoints/`
- 训练日志：`artifacts/runs/`
- 逐样本轨迹、summary与paired comparison：`artifacts/evaluations/`
