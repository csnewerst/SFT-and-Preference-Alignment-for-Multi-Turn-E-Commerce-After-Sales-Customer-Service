# 1.5B 多粒度DPO等量消融结果

## 对照设置

两组数据来自同一rollout hard-negative质量集，统一为173 train / 19 validation，并固定相同SFT起点、数据split、父样本约束、训练步数、beta、学习率、batch和seed：

- Response-only：173条Response偏好
- Multigranularity：124条Decision、7条Parameter、42条Response偏好

训练使用Qwen2.5-1.5B-Instruct、BF16 LoRA、`beta=0.1`、学习率`5e-6`、effective batch 16、seed 42；评测使用同一200条开发screen和10,000次paired bootstrap。

## 实验结果

| Checkpoint | Response-only task success | Multigranularity task success | Multi - Response（95% CI） |
|---:|---:|---:|---:|
| step 5 | 48.5% | 54.0% | `+5.5pp [0.0, 11.0]` |
| step 10 | 40.0% | **54.5%** | **`+14.5pp [8.0, 21.0]`** |
| step 20 | 32.0% | 52.0% | **`+20.0pp [12.5, 27.0]`** |

在相同来源、相同数量和相同训练参数下，多粒度DPO在step 10相对Response-only提升`14.5pp`。对应数据位于`data/ecommerce/dpo_v1_4_rollout_quality_composition_matched_v1/`。
