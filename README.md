# E-commerce After-sales SFT & DPO

面向电商售后场景的多轮客服模型后训练项目。基于 Qwen2.5-7B-Instruct，构建从领域数据重构、LoRA SFT、失败轨迹驱动的多粒度 DPO，到多轮 Function Calling 与确定性业务评测的完整离线闭环。

> 本仓库公开代码、配置、项目生成的数据集、最终LoRA checkpoint、训练日志、逐样本评测结果和实验报告。Qwen基座模型需按下文路径单独下载。

##项目亮点

- **可执行数据构建**：将 CSDS / DCH-2 的多轮客服对话重构为10类售后任务，由确定性业务模拟器生成订单事实、标准工具轨迹和 Observation；完成 PII 检测、SimHash 去重、父样本切分和跨 split 泄漏审计。
- **失败轨迹驱动的 DPO**：通过rollout诊断定位基于规则合成负例造成的动作偏置；从冻结SFT的离线rollout中定位与Oracle标准轨迹的首次行为分歧，以同状态下的标准行为和模型实际错误构造hard negative。
- **多粒度偏好建模**：同时覆盖 Decision、Parameter 和 Response 三类偏好，保留“是否调用工具、参数是否正确、回复是否忠于执行结果”三个层面的训练信号。
- **可执行评测**：实现订单查询、政策校验和售后建单三个工具，逐 case 检查工具序列、参数、Observation、状态转移、事实忠实、禁用工具和最终回复，并使用10,000次 paired bootstrap 与失败迁移分析比较模型。

##方法总览

```text
CSDS / DCH-2
    │  结构抽取、业务重构、PII/去重/父样本切分
    ▼
12K SFT trajectories ──LoRA SFT──► frozen SFT policy
                                      │ offline rollout
                                      ▼
                              first divergence mining
                                      │
                     Decision / Parameter / Response pairs
                                      │ 难度筛选与行为配平
                                      ▼
                              720 / 80 DPO pairs
                                      │ DPO
                                      ▼
                            SFT + DPO policy
                                      │
                                      ▼
                600-case frozen executable evaluation
```

###确定性业务模拟器

模拟器由版本化的工具 schema、售后政策和订单场景驱动，提供：

1. `query_order_status`：查询履约与订单事实；
2. `check_return_policy`：根据签收时长、问题类型和证据要求判断可受理范围；
3. `create_after_sales_request`：校验身份、政策资格、申请类型和重复建单后更新环境状态。

同一配置、初始状态和调用序列始终产生相同Observation与状态快照，不依赖外部API、随机数或系统时间，可用于稳定复现训练数据和业务评测结果。

###Rollout hard-negative DPO

对冻结 SFT 运行多轮工具轨迹，将模型轨迹逐步与业务 Oracle 对齐，在第一个行为分歧处构造偏好对：

- `chosen`：当前状态下的标准工具行为或合规终止回复；
- `rejected`：冻结 SFT 在同一状态下实际生成的错误行为；
- 难度：`mean_logp(chosen) - mean_logp(rejected)`，越小表示 SFT 越难区分；
- 筛选：结合继续执行/终止回复状态、偏好粒度、难度和父样本贡献上限，形成720/80对训练/验证集。

##部分实验与结果

###主实验

统一使用同一组600条冻结测试 case，包含 IID、Compositional 和 Challenge 三个层级；正式测试在训练配置和 checkpoint 冻结后一次性打开。

| 模型 | Task success | Eligible auto-resolution | 工具/参数正确 | 事实忠实 | 禁用工具未调用 |
|---|---:|---:|---:|---:|---:|
| Initial Qwen2.5-7B-Instruct | 36.67% | 34.35% | 48.67% / 48.67% | 89.33% | 83.00% |
| LoRA SFT | 54.50% | 56.49% | 71.83% / 71.83% | 95.33% | 79.17% |
| LoRA SFT + DPO | **57.50%** | **60.11%** | **74.67% / 74.67%** | **95.67%** | **82.17%** |

10,000次 paired bootstrap：

- SFT - Initial：`+17.83pp`，95% CI `[+13.00, +22.67]`；
- SFT+DPO - SFT：`+3.00pp`，95% CI `[+0.67, +5.33]`；
- SFT+DPO - Initial：`+20.83pp`，95% CI `[+16.16, +25.50]`。

固定配置在第二个训练seed上复现出同方向的DPO增益（`+2.67pp`，95% CI `[+0.33, +5.00]`）。表中的auto-resolution为模拟器内eligible case子集的自动解决率。

###消融与诊断

项目完成 LoRA 配置、偏好粒度、数据规模和 hard-negative 配方四组核心消融。代表性结果：

- 在Qwen2.5-1.5B、同一SFT起点、同源等量173对、相同训练参数的实验中，多粒度DPO在step 10较Response-only提升`14.5pp`，95% CI `[+8.0, +21.0]`。
- 规则合成负例曾达到100% validation preference accuracy，但多轮 rollout 发生动作抑制；改为失败轨迹 hard negative 后，1.5B开发 screen 的 task success 从49.5%提升至58.0%。

完整实验协议与结果见 [`reports/`](reports/)。7B主结果见 [`reports/ecommerce_7b_results_public.md`](reports/ecommerce_7b_results_public.md)，多粒度等量消融见 [`reports/ecommerce_1p5b_multigranularity_ablation_public.md`](reports/ecommerce_1p5b_multigranularity_ablation_public.md)。

##仓库结构

```text
configs/ecommerce/   # 工具、政策、场景、数据配方与实验配置
scripts/ecommerce/   # 数据准备、审计、rollout、DPO挖掘、训练与评测脚本
training/            # 本项目使用的LoRA SFT与DPO训练入口
tests/               # 模拟器、数据、评测和实验配置的自包含测试
reports/             # 数据规范、实验协议与公开结果摘要
docs/                # 项目路线与设计记录
data/README.md       # 外部数据获取、目录约定与许可说明
artifacts/           # 最终LoRA checkpoint、训练日志、人工复核记录与正式评测产物
```

##可复现产物

仓库直接提供主实验与核心消融所需的生成数据：

| 路径 | 内容 | 规模 |
|---|---|---:|
| `data/ecommerce/domain_train_v1_3_2_zh/` | SFT轨迹与三粒度偏好候选 | 12K SFT / 5K DPO候选 |
| `data/ecommerce/dpo_v1_4_rollout_quality_screen_800_v2/` | rollout hard-negative主训练集 | 720 train / 80 validation |
| `data/ecommerce/dpo_v1_4_rollout_quality_composition_matched_v1/` | 多粒度与Response-only等量消融 | 173 train / 19 validation |
| `data/ecommerce/formal_test_v2/` | IID、Compositional、Challenge冻结测试集 | 600 cases |

最终LoRA适配器通过Git LFS发布：

| 路径 | SHA256 |
|---|---|
| `artifacts/checkpoints/7b-seed42-sft-step200/` | `bc00bbcaf6381a4a0af7f76a90f4870e0f30639be19b8242b5eec9f4afabdd39` |
| `artifacts/checkpoints/7b-seed42-dpo-step5/` | `e1e5987ca786c8bf2115ad28061e03102f8ab679e81a9bbf02c4f92393f4800e` |
| `artifacts/checkpoints/7b-seed43-sft-step200/` | `3cd61db086ad54d16a21a560ccd1e0b996c663e265b9fbfaad37f705e907fad6` |
| `artifacts/checkpoints/7b-seed43-dpo-step5/` | `5976d76daec64dd113165c04ed737857e59be217365573876a544a36f1383de7` |

`artifacts/runs/`保存两个训练seed的命令、环境、manifest、训练日志、显存采样和开发集轨迹；`artifacts/evaluations/`保存600条正式测试上的逐样本轨迹、确定性评测结果、paired bootstrap和失败迁移；`artifacts/human_review/`保存训练候选的人工复核队列、审核填写结果与准入结论。`artifacts/checkpoints/`中的适配器可直接用于复现SFT与SFT+DPO评测。

##环境

原始实验环境：Linux、Python 3.12.3、CUDA 12.8、PyTorch 2.8.0+cu128、单张 NVIDIA A800-SXM4-80GB。7B BF16 LoRA 前检峰值显存约38.6GiB；正式训练脚本均为单卡训练，不依赖多卡分布式。

推荐先单独安装与CUDA匹配的PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`requirements.txt`是可移植的核心依赖；`requirements-lock.txt`保留原实验环境的完整快照。

##数据与模型准备

1. 下载 `Qwen/Qwen2.5-7B-Instruct` 到 `models/base/Qwen2.5-7B-Instruct/`，或修改运行脚本中的 `MODEL_PATH`。
2. 仓库已提供生成后的主实验数据，可直接进入训练；数据目录和hash见 [`data/README.md`](data/README.md)。
3. 如需从源数据完整重建，可运行数据适配和领域任务构建：

```bash
python scripts/ecommerce/prepare_public_pilot.py \
  --source csds-emnlp21 --input /path/to/csds \
  --rights-acknowledged

python scripts/ecommerce/prepare_public_pilot.py \
  --source dch2-dialeval2 --input /path/to/dch2 \
  --rights-acknowledged

python scripts/ecommerce/build_domain_pilot_v1.py \
  --public-pilot-root data/ecommerce/public_pilot \
  --output-root data/ecommerce/domain_train_v1_3_2_zh \
  --sft-limit 12000 --dpo-limit 5000 --variants-per-parent 2
```

最终SFT/DPO适配器和正式评测产物位于`artifacts/`。克隆仓库后执行：

```bash
git lfs install
git lfs pull
```

##训练与评测

###1. 运行测试

```bash
pytest -q
```

测试使用临时目录内的合成fixture，不需要下载真实数据或模型。

###2. 7B SFT

```bash
PYTHON_BIN="$(command -v python)" LORA_RANK=8 TRAIN_SEED=42 \
  bash scripts/ecommerce/run_7b_sft_main.sh
```

原实验使用BF16、all-linear LoRA、rank 8、alpha 16、学习率`1e-5`、有效batch 32、1 epoch，并通过200条开发screen选择冻结的SFT起点。详细参数以 [`configs/ecommerce/experiments_7b_v1.json`](configs/ecommerce/experiments_7b_v1.json) 为准。

###3. 冻结SFT rollout、首次分歧挖掘与DPO数据筛选

核心入口：

- `run_ecommerce_rollout.py`：执行模型—工具多轮交互；
- `evaluate_rollout_v1.py`：计算逐case确定性指标；
- `mine_dpo_from_sft_rollouts.py`：定位首次行为分歧并构造偏好对；
- `score_dpo_pair_hardness.py`：使用冻结SFT计算chosen/rejected平均log-prob margin；
- `build_dpo_v1_4_quality.py`：按行为、粒度、难度和父样本配额冻结DPO集合。

###4. 7B DPO

```bash
SFT_ADAPTER=experiments/local/7b/<sft-run>/adapter/checkpoint-200 \
PYTHON_BIN="$(command -v python)" TRAIN_SEED=42 \
  bash scripts/ecommerce/run_7b_dpo_calibration.sh
```

若直接复现已发布结果，可将`SFT_ADAPTER`设为`artifacts/checkpoints/7b-seed42-sft-step200`，并使用`artifacts/checkpoints/7b-seed42-dpo-step5`执行最终DPO模型评测。

原实验使用720/80偏好对、`beta=0.1`、学习率`2e-6`、有效batch 16、20 steps，并评测step 5/10/20。正式结果使用训练前冻结的配置与测试集，不能依据正式测试结果重新调参。

###5. 可执行评测

```bash
python scripts/ecommerce/run_ecommerce_rollout.py \
  --cases /path/to/frozen_cases.jsonl \
  --base-model models/base/Qwen2.5-7B-Instruct \
  --adapter /path/to/adapter \
  --output experiments/local/eval/traces.jsonl \
  --device cuda:0 --max-steps 6

python scripts/ecommerce/evaluate_rollout_v1.py \
  --cases /path/to/frozen_cases.jsonl \
  --traces experiments/local/eval/traces.jsonl \
  --output-dir experiments/local/eval/metrics
```

Task success要求解析、工具序列、参数、Observation结果、禁用工具、状态断言、事实忠实、回复要求和步数约束全部通过。结构化指标由确定性规则计算，不使用LLM-as-Judge。

##License

代码按 [`LICENSE`](LICENSE) 中的 Apache License 2.0 发布。
