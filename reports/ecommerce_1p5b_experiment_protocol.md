# Qwen2.5-1.5B 筛选实验协议 v1

> 状态：协议与执行脚本已实现，训练尚未开始。任何结果必须由实际运行产物回填，禁止将计划值写成实测值。

##1. 实验目的

1. 验证 SFT 相对 Initial 是否学习到条件化工具流程，而非固定工具链。
2. 在控制数据量后，比较 response-only 与多粒度 DPO 的贡献。
3. 选择 LoRA rank、target modules 和 DPO beta 的候选区间，降低 3B/7B 试错成本。
4. 建立可复现的命令、环境、数据哈希、随机种子、资源和逐样本业务指标证据链。

1.5B 是筛选模型，不承担最终业务效果结论。800 条 rollout 是 pre-freeze 开发候选，不是正式 test。

##2. 固定数据和评测切分

- SFT：10,800 train / 1,200 validation；
- DPO：4,436 train / 564 validation；
- pre-freeze rollout：按 `tier + category` 和固定 seed `20260809` 确定性拆成 200 条 screen 与 600 条 SFT development gate；
- screen 可用于 SFT/DPO 超参数选择和失败分析；现有 600 条只用于 SFT r4/r16 选择，已经打开，不属于正式 test；
- split 脚本拒绝重复 `parent_id`，防止同父样本跨集合。

正式系统比较必须另建与训练集、200 screen、600 development gate 均无父样本重叠的 formal test v2。它只能在 7B Initial/SFT/SFT+DPO 配置全部冻结后打开一次，并对相同样本报告 paired bootstrap 差值。

正式执行前必须记录模型 revision、模型文件哈希、数据目录聚合哈希、rollout artifact 哈希和 Git commit。

##3. SFT 消融

共同配置：BF16 LoRA、1 epoch、`lr=2e-5`、cosine、warmup steps 10、weight decay 0.01、有效 batch 32、gradient checkpointing、seed/data_seed 42。最大长度由 token 审计决定，目标截断率不高于 1%。

| run | rank | alpha | target modules |
|---|---:|---:|---|
| SFT-R4 | 4 | 8 | all-linear |
| SFT-R16 | 16 | 32 | all-linear |
| SFT-R64 | 64 | 128 | all-linear |
| SFT-QV | 16 | 32 | q_proj,v_proj |

保持 `alpha/rank=2`。100步benchmark实测完整单次SFT训练本体约18分钟，因此四种配置直接使用相同的完整10,800条train和1,200条validation，避免子集分布成为额外变量；四组只在screen上选择，最多两个配置进入gate。训练 loss 只作诊断；checkpoint 由rollout、验证指标和失败类型共同选择。

##4. DPO 消融

DPO 必须从同一个入围 SFT adapter 继续训练，并继承其 LoRA 结构。TRL 在训练器初始化时复制该 SFT adapter 为冻结 `ref` adapter；第一个 optimizer step 前必须通过 policy/reference log-prob 数值等价前检。共同配置：BF16、`lr=5e-6`、weight decay 0.01、有效 batch 16、seed 42。

先比较三种数据构成：

| run | 数据 | 目的 |
|---|---|---|
| DPO-R | response-only matched | 回复级基线 |
| DPO-M | multigranularity matched | 控制数量后验证偏好层级设计 |
| DPO-F | multigranularity full | 验证完整 5k 配方收益 |

matched 数据运行 99 steps，full 数据运行 278 steps，均对应固定数据的一次受控遍历；warmup 为总 steps 的 3% 向上取整，即 3/9 steps。随后只对完整多粒度数据比较 `beta=0.05/0.1/0.3`。最终 SFT 与 DPO 使用第二种子 `20260809` 复验，失败配置不重复消耗算力。

##5. 资源策略与时间

1.5B 主实验单卡运行，不使用 DeepSpeed。四张 A800 用于并行独立配置；可另做固定步数的单卡与两卡 ZeRO-2 吞吐验证，但不混入模型效果结论。

当前墙钟预估为 6～10 小时。该数字不是实测；第一组 100 步 benchmark 后必须用真实 step time 和 rollout latency 更新 ETA。

##6. 每个 run 的必备证据

- `manifest.json`：配置、输入哈希、Git commit、dirty 状态、Python/PyTorch/CUDA/GPU；
- `command.sh`、`environment.txt`、`git_status.txt`；
- `hardware.csv`：每秒 GPU 利用率、显存、功耗和温度；
- Trainer 日志、state、checkpoint、TensorBoard；
- train/eval loss、吞吐、时长、可训练参数、峰值显存；
- DPO chosen/rejected reward、margin、accuracy 和长度/过滤统计；
- 逐样本 rollout trace、失败标签及总体/分层业务指标；
- 固定推理参数：`do_sample=false`、`temperature=0`、最大新 token 512、最大工具轮次 6。

核心业务指标包括 task success、工具选择/序列、参数精确正确率、多步创建、提前停止、不必要调用、政策合规、observation 忠实和无依据事实率。模型差异使用相同样本配对比较，报告绝对百分点与 bootstrap 95% 区间。

##7. 执行门槛

进入训练前：数据与token审计通过、200/600拆分稳定、模型与数据哈希齐全、单卡100步无OOM、DPO policy初始状态与SFT adapter一致。

进入3B前：SFT明确超过Initial；多粒度DPO在总体、Compositional、多步创建或参数正确率上稳定超过SFT/response-only，且政策合规、忠实性、提前停止和不必要调用无关键回退。

##8. 已实现入口

- `configs/ecommerce/experiments_1p5b_v1.json`：机器可读实验协议；
- `scripts/ecommerce/prepare_1p5b_eval_split.py`：确定性 screen/gate 拆分；
- `scripts/ecommerce/build_1p5b_dpo_variants.py`：等量 DPO 数据构成消融；
- `scripts/ecommerce/audit_1p5b_token_lengths.py`：训练格式 token 长度、尾部分布和候选截断率；
- `scripts/ecommerce/capture_experiment_manifest.py`：环境与输入证据；
- `scripts/ecommerce/monitor_gpu.sh`：外部 GPU 采样；
- `scripts/ecommerce/validate_formal_test_v2.py`：校验 formal test v2 已密封、artifact 哈希一致；通过 `--development-dir` 与所有开发评测集检查 case/parent/source-record 重叠，并通过可重复的 `--reference-jsonl` 与 SFT/DPO 训练文件检查顶层、`source_ref`、`metadata` 中的同源标识；
- `scripts/ecommerce/run_1p5b_sft.sh`、`run_1p5b_dpo.sh`：单卡、不可变 run 目录训练入口。
- `scripts/ecommerce/run_1p5b_sft_matrix.sh`：四卡并行、每卡一个独立完整SFT消融run。
- `scripts/ecommerce/summarize_1p5b_runs.py`：从run产物聚合配置、训练/验证和GPU资源指标。
- `scripts/ecommerce/run_1p5b_sft_screen_eval.sh`：四卡并行运行固定200条screen多轮工具评测。

上述脚本只写入 Git 忽略的模型、数据、日志与 `experiments/local/` 目录，不提交真实数据、模型或密钥。
