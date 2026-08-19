# Qwen2.5-1.5B 筛选实验 preflight 结果
> 状态：模型、数据派生、token长度和代码门禁通过；训练效果结果尚未产生。

##1. 代码与环境

- 分支：`codex-sync`；
- 远端工作区：`/root/autodl-tmp/Resume/customer-service-posttraining`；
- 相关测试：本地新增测试通过，远端1.5B准备与pre-freeze测试12/12通过；
- 训练策略：1.5B主实验单卡BF16 LoRA，不使用DeepSpeed，多卡用于并行独立配置。

##2. 模型

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`；
- 来源：ModelScope，revision `master`；
- 本地目录：`models/base/Qwen2.5-1.5B-Instruct`；
- 目录大小：2.9G；
- `model.safetensors` SHA256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`；
- `tokenizer_config.json` SHA256：`5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`。

模型和数据保留在AutoDL数据盘的Git忽略目录，没有上传Git。

##3. 训练数据确认

- SFT：12,000，总切分10,800 train / 1,200 validation；内容集哈希`58578c61decb75e8c13c876655886a5953e2e4829c74f923c86e7cc355a8a5c6`；
- DPO：5,000，总切分4,436 train / 564 validation；内容集哈希`69186aefa94f12f1ac5d3f5c276307f373828fabbaf1975ee31426991faef96d`；
- 800条中文pre-freeze：IID 400、Compositional 200、Challenge 200，audit通过。

##4. pre-freeze开发拆分

固定seed `20260809`，先保持tier比例，再在tier内保持category比例：

| split | 总数 | IID | Compositional | Challenge | cases SHA256 |
|---|---:|---:|---:|---:|---|
| screen | 200 | 100 | 50 | 50 | `66c06ab29ffcabd6dbad53e7a3fa306549338de0eaab9cb3644efb05d7d30439` |
| gate | 600 | 300 | 150 | 150 | `4927ebfe01b35e0b4c6f01feee07708d1bfc275ad042874c17f115aa7fca01cd` |

两者都属于开发集，不是最终test。拆分脚本拒绝重复parent跨集合。

##5. DPO等量消融数据

| variant | train | validation | train层级分布 |
|---|---:|---:|---|
| response-only matched | 1,571 | 179 | response 1,571 |
| multigranularity matched | 1,571 | 179 | decision 622 / parameter 393 / response 556 |
| multigranularity full | 4,436 | 564 | decision 1,756 / parameter 1,109 / response 1,571 |

该设计分别隔离“偏好层级结构”和“完整数据规模”的影响，避免把1,750对与5,000对直接比较后误判。

##6. token长度实测

使用Qwen2.5-1.5B tokenizer和训练工具格式审计全部12,000条SFT与5,000对DPO：

| 序列 | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| SFT完整对话 | 658 | 827 | 834 | 843 | 851 |
| DPO prompt | 468 | 652 | 771 | 784 | 796 |
| DPO chosen | 32 | 45 | 50 | 52 | 53 |
| DPO rejected | 35 | 48 | 50 | 52 | 53 |
| prompt + chosen | 500 | 689 | 821 | 835 | 847 |
| prompt + rejected | 510 | 692 | 815 | 829 | 843 |

因此固定：

- SFT `model_max_length=1024`；
- DPO `max_source_length=1024`；
- DPO `max_target_length=128`；
- 当前数据在上述限制下实测超限率为0。

审计过程中修复了两项会污染实验的实现问题：tokenizer返回mapping时错误把字段数当token数；DPO入口原先用字符数过滤token上限且训练shuffle没有显式seed。修复后才允许进入测速。

##7. 下一门槛

单卡100步SFT-R16中心配置已通过：

- run ID：`sft-r16-all-bench100-seed42-v1`；Git commit：`35bc5a0`；
- 实际训练样本10,800，未出现异常过滤；可训练参数18,464,768，占总参数1.1820%；
- 100步训练355.3秒，0.281 step/s、9.006 samples/s；峰值显存33,345 MiB，GPU利用率峰值100%；
- train loss 1.0736；validation 1,200条，eval loss 0.8105、PPL 2.2491；这些是流水线诊断，不是业务效果；
- adapter权重73,911,112 bytes，已从磁盘重新加载，并完成一条确定性多轮工具rollout；
- 完整1 epoch约338个optimizer steps，按benchmark速度估计训练本体约18分钟。

benchmark日志提示Transformers弃用`warmup_ratio`，后续配置已改为显式`warmup_steps=10`。GPU采样也改为只记录该run分配的卡。

鉴于完整训练成本已实测较低，四组rank/target配置直接使用完整10,800条训练数据并行运行，避免小型子集引入额外分布差异；之后只在200条screen上选出最多两个配置进入600条gate。
