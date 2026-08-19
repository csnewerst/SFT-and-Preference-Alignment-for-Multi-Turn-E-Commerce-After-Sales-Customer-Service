# Qwen2.5-0.5B 电商售后 SFT+DPO 小规模闭环结果

> **后续评测更新：** 本报告的 `0/6 → 5/6 → 5/6` 只保留为首动作格式冒烟结果。接入真实工具执行和多轮 observation 后，9 条开发诊断集结果为 Initial `0/9`、SFT `2/9`、SFT+DPO `2/9`。能力判断以 `reports/ecommerce_rollout_dev_v1_results.md` 为准。
>
> 代码分支：`codex-sync`
>
> 代码基线：`81a66d9`
>
> 运行环境：AutoDL，单张 NVIDIA A800-SXM4-80GB，Conda base，PyTorch 2.8.0+cu128

##1. 结论

0.5B 的 SFT、从 SFT adapter 继续 DPO、保存、重新加载、三阶段固定提示推理和自动评分已经完整跑通。扩大版固定集的 `first_action_exact` 从 Initial 的 `0/6` 提升到 SFT 的 `5/6`，SFT+DPO 仍为 `5/6`。

这个结果只证明首动作格式与工具路由已经在小规模训练中学到，不能等价为完整售后任务成功率。唯一失败样本是用户未提供订单号时，SFT 和 SFT+DPO 都虚构了 `EC-1001` 并调用查询工具；模型尚未学会先追问必要参数。DPO 在偏好验证集上的排序指标明显改善，但未在这 6 条首动作测试上进一步超过 SFT。

##2. 数据与质量门禁

- 数据版本：`domain_pilot_v1_1_1`
- SFT：2,000 条；按父样本分组切分为 train/validation/test = 1,565/209/226
- DPO：800 对；按父样本分组切分为 train/validation/test = 618/87/95
- SFT 近重复移除：454；DPO 近重复移除：121
- SFT 数据哈希：`3d75780efefdb8c2c5f8255aa08e69839caa8d0e21951cb92f0d86130f68d636`
- DPO 数据哈希：`22448bae7bb96e6ba89698caf5478c0b496a9c496b67a56fb6aa2b04b9fae8ff`
- 第二轮人工复审：30/30 已审，28 通过、2 待修订，通过率 93.3%；业务正确 30/30，工具正确 30/30，DPO 偏好清晰 12/12，严重问题 0
- 两条待修订均为 `missing_order_id` 回复模板重复；生成器 v1.1.1 已改为组合式缺参回复并重新生成、审计

CSDS、DCH-2 已完成外部权利核验且代码支持本地适配，但本次运行目录中没有这两个来源的原始文件。因此当前仍是可训练的领域 pilot，不冒充已经接入真实公开语料的正式数据集。

##3. 训练配置与指标

###3.1 SFT

- 基座：`Qwen2.5-0.5B-Instruct`
- 方法：BF16 LoRA，单卡
- 训练/验证样本：800/64
- 最大步数：100；实际 epoch：1.0
- 训练 loss：1.3686
- 验证 loss：0.9382；perplexity：2.5554
- 训练时长：110.06 秒；吞吐：7.268 samples/s
- adapter SHA-256：`9fc4631d848f87e46e0ab27720fee58affe5bad78d6895a1d3c8f848ddbeedac`

###3.2 DPO

- 初始化：加载上述 SFT adapter，并保持 adapter 可训练
- 方法：BF16 LoRA，`beta=0.1`，单卡
- 训练/验证偏好对：400/64
- 最大步数：50；实际 epoch：0.5
- 训练 loss：0.5006
- 验证 loss：0.4298
- 验证 reward accuracy：1.0000
- 验证 reward margin：0.6376
- chosen/rejected reward：0.0269 / -0.6107
- 训练时长：77.04 秒；吞吐：2.596 samples/s
- adapter SHA-256：`ef503294c2b8bb028908f34c302a9ee47cf5bf173467370a8285b638adc1f444`

##4. 固定提示对比

评测包含 6 个预先固定场景：缺订单号、破损换货、错发退货、状态查询、反幻觉和重复申请保护。每个阶段各生成 6 条，共 18 条逐样本记录。

| 阶段 | first_action_exact | 观察 |
| --- | ---: | --- |
| Initial | 0/6（0%） | 输出格式不统一，存在直接创建售后和猜测状态 |
| SFT | 5/6（83.3%） | 有订单号时均先调用 `query_order_status` |
| SFT+DPO | 5/6（83.3%） | 与 SFT 相同，未改善缺参追问 |

失败案例：`耳机到货就是坏的，帮我换一副。` 未提供订单号，期望模型先询问订单号；SFT 与 SFT+DPO 均输出 `Action: query_order_status`，并虚构 `EC-1001`。

对比产物哈希：

- `comparison.jsonl`：`6d1c179ba61e98ee45f72d8a3af39c1face26f3fa00869eff7ceb9e6c81b86ce`
- `comparison_summary.json`：`92838f96fac5b7a6eeda72477cdad332b7c56317d841ca8b47817874322d281a`

此前极小配置（SFT 20 步、DPO 10 步）三阶段均为 0/6，说明仅验证脚本可运行不足以形成可观察行为；扩大到 SFT 100 步后才出现稳定的首步工具格式。

##5. 工程验证

- 修复 DPO 入口原先声明但未使用 `peft_path` 的问题，使 DPO 真正从 SFT adapter 继续训练
- 将 `beta` 和随机种子显式纳入 DPO 配置并记录
- 加入可复现的 0.5B SFT/DPO 脚本、固定提示、三阶段推理和首动作评分脚本
- AutoDL 工作区测试：48 passed，1 条来自 TRL experimental API 的已知 warning
- 模型、训练数据、checkpoint、输出和日志只保存在 AutoDL 项目目录，不进入 Git

##6. 结果边界与下一步

当前完成的是 Phase 3 的 0.5B 主链路闭环，不是正式效果实验。下一步应：

1. 接入已核权的 CSDS、DCH-2 原始文件，构造带真实表达分布的 v2 数据，并保留当前 v1.1.1 作为冻结 pilot 基线。
2. 专门增强缺参追问、工具 observation 后续回复、工具参数 schema 和多步任务数据；扩大固定测试集，报告完整 task success、tool-call、policy、faithfulness 等指标。
3. 用 1.5B/3B 做数据配比和 LoRA/DPO 超参数筛选，再冻结正式数据与测试集。
4. 在 Qwen2.5-7B-Instruct 上运行 Initial/SFT/SFT+DPO 主实验、多随机种子与人工盲评；OPD 仍为最后的可选扩展。
