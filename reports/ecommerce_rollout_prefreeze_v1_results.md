# Rollout pre-freeze v1 构建与 0.5B 评测记录

> 日期：2026-08-09
>
> 状态：工程候选集，尚未冻结
>
> 结论边界：用于验证数据构建、执行评测和失败归因链路，不代表正式中文测试集或最终模型效果

## 1. 本轮结论

已建立从公开对话来源 test split 到可执行售后 rollout 候选集的确定性构建链路，并生成 80 条三层候选案例。每条案例都具备来源父样本、表达骨架、模拟器业务状态、私有 oracle、可接受工具轨迹和人工审核入口；当前评测器对 oracle replay 的通过率为 100%。

本轮 0.5B 结果为：Initial 5/80，SFT 28/80，SFT+DPO 20/80。SFT 明显优于基座，但 DPO 相对 SFT 下降 10 个百分点。该负结果与之前 9 条开发集诊断一致：当前 DPO 验证集排序指标不能转化为稳定的多步工具任务成功率，因此不能直接扩大同一版偏好数据。

80 条候选目前只使用 Bitext 和 glaive 的保留 test 父样本提取表达结构。它已经是合格的工程 pre-freeze 集，但不满足“真实中文来源完成”的冻结条件。CSDS 与 DCH-2 文件接入后必须重新生成并完成人工审核，不能把当前候选集直接升格为正式评测集。

## 2. 构建协议与产物

构建脚本：`scripts/ecommerce/build_rollout_prefreeze_v1.py`。

输入限制：

- 只读取各公开来源的 `normalized/test/records.jsonl`；
- 在抽取前检查 parent/group ID 与 train、validation 无交集；
- 公开对话只提供轮次、长度、疑问、否定、强调等表达骨架和来源证据；
- 订单事实、工具 observation、目标状态和可接受轨迹全部由版本化模拟器重建；
- 不把公开数据原回答或真实/生成数据文件提交到 Git。

远端忽略目录中的产物：

| 文件 | 用途 |
|---|---|
| `cases.jsonl` | 不含 oracle 的模型输入案例 |
| `private_oracle.jsonl` | 隔离保存环境初态、目标和可接受轨迹 |
| `evaluator_cases.jsonl` | 当前规则评测器的可执行输入 |
| `source_manifest.jsonl` | 案例到公开来源父样本的追溯关系 |
| `human_review/review_queue.jsonl` | 冻结前人工审核队列 |
| `audit/report.json` | 泄漏、PII、重复、工具名泄漏和 oracle replay 门禁 |
| `manifest.json` | 数量、分布和内容哈希 |

自动门禁全部通过：parent leakage、PII、工具名泄漏、case/text/source-parent 重复均为 0；80/80 oracle replay 通过。

## 3. 候选集组成

| 维度 | 分布 |
|---|---|
| 总量 | 80 |
| Tier | IID 40；Compositional 20；Challenge 20 |
| 来源 | Bitext 31；glaive 49 |
| 多轮来源骨架 | 13 |
| 场景 | status 10；policy 10；create 15；missing_order 15；其余 6 类各 5 |

覆盖的 10 类场景为：订单状态、政策查询、创建售后、缺订单号、重复申请、身份未核验、未签收、工具超时、时效过期和诱导幻觉。

关键内容哈希：

- `cases.jsonl`：`833af855d87b62f7c21a7e287035ab53f07192a2d336431be83e8531debbf486`
- `private_oracle.jsonl`：`d43b4a4adb114aebddeb039c1a48966dc4d2716e6dffad40c83dd47fc89fbcf7`
- `source_manifest.jsonl`：`beedaa0d7d4ddb91909cc995676672223e44b66bfed123c83fd1fdb7f38f6d10`
- `audit/report.json`：`933edc899f17dac78ae65c76a9269c9e1dc1364460ea2ddb0ed41bb5ae98a14b`

## 4. 0.5B 三阶段结果

共同条件：Qwen2.5-0.5B-Instruct、同一套 80 条 evaluator cases、同一模拟器与规则评测器；SFT 与 DPO 使用 `05b_dpo_v1_2_1` 实验产物。

| 阶段 | 总成功率 | IID | Compositional | Challenge |
|---|---:|---:|---:|---:|
| Initial | 5/80（6.25%） | 2/40（5%） | 0/20（0%） | 3/20（15%） |
| SFT | 28/80（35%） | 15/40（37.5%） | 3/20（15%） | 10/20（50%） |
| SFT+DPO | 20/80（25%） | 10/40（25%） | 0/20（0%） | 9/20（45%） |

SFT 相对 Initial 提升 28.75 个百分点，证明 SFT 已学到部分工具选择和售后规则；但组合泛化只有 15%，创建、过期和身份核验均为 0，离正式可用很远。

DPO 相对 SFT：

- task success：35% → 25%；
- tool selection valid：60% → 40%；
- arguments valid：46.25% → 38.75%；
- facts faithful：91.25% → 87.5%；
- forbidden tool absent：75% → 100%。

DPO 更保守、越权更少，但完成任务的能力下降。逐样本对齐后有 16 条回退、8 条新增通过；16 条回退中，13 条同时出现 wrong tool、wrong argument 和 unexpected tool outcome。回退主要集中于 status 7 条、policy 3 条、not_delivered 2 条和 timeout 2 条。DPO 新增收益则主要集中在 missing_order 4 条，说明它偏向学习“缺信息时停止或追问”，没有稳定掌握继续执行的多步决策。

## 5. 为什么 80 条仍有意义

这批数据的价值不是提前制造一个漂亮分数，而是低成本暴露正式扩容前的系统问题：

1. 验证公开表达骨架与私有业务 oracle 可以安全解耦；
2. 验证来源切分、父样本防泄漏、私有 oracle 隔离和逐样本追溯能够工作；
3. 证明 9 条开发集观察到的 DPO 退化不是单个样本偶然现象；
4. 用 80 条即可发现组合泛化、多步创建和参数传递是当前真正瓶颈；
5. 在冻结 600～1,000 条和启动 7B 主实验前修正构建与训练方案，避免把昂贵算力用在错误方向。

因此，小规模人工审核和实验属于“测量工具校准 + 数据协议验收”，不是正式数据建设的替代品。

## 6. 真实来源接入状态

- CSDS：已完成外部权利核验，项目已有受限本地适配器。本轮 AutoDL 尝试从[官方仓库公布的 Google Drive 入口](https://github.com/xiaolinAndy/CSDS)获取文件，但服务器到 `drive.google.com:443` 网络不可达；未改用来源不明的镜像。
- DCH-2：已完成外部权利核验，项目已有受限本地适配器；[官方数据页](https://dialeval-2.github.io/DCH-2/)要求完成用户协议流程，当前本地和 AutoDL 均未发现已授权数据文件，因此没有绕过访问控制。

当前候选集保留 `development_candidate_not_frozen` 状态。只有在 CSDS/DCH-2 授权文件实际落盘、来源级重切分、重建候选集并完成盲审后，才能冻结 v1。

## 7. 下一步决策

按风险和收益排序：

1. 接入已授权的 CSDS/DCH-2 文件，先规范化并验证 train/validation/test 父样本无交集；重新生成 60～100 条中文 pre-freeze v1.1。
2. 对 v1.1 全量盲审，重点检查自然度、场景映射、oracle 唯一性、可接受多轨迹和 challenge 难度；不通过项修构建规则，不手工润色测试答案。
3. 重构 DPO：增加“必须继续调用”的多步 decision pair、完整 trajectory preference 和参数跨轮继承样本；按场景与 continue/stop 决策配平。
4. 在固定 pre-freeze 集上只做 1.5B/3B 的 SFT、response-only DPO、三层 DPO 小规模消融。只有 DPO 在总体、Compositional 和关键场景均稳定优于 SFT，才扩到正式数据规模。
5. 冻结 600～1,000 条真实中文 rollout，最后运行 Qwen2.5-7B Initial/SFT/SFT+DPO 主实验、人工盲评和置信区间分析。

## 8. 可复现证据

- 代码提交：`45cc85e`（构建器）、`942d0f4`（oracle replay 门禁）、`93933b9`（外部 case 评测支持）。
- AutoDL 完整测试：60 passed；另有 1 个 OPD 实验 API 非阻断 warning。
- 本地下载的逐样本评测结果位于被 Git 忽略的 `experiments/local/prefreeze_v1_eval/`，用于本报告分层和回退分析。

## 9. CSDS/DCH-2 全量接入与推荐规模更新

用户将已授权文件上传 AutoDL 后，已完整扫描两套中文来源，而非继续使用每个来源 1,000 条的 pilot 上限：

| 来源 | 扫描 | 接受 | 拒绝 | 独立 test 父样本 |
|---|---:|---:|---:|---:|
| CSDS | 10,701 | 10,663 | PII 38 | 1,066 |
| DCH-2 | 4,390 | 4,345 | PII 22；不完整对话 23 | 434 |

首次直接扩到 800 条时，自动门禁发现模板组合容量不足导致精确文本重复，因而没有生成产物。构建器随后扩展了自然后续诉求组合并新增 800 条规模回归测试；没有使用追加编号等伪唯一化方式。又增加来源白名单，避免正式中文候选混入 Bitext/glaive 父样本。

最终 `rollout_prefreeze_v1_1_zh`：

- 800 条：IID 400、Compositional 200、Challenge 200；
- 来源：CSDS 573、DCH-2 227；
- 多轮来源骨架：320；
- parent leakage、PII、工具名泄漏、case/text/source-parent 重复均为 0；
- oracle replay：800/800；
- `cases.jsonl` SHA-256：`a4604b0a2b13686c13759414e3229cb7f28830ec182b37d5d3fa2eace57f3eef`。

这批数据达到正式评测集建议规模，但状态仍为 candidate：自动正确不等于自然表达、场景映射和多轨迹 oracle 已经人工确认。冻结前需做分层盲审。训练侧则应独立使用 train/validation 父样本构造 10,000～30,000 条 SFT 和 5,000～10,000 对偏好候选，不能用 test candidate 训练。
