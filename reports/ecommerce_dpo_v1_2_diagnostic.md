# DPO v1.2/v1.2.1 与正式 Rollout 集诊断记录
> 性质：0.5B 工程诊断，不是正式模型结论
> 开发评测：`ecommerce_rollout_dev_v1`，9 条，仅用于快速定位失败

##1. 本轮解决了什么问题

旧版 DPO 主要比较最终回复，无法直接监督“下一步应该调用哪个工具、参数是否正确、何时应该停止”。v1.2 将偏好对重构为三个粒度：

| 粒度 | 目标 | v1.2/v1.2.1 数量 |
|---|---|---:|
| decision | 正确继续、正确停止、避免越权或重复调用 | 320 |
| parameter | 正确 Action 参数 vs 缺失或越界参数 | 200 |
| response | 忠实回复 vs 幻觉、政策违规或不完整回复 | 280 |

每条偏好对记录 `preference_level`、`target_turn_index`、`primary_error`、chosen/rejected 目标类型和父样本来源。审计器会阻断目标类型与序列化内容不一致、跨 split 泄漏、近重复和 schema 错误。

##2. 数据版本与审计证据

###v1.2.0

- SFT：2,000 条；train/validation/test = 1,560/207/233。
- DPO：800 对；train/validation/test = 609/84/107。
- DPO 内容集哈希：`212c6d4adda0404240a566607ddac2412bfdf84e8bdc0236f2e03502c96432b9`。
- 首次生成被审计器以 18 个近重复和 5 个跨 split 近重复阻断；改为跨偏好层全局去重后通过。
- 问题：decision 级 chosen=停止与 chosen=继续没有显式配平，存在把“避免不必要调用”泛化为“过早停止”的风险。

###v1.2.1

- SFT 内容集哈希保持 `619ae75ac76121d4a661a65b45a7795589ca239097b282bf640ab4832b4af56e`，说明 SFT 内容未变化。
- DPO 内容集哈希：`f46e0266ef372bce2646603b56fd6612b8986c147faf4d2ef957824e6a1ac5fb`。
- decision chosen 类型严格配平：Action 160 / Final response 160。
- DPO 仍为 decision/parameter/response = 320/200/280。
- AutoDL 完整测试：56 passed，1 个与 OPD 实验 API 有关的非阻断 warning。

v1.2/v1.2.1 均保存在独立目录，没有覆盖已经人工审核的 v1.1.1。新版本尚未完成人工抽检，不能直接升格为正式训练数据。

##3. 0.5B 受控实验

共同条件：Qwen2.5-0.5B-Instruct、LoRA、同一个 v1.2 SFT adapter、同一个 9 条开发 rollout 集、固定随机种子。v1.2.1 复用相同 SFT，只重训 DPO，以隔离决策配平的影响。

###训练指标

| 实验 | 样本/steps | 关键验证结果 |
|---|---|---|
| SFT v1.2 | 800 / 100 | eval loss 1.0838，perplexity 2.9559 |
| DPO v1.2 | 400 / 50 | eval loss 0.5020，preference accuracy 1.0000，reward margin 0.4467 |
| DPO v1.2.1 | 400 / 50 | eval loss 0.5064，preference accuracy 1.0000，reward margin 0.4309 |

###端到端结果

| 阶段 | task success | 通过场景 |
|---|---:|---|
| Initial/Instruct | 0/9 | 无 |
| SFT | 2/9 | policy boundary、tool failure |
| SFT + DPO v1.2 | 1/9 | order status |
| SFT + DPO v1.2.1 | 1/9 | order status |

v1.2 DPO 相对 SFT 的变化：

- `forbidden_tool_absent`：0.6667 → 1.0000；
- `facts_faithful`：0.7778 → 0.8889；
- `tool_selection_valid`：0.5556 → 0.4444；
- `task_success_rate`：0.2222 → 0.1111。

结论：DPO 学会了验证集中的 chosen/rejected 排序，也降低了越权调用和部分幻觉，但没有学会稳定完成多步任务。pairwise accuracy=1.0 与 rollout 退化同时发生，说明偏好分类指标不能替代可执行评测。

##4. 逐案例退化解释

v1.2 DPO 后，多数轨迹只执行一次 `query_order_status` 就给出最终回复；原 SFT 在政策查询、创建申请和身份核验场景通常会继续执行第二、第三步。典型现象包括：

- 创建申请：SFT 走到 query → policy → create；DPO 查询后提前结束；
- 重复申请：DPO 未充分利用 observation 中已有申请，转去给出错误政策判断；
- 身份未核验：避免了越权创建，但回复中又错误描述履约状态；
- 缺订单号：DPO 学会保守拒绝，但没有按要求向用户追问订单号。

v1.2.1 的 Action/Final 配平没有改变 9 条结果，因此简单类别配平不是充分修复。剩余问题更可能来自：

1. 0.5B 容量不足以稳定做多步状态跟踪；
2. 单步 DPO 对无法完整表达整条轨迹的长期收益；
3. 训练仅 400 对/50 steps，且开发集只有 9 条，方差很大；
4. 真实多轮澄清、追问和用户反馈覆盖仍不足。

##5. 正式 Rollout 集不能怎样构造

不能直接在 CSDS、DCH-2 或其他公开客服数据中筛选 500 条，然后把原回复当作标准答案。公开对话通常缺少本项目三个工具的真实 observation、订单状态、政策版本和可验证终态；直接筛选会得到“像客服对话”的文本集，而不是可执行评测集。

9 条现有 rollout 的意义是验证 runner、模拟器、评测器和失败归因是否闭环，并快速暴露首动作指标高估、DPO 过早停止等系统性问题。它不是为了估计正式模型效果，也不会扩写后继续冒充冻结测试集。

##6. 真正 Rollout 集的构造方法

采用“公开表达骨架 + 版本化业务世界 + 可执行 oracle”的方式：

1. **先按父对话划分**：CSDS、DCH-2 原始会话先去标识化、去重并按 parent conversation 划分 train/dev/test，之后才允许改写，避免同源表达泄漏。
2. **只抽取表达证据**：保留自然口语、情绪、追问、省略、纠错和多轮结构；不继承公开数据中的订单事实或答案。
3. **生成业务初始状态**：由模拟器生成订单、身份核验、履约、既有售后、超时和政策边界；所有 ID 均为合成脱敏 ID。
4. **定义用户目标**：例如查询状态、确认政策、创建换货、缺参澄清、已有申请防重、工具失败恢复。
5. **执行 oracle**：用三个真实 schema 在模拟器中执行期望轨迹，保存每一步 Action、参数、observation、允许分支和终态断言。
6. **表面实现多样化**：将公开表达骨架映射到业务状态，生成自然多轮用户话语；事实仍由模拟器控制。
7. **自动门禁**：schema、可执行性、终态、事实忠实、近重复、跨 split 泄漏、模板指纹和训练集污染全部通过后才进入候选集。
8. **双人审核并冻结**：正式 test 由两人独立审核关键事实、允许轨迹和判分规则；冻结后只新增新版本，不回填训练。

推荐规模：

| 子集 | 数量 | 目的 |
|---|---:|---|
| IID | 300～500 | 常见场景、自然表达和主流程 |
| compositional | 150～250 | 已见原子能力的新组合与多轮状态变化 |
| challenge | 150～250 | 缺参、矛盾、越权诱导、重复申请、超时、政策边界 |
| 合计 | 600～1,000 | 正式冻结评测；建议另留 60～100 条 pre-freeze dev |

详细字段、污染防护和冻结门槛见 `reports/ecommerce_rollout_dataset_spec.md`。

##7. 下一步门槛

1. 从 CSDS/DCH-2 构造 60～100 条 **pre-freeze rollout candidate**，先验证表达映射器与 oracle executor；不立即冻结 1,000 条。
2. 对 v1.2.1 做 120 对分层人工抽检：decision/parameter/response 各 40；decision 内 Action/Final 各 20。审核失败按 `primary_error` 和场景回流到生成规则。
3. 在 1.5B/3B 上用相同 SFT adapter 对照：SFT、DPO response-only、DPO 三层混合；至少 2 个 seed。
4. 只有当三层 DPO 在 pre-freeze rollout 上稳定不低于 SFT，且关键安全指标不退化，才扩到 5,000～10,000 对并进入 7B 主实验。
5. 9 条开发集继续保留为回归测试，但不用于简历效果数字。

当前能写进简历的是：实现可执行 Function Calling 环境、三层偏好数据、全局去重/泄漏审计、Initial/SFT/DPO 受控评测，并通过负实验发现 pairwise 指标与任务成功率不一致。当前不能写“DPO 提升任务成功率”。
