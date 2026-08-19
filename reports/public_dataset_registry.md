# 电商售后公开数据源登记 v1
##1. 登记原则

公开数据只作为语言表达、多轮结构和 Function Calling 模式的原料，不能直接定义本项目的订单事实、售后政策或冻结测试答案。每个来源进入处理流水线前必须记录版本、许可、用途、PII 风险和采用决策。

决策状态：

- `adopt-pilot`：允许下载少量样本进入审计和 pilot 改写。
- `hold-license`：许可证或使用条款未确认前不下载、不训练。
- `reject`：与任务不匹配或风险不可接受。

所有来源都必须经过字段审计、脱敏、精确去重、近重复检查和领域改写。来自公开数据的原始对话及其改写不得进入冻结测试集。

##2. 候选来源总表

| ID | 数据集 | 语言/规模 | 已知许可或条款 | 拟用范围 | 决策 |
|---|---|---|---|---|---|
| `bitext-support-2024` | Bitext Customer Support LLM Chatbot Training Dataset | 英文，26,872 行 | CDLA-Sharing-1.0 | 意图表达、口语/错别字/礼貌与情绪变体 | `adopt-pilot` |
| `glaive-fc-v2` | glaive-function-calling-v2 | 英文，约 113k 行 | Apache-2.0 | 工具 schema、调用/不调用、缺参追问结构 | `adopt-pilot` |
| `csds-emnlp21` | CSDS | 中文客服多轮对话 | 官方仓库未明确展示数据许可证 | 多轮客服语气、问题与处理进展结构 | `hold-license` |
| `dch2-dialeval2` | DCH-2 | 中英各 4,390 段 | 需要提交 user agreement，受单独条款约束 | 多轮帮助台结构、任务完成与对话质量参考 | `hold-license` |

##3. 来源明细

###3.1 Bitext Customer Support LLM Chatbot Training Dataset

- 登记 ID：`bitext-support-2024`
- 官方页面：https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- 仓库版本：`main@430d1a8`（正式下载时再次解析完整 commit）
- 页面记录规模：26,872 行，19.2 MB
- 格式/语言：CSV，英文
- 页面许可证：`cdla-sharing-1.0`
- 内容特征：包含意图、类别和多种语言变体标签，例如否定、礼貌、口语、冒犯、缩写和错别字。
- PII 风险：页面未提供足以免除逐字段审计的信息；pilot 下载后仍按中风险处理。
- 采用用途：
  - 客服意图和表达多样性原料。
  - 对抗表达、错别字、简写和情绪化输入模板。
  - 不直接使用原始答案作为中国电商售后政策答案。
- 禁止用途：
  - 不进入冻结测试集。
  - 不把银行、电信等原领域事实直接映射为电商事实。
  - 不把许可标签视为已完成全部合规审计。
- pilot 决策：`adopt-pilot`，先抽取最多 1,000 行做字段、重复率和领域覆盖审计。

###3.2 glaive-function-calling-v2

- 登记 ID：`glaive-fc-v2`
- 官方页面：https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
- 仓库版本：`main@e7f4b64`（正式下载时再次解析完整 commit）
- 页面记录规模：约 113k 行
- 格式/语言：JSON，英文
- 页面许可证：`apache-2.0`
- 内容特征：system 中提供函数签名，chat 中包含调用、函数响应、自然语言回答和缺参追问。
- PII 风险：以合成工具调用为主，但仍需扫描自由文本中的邮箱、电话、密钥样式和真实人物信息。
- 采用用途：
  - 学习 Function Calling 轨迹结构。
  - 提取调用、无需调用、缺参先追问和调用后解释的模式。
  - 经领域重写后映射到本项目三个工具。
- 禁止用途：
  - 不保留原始工具名作为电商工具训练目标。
  - 不直接把新闻、天气、密码生成等领域 observation 混入客服数据。
  - 不进入冻结测试集。
- pilot 决策：`adopt-pilot`，按轨迹类型分层抽取最多 2,000 行，先统计解析成功率和模板污染率。

###3.3 CSDS

- 登记 ID：`csds-emnlp21`
- 官方仓库：https://github.com/xiaolinAndy/CSDS
- 论文：https://aclanthology.org/2021.emnlp-main.365/
- 版本：官方仓库 `main`，正式采用前固定 commit
- 语言/任务：中文客服多轮对话摘要；提供整体、用户视角和客服视角摘要，以及主题片段。
- 已知许可证：经核验，官方 README 和仓库根目录未明确展示数据许可证。
- 数据与隐私：论文说明数据来自 JDDC 的真实电商售前/售后对话，并称隐私信息已按 JDDC 方式匿名化；进入本项目后仍必须二次扫描。
- 潜在用途：
  - 中文多轮客服表达和对话进展结构。
  - 用户诉求、客服动作和主题切换的抽取原料。
- 限制：任务本身是摘要，不包含本项目的工具状态和政策，不可直接作为工具调用监督数据。
- 当前决策：`adopt-authorized-research`。已完成外部权利核验；代码通过本地授权文件和 `--rights-acknowledged` 留存处理证据，不在仓库再分发原数据。

###3.4 DCH-2

- 登记 ID：`dch2-dialeval2`
- 官方页面：https://dialeval-2.github.io/DCH-2/
- 论文：https://arxiv.org/abs/2104.08755
- 语言/规模：4,390 段中文客户—帮助台对话及对应英文人工翻译。
- 标注：每轮 nugget 类型；对话级任务完成、客户满意和交流效率分数。
- 来源：官方页面说明对话抓取自微博。
- 隐私：官方页面说明电话和邮箱等敏感信息已做掩码，但仍需二次 PII 扫描。
- 访问条款：必须填写 user agreement。官方条款说明对话仅限非商业研究和教育用途，组织者不拥有对话版权；标注使用 CC BY 4.0。
- 潜在用途：
  - 多轮帮助台结构和任务是否解决的参考。
  - 构造任务完成与无效轮次的分析规则。
- 限制：微博帮助台场景并不等于电商售后；数据再分发和训练用途必须服从单独协议。
- 当前决策：`adopt-authorized-research`。已完成外部权利核验；仅用于非商业研究项目，通过本地授权文件接入，不再分发原始对话。

##4. Pilot 接入顺序

1. 自动接入 `bitext-support-2024`，建立通用客服表达清洗和领域筛选。
2. 自动接入 `glaive-fc-v2`，建立工具轨迹解析和负例分类；三个工具的领域重写另行生成。
3. CSDS 与 DCH-2 已完成权利核验并提供授权本地适配器；取得官方文件后纳入同一证据层。
4. 四个来源都只进入来源证据层，不直接作为政策答案、工具 observation 或冻结测试集。

##5. 每次下载必须生成的 manifest

```json
{
  "source_id": "glaive-fc-v2",
  "resolved_revision": "full commit hash",
  "downloaded_at": "ISO-8601 UTC timestamp",
  "source_url": "https://...",
  "license": "apache-2.0",
  "raw_files": [
    {"path": "...", "sha256": "...", "bytes": 0}
  ],
  "raw_rows": 0,
  "accepted_rows": 0,
  "rejected_rows": 0,
  "pii_scan_version": "...",
  "transform_version": "..."
}
```

manifest 可以提交，但原始数据、清洗后的训练数据及生成对话继续保留在 `data/ecommerce/` 忽略目录中。

##6. 下载前检查清单

- [ ] 解析并记录完整 commit/revision。
- [ ] 保存数据卡和许可证快照 URL。
- [ ] 确认下载命令不会写到 Conda 或系统 CUDA 目录。
- [ ] 确认目标位于项目 `data/ecommerce/raw/<source_id>/`。
- [ ] 计算每个原始文件的 SHA-256。
- [ ] 统计字段、空值、语言、长度和重复率。
- [ ] 扫描姓名、电话、邮箱、地址、账号、订单号和密钥样式。
- [ ] 抽检至少 100 行并记录采用/拒绝理由。
- [ ] 检查与冻结测试集的文本和模板泄漏。
