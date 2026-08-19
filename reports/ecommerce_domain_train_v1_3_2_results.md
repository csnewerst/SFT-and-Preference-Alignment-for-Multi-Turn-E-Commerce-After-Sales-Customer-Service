# 推荐规模中文训练候选 v1.3.2 构建记录
>
> 状态：自动审计与 100 条风险分层人工复核通过，可进入 1.5B/3B 消融；不得直接标记为正式 7B 训练集

##1. 结论

已从 CSDS/DCH-2 的 train/validation 来源池构建推荐规模候选：SFT 12,000 条、DPO 5,000 对。没有读取两套来源的 test 父样本，数据文件仅保存在 AutoDL 的 Git 忽略目录。

v1.3.2 不是把旧版 2,000/800 简单复制扩写：它增加了来源与 split 白名单、同一父样本的独立业务重构变体、多步每个工具位置的 continue/stop 决策偏好、跨轮参数偏好，以及去重后的 split×来源×场景重采样。

##2. 输入边界

- 来源：仅 `csds-emnlp21`、`dch2-dialeval2`；
- 来源 split：仅 train、validation；
- 原始父样本：13,508；
- 每个父样本最多生成 2 个独立业务重构变体；
- 原始 SFT 候选：27,016；
- test 来源父样本：0；
- 订单事实、工具 observation 与售后结论由版本化模拟器产生，不把公开数据原答案当作业务真值。

##3. 最终规模与分布

| 数据 | 总量 | Train | Validation | Test |
|---|---:|---:|---:|---:|
| SFT | 12,000 | 10,800 | 1,200 | 0 |
| DPO | 5,000 | 4,436 | 564 | 0 |

SFT 来源：CSDS 7,301；DCH-2 4,699。

SFT 场景：

| 场景 | 数量 |
|---|---:|
| damaged_exchange | 967 |
| wrong_item_return | 924 |
| missing_item_refund | 964 |
| duplicate_request | 1,350 |
| identity_required | 1,029 |
| missing_order_id | 1,584 |
| no_reason_expired | 1,242 |
| not_delivered | 934 |
| order_not_found | 1,512 |
| tool_timeout | 1,494 |

`missing_order_id + order_not_found + tool_timeout` 占 38.25%。相较未重采样 v1.3.1 的约 46% 已下降，但仍应在人工审核与 1.5B/3B 消融中重点监控“过早停止”。

##4. DPO 结构

| 偏好层级 | 数量 |
|---|---:|
| decision | 2,000 |
| parameter | 1,250 |
| response | 1,750 |

decision chosen 目标严格配平：Action 1,000 / Final response 1,000。

主要负例：

- `premature_stop`：836；
- `invalid_argument`：1,250；
- `unnecessary_tool`：865；
- `hallucinated_state`：892；
- `policy_violation`：608；
- `missing_argument`：549。

对多步 SFT 轨迹，不再只监督最后一个工具调用：每个中间工具位置都可产生“正确继续调用 vs 提前给最终回复”的 decision pair；每个工具位置都可产生正确参数 vs 缺失/非法参数的 parameter pair。

##5. 自动审计

SFT 与 DPO 均通过：

- schema 与三工具定义；
- sample/group ID；
- PII；
- 精确重复与 SimHash 近重复；
- group/content/near-duplicate 跨 split 泄漏；
- DPO chosen/rejected 目标类型；
- preference level 与 target turn index。

内容集哈希：

- SFT：`58578c61decb75e8c13c876655886a5953e2e4829c74f923c86e7cc355a8a5c6`
- DPO：`69186aefa94f12f1ac5d3f5c276307f373828fabbaf1975ee31426991faef96d`

##6. 规模扩展中发现的问题

###v1.3.0

单个父样本只生成一个重构，去重后仅剩 9,039 条 SFT，低于 10k 下限；DPO 5,000 已通过。

###v1.3.1

每个父样本支持两个独立重构后，SFT 达到 12,000、DPO 达到 5,000，但保守终止三类约占 46%。原因是多步成功轨迹更长、更相似，更容易被近重复门禁淘汰。

###v1.3.2

在不放宽去重阈值的情况下，先全局去重，再按 split×来源×场景确定性轮转抽样。最终 train/validation 为 90/10，保守终止三类降到 38.25%，同时保持 12k/5k 和零审计错误。

##7. 机器预审池与人工复核

已确定性抽取 400 条机器预审池：SFT 200、DPO 200。该池用于覆盖来源、场景、split 和偏好层级，不再要求非业务专家逐条人工填写全部客观字段。

- 来源：CSDS 230、DCH-2 170；
- 覆盖全部 10 个场景；
- DPO：decision 92、parameter 51、response 57；
- 队列 SHA-256：`8e90baed2720f9e857b5e3a7a73fea7e4f8c2dddd7feeec71a482edeec85522a`。

在机器预审池上继续确定性筛选 100 条人工复核：SFT 60、DPO 40，其中约 60% 为身份、重复申请、超期、工具超时、政策/幻觉/提前终止等高风险样本，约 40% 为随机分层控制样本。

程序负责结构、工具定义、参数 JSON、observation 配对、DPO 成对结构及上游模拟器门禁。人工只填写三个直观字段：表达是否自然、事实是否有工具或上下文依据、DPO chosen 是否明显更优；无法判断可明确标记为“无法确定”并进入仲裁，而不是强制猜测。

准入建议：机器门禁错误为 0；事实有依据率不低于 98%；自然表达通过率不低于 95%；DPO 偏好清晰率不低于 98%；“无法确定”样本必须完成仲裁。任一来源、场景或偏好层级出现系统性错误时，修生成规则并重建，不直接手改训练样本。

人工复核已完成：100/100 已填写，无仲裁项；自然表达 100/100、事实有依据 100/100、DPO 偏好清晰 40/40。归一化后的 48 种 SFT 最终回复、34 种 chosen 和 33 种 rejected 模式未发现明显系统性错误。该结果属于项目作者单人复核，不宣称专家双盲标注；完整记录见 `reports/ecommerce_domain_train_v1_3_2_human_review.md`。

##8. 下一步

1. 已完成 100 条风险分层人工复核；400 条仅保留为机器预审与选样母池。
2. 固定 v1.3.2 内容哈希，不因本轮审核发布 v1.3.3。
3. 用固定 800 条 pre-freeze 评测候选，在 1.5B/3B 上比较 SFT、response-only DPO、v1.3.2 多粒度 DPO。
4. 只有 DPO 在总体 task success、Compositional、多步创建和参数正确率上稳定优于 SFT，才进入 Qwen2.5-7B 主实验；正式验收前再对 10～20 条困难样本补充独立复核。
