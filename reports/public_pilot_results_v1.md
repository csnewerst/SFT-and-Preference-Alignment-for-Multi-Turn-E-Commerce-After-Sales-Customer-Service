# 公开数据来源证据层 pilot v1 结果
代码版本：`1b632cb`

抽样 seed：`20260809`
运行环境：AutoDL base，数据与缓存位于项目下被 Git 忽略的 `data/ecommerce/public_pilot/`

本报告只记录聚合统计、版本和哈希，不包含公开数据原文、生成训练数据或 PII。

##结果汇总

| 来源 | 固定 revision | 扫描 | 抽样 | 接受 | PII 拒绝 | 重复拒绝 | train/validation/test |
|---|---|---:|---:|---:|---:|---:|---:|
| Bitext | `430d1a89bd93bd1fa23c16f29dd53e73f0087443` | 26,872 | 1,000 | 998 | 2 | 0 | 798 / 100 / 100 |
| glaive-function-calling-v2 | `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` | 112,960 | 2,000 | 1,852 | 48 | 100 | 1,482 / 185 / 185 |

接受率：

- Bitext：99.8%
- glaive：92.6%

glaive 接受样本的轨迹结构分布：

| 轨迹类型 | 数量 |
|---|---:|
| `no_call` | 881 |
| `multi_turn_call` | 474 |
| `multi_call` | 302 |
| `single_call` | 195 |

##可复现证据

| 来源 | 抽样内容集合 SHA-256 | 原始抽样文件 SHA-256 |
|---|---|---|
| Bitext | `124fbfb26df61a755e1dba18e80d397a7a8fcdc797a1c178a74b11b31712cfe7` | `efa653865f50bd281f10fe48da0d63c2273c2d8a423ba330586e0c5efb34f37d` |
| glaive | `331dc1fac79a317b1c353783b90aed9136c6222ae4a080892a12316679fb879f` | `a166be7c8bf3563cac0076be61847673176d6b37913e9d53836448baa3babe34` |

group-aware 切分算法调整前后，上述抽样集合哈希和原始抽样文件哈希保持一致；变化仅发生在 split 归属。最终切分在 group 不跨 split 的约束下达到约 80/10/10。

远端代码测试：`42 passed`；`pip check`：无依赖冲突；Git 工作区干净。

##结论与限制

该结果证明来源解析、版本固定、确定性抽样、PII 过滤、精确去重、group-aware 切分、拒绝原因和产物哈希链路可以在真实公开数据上执行。

仍未满足最终训练数据验收：

- 当前产物是来源证据层，不是可直接训练的中文电商售后 SFT/DPO。
- Bitext 原答案和 glaive 原工具/observation 不作为最终训练标签。
- PII 当前以规则扫描为主，正式数据还需要 NER/词典补充和人工抽检。
- CSDS 未明确仓库许可证，DCH-2 需要 user agreement；两者适配器已测试，但未下载或运行真实数据。
- 尚未完成领域重构后的场景分布、长度分布、近重复率和人工抽检通过率。

下一步以 `source_record_id` 为父样本、`group_id` 为切分键，调用本项目工具模拟器重建中文多轮轨迹，生成 pilot SFT/DPO，再执行严格质量门禁与人工抽检。
