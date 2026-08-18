# 电商售后领域重构 pilot v1 结果

运行日期：2026-08-09（Asia/Shanghai）

生成器版本：`1.0.0`

代码提交：`fb045f9`

运行环境：AutoDL base；数据位于 Git 忽略的 `data/ecommerce/domain_pilot_v1/`。

## 数据规模

| 任务 | 总量 | train | validation | test | 内容集合 SHA-256 |
|---|---:|---:|---:|---:|---|
| SFT | 2,000 | 1,543 | 226 | 231 | `389d7206a1181a21578e85cff291d6081829971ff15fc0efd9d873babe9a0769` |
| DPO | 800 | 600 | 96 | 104 | `af38ef86e3024d7be748c978e653d87a35c695327dd1cfbcf3bb549196846ee6` |

来源贡献：

| 来源 | SFT | DPO |
|---|---:|---:|
| Bitext | 679 | 265 |
| glaive-function-calling-v2 | 1,321 | 535 |

生成前共考虑 2,850 条来源证据。为满足汉明距离不大于 3 的 SimHash 门禁，SFT 淘汰 572 条近重复候选，DPO 淘汰 295 对近重复候选。

## 场景分布

| 场景 | SFT | DPO |
|---|---:|---:|
| `damaged_exchange` | 135 | 58 |
| `wrong_item_return` | 104 | 44 |
| `missing_item_refund` | 145 | 60 |
| `no_reason_expired` | 339 | 114 |
| `not_delivered` | 218 | 99 |
| `identity_required` | 118 | 49 |
| `duplicate_request` | 414 | 158 |
| `tool_timeout` | 93 | 42 |
| `missing_order_id` | 310 | 123 |
| `order_not_found` | 124 | 53 |

## 质量门禁

- SFT 严格 metadata/schema/PII/重复/跨 split 泄漏审计：`passed=true`，零 issue。
- DPO 严格审计：`passed=true`，零 issue。
- 所有订单事实、政策结果和工具 observation 均由版本化配置与确定性模拟器生成。
- 每条样本记录 `sample_id`、`parent_id`、`group_id`、来源、场景、意图、政策版本、工具版本和生成器版本。
- DPO 每对记录单一主错误、构造来源与审核理由。
- MedicalGPT 本地 JSON 数据加载器验证通过，SFT/DPO 列结构与训练器兼容。

文本长度（字符）：

| 任务 | min | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|
| SFT | 81 | 791 | 1,222 | 1,228 | 1,238 |
| DPO | 117 | 827 | 1,258 | 1,265 | 1,273 |

## 当前限制与下一验收项

- 当前正式 pilot 的实际来源为 Bitext 与 glaive；CSDS/DCH-2 已完成权利核验和适配器测试，但 AutoDL 尚未取得官方数据文件。
- 自动规则可以证明结构和业务状态一致，不能替代对自然度、模板感、chosen/rejected 难度和语气的人工判断。
- 下一步按来源、场景、split 和 DPO 错误类型分层抽取至少 100 条，完成人工盲审并记录通过率与修改原因；通过后再运行 0.5B SFT+DPO 全链路。
