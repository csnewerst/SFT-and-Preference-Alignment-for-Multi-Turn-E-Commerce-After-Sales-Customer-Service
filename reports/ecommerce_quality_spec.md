# 电商售后数据质量门禁 v1

本文档定义进入 SFT、DPO 和正式评测之前必须执行的数据质量检查。真实数据、生成数据和审计产物均保存在 Git 忽略的 `data/ecommerce/` 下；仓库只提交检查代码、规范和合成测试 fixture。

##输入约定

- 数据目录递归读取 `*.jsonl`，路径必须能识别为 `train`、`validation` 或 `test`。
- 每行必须是包含非空 `conversations` 的 JSON 对象；DPO 的 `chosen` 与 `rejected` 必须成对出现。
- `function_call.value` 和 `observation.value` 必须是合法 JSON；工具名必须来自 `configs/ecommerce/tools_v1.json`。
- 最终标准化数据使用 `--require-metadata`，每行需要 `sample_id`、`group_id`、`source_id`、`scenario` 和 `intent`。
- 元数据可以放在样本的 `metadata` 对象内，也可以通过 JSONL sidecar 提供。sidecar 每行以相对路径 `file` 和从 1 开始的 `row_index` 定位样本。

##阻断规则

以下问题产生 `error`，命令返回非零状态：

- 无 JSONL、未知 split、非法 schema、非法工具 JSON 或未知工具；
- 邮箱、中国大陆手机号、身份证号、银行卡号、私钥、常见 API token 或详细中文地址；
- 重复 `sample_id`；
- 同一 `group_id` 出现在多个 split；
- 规范化内容在多个 split 完全重复；
- SimHash 近重复跨越不同 split；
- 严格模式下缺少必需元数据。

同一 split 内的完全重复和近重复当前记为 `warning`，需要在正式数据冻结前完成复核或去重。近重复默认使用字符 4-gram、64 位 SimHash 和汉明距离阈值 3；报告会注明阈值及候选对是否被上限截断。

##输出与执行

```bash
python scripts/ecommerce/audit_ecommerce_data.py \
  --dataset-root data/ecommerce/processed/sft_v1 \
  --metadata-manifest data/ecommerce/manifests/sft_v1.jsonl \
  --output-dir data/ecommerce/audits/sft_v1 \
  --require-metadata
```

输出：

- `report.json`：通过状态、文件/样本数、split/source/scenario 分布、文本长度分位数、问题计数和内容集合哈希；
- `issues.jsonl`：逐问题证据，包含文件、行号、样本 ID 和关联样本位置。

训练入口只接受 `report.json` 中 `passed=true` 的版本。数据内容或 manifest 发生变化后必须重新审计，不复用旧报告。
