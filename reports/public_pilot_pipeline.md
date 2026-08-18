# 公开数据 pilot 流水线

## 目标

公开数据首先进入“来源证据层”，不直接成为 SFT 或 DPO 标签。证据层只提取客服表达、多轮结构、主题和工具轨迹模式；订单事实、售后政策、工具名、observation 和最终答案必须在下一阶段由本项目的版本化配置与确定性模拟器重新生成。

机器可读来源登记位于 `configs/ecommerce/public_sources_v1.json`，处理入口为 `scripts/ecommerce/prepare_public_pilot.py`。

## 来源策略

| 来源 | 自动下载 | 进入证据层的用途 | 训练限制 |
|---|---:|---|---|
| Bitext | 是 | 英文客服意图、错别字、语气和表达变体 | 原答案不得直接成为中国电商政策标签 |
| glaive-function-calling-v2 | 是 | 调用、不调用、多轮补参和 observation 轨迹结构 | 原工具名和原 observation 不进入最终训练集 |
| CSDS | 否 | 中文真实电商多轮结构、主题和角色摘要 | 官方仓库未声明数据许可；仅处理已获授权的本地文件 |
| DCH-2 | 否 | 帮助台多轮结构、任务完成和对话效率标注 | 需 user agreement；对话限非商业研究/教育用途 |

CSDS 与 DCH-2 已于 2026-08-09 完成外部权利核验，处理时仍必须同时提供 `--input` 和 `--rights-acknowledged`，以把授权状态写入 manifest。该参数不改变原数据条款，也不会自动绕过官方访问流程。

## 可复现处理

自动来源在 AutoDL 直接执行：

```bash
python scripts/ecommerce/prepare_public_pilot.py \
  --source bitext-support-2024 \
  --hf-endpoint https://hf-mirror.com \
  --output-root data/ecommerce/public_pilot

python scripts/ecommerce/prepare_public_pilot.py \
  --source glaive-fc-v2 \
  --hf-endpoint https://hf-mirror.com \
  --output-root data/ecommerce/public_pilot
```

受限来源示例：

```bash
python scripts/ecommerce/prepare_public_pilot.py \
  --source csds-emnlp21 \
  --input data/ecommerce/authorized/csds \
  --rights-acknowledged \
  --output-root data/ecommerce/public_pilot
```

流水线会：

1. 将 Hugging Face `main` 解析为不可变 commit；本地输入计算文件树哈希。
2. 用固定 seed 和内容哈希进行确定性抽样。
3. 统一为 `turns + labels + provenance` 的来源证据格式。
4. 扫描 PII，拒绝记录只保存哈希与原因，不复制敏感原文。
5. 做规范化精确去重，并在 group 不跨 split 的前提下确定性逼近 80/10/10 切分。
6. 输出 `manifest.json`、`report.json`、逐条拒绝原因和所有产物 SHA-256。

`raw_selected.jsonl`、规范化数据和报告均位于被 Git 忽略的 `data/ecommerce/`。报告中的来源版本、过滤漏斗和产物哈希进入实验记录；原始或生成数据不提交仓库。

## 下一阶段接口

证据层的每条记录都有稳定的 `source_record_id` 和 `group_id`。领域重构必须以它们作为 `parent_id`/分组键，先完成 split，再生成中文改写、工具轨迹和偏好负例，最后通过 `audit_ecommerce_data.py --require-metadata`。这样同一公开样本的所有改写不会跨 split 泄漏。
