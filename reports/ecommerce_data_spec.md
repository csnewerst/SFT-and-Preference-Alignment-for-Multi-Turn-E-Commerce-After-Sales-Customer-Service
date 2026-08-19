# 电商售后数据协议 v1

##1. 目的与适用范围

本协议固定首版电商售后 SFT、DPO、工具模拟和自动评测共享的业务语义。目标不是让模型记忆订单事实，而是训练并评测以下行为：

1. 从多轮对话中识别诉求并补齐必要参数。
2. 需要动态事实时选择正确工具，不猜测订单或售后状态。
3. 忠实解释工具 observation，不篡改结果。
4. 遵守售后政策和身份权限边界。
5. 给出明确、可执行且不过度承诺的下一步。

机器可执行配置位于：

- `configs/ecommerce/tools_v1.json`
- `configs/ecommerce/policies_v1.json`
- `configs/ecommerce/scenarios_v1.json`

文档解释语义，JSON 配置是模拟器、数据生成和规则评测的单一事实源。若两者冲突，以同一版本的 JSON 配置为准，并在修订文档后提升版本号。

##2. 版本与数据边界

- 数据协议版本：`1.0.0`
- 工具 schema 版本：`1.0.0`
- 政策版本：`2026-08-v1`
- 场景 fixture 版本：`1.0.0`
- 模型起点命名：`Initial/Instruct`
- 主实验命名：`SFT`、`SFT+DPO`
- 可选实验命名：`SFT+OPD`

真实订单、用户身份、地址、电话、支付信息和客服工单不得进入仓库。训练集、偏好集、冻结测试集及其生成结果放在被 Git 忽略的 `data/ecommerce/` 下。仓库只提交规范、配置、生成代码、manifest、统计和不包含用户数据的测试 fixture。

##3. 核心实体

###3.1 Order

| 字段 | 类型 | 说明 |
|---|---|---|
| `order_id` | string | 脱敏订单号，格式为 `EC-*` |
| `payment_status` | enum | 首版使用 `paid`，为未来未支付场景保留扩展位 |
| `fulfillment_status` | enum | `shipped` 或 `delivered` |
| `days_since_delivery` | integer/null | 未签收时为 null |
| `identity_verified` | boolean | 只供权限判断，不通过查询工具暴露 |
| `item_summary` | string | 不包含真实个人信息的商品摘要 |
| `after_sales_request` | object/null | 当前售后申请状态 |
| `tool_failures` | object | 确定性故障注入，不进入模型自由生成内容 |

###3.2 AfterSalesRequest

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | string | 模拟器确定性生成的售后申请号 |
| `order_id` | string | 所属订单 |
| `request_type` | enum | `refund_only`、`return_refund`、`exchange` |
| `reason` | enum | 与 `issue_type` 共用原因代码 |
| `status` | enum | 首版创建后固定为 `submitted` |
| `evidence_required` | boolean | 是否需要用户后续补充凭证 |

###3.3 IssueType

首版固定五类问题：

- `no_reason`：无理由退货
- `damaged`：商品破损
- `wrong_item`：错发商品
- `missing_item`：漏发商品
- `quality_issue`：其他质量问题

训练数据中可以保留用户的自然语言原句，但传给工具的参数必须映射为上述稳定代码。

##4. 三个工具

首版只能使用以下三个工具，不增加语义重叠的物流、退款进度或转人工工具。

###4.1 `query_order_status`

用途：查询订单、支付、履约和已有售后申请。订单动态事实必须来自此工具。

必填参数：

```json
{"order_id": "EC-1001"}
```

允许在以下情况调用：

- 用户询问订单、发货、签收或已有售后状态。
- 后续政策判断依赖订单当前状态。
- 创建售后申请前需要确认订单事实。

禁止行为：

- 缺少订单号时猜测订单。
- 把历史对话中的旧状态当成当前状态。
- 将 `identity_verified` 等内部权限字段告诉用户。

###4.2 `check_return_policy`

用途：根据订单状态、签收时长和问题类型查询允许的售后动作。

必填参数：

```json
{"order_id": "EC-1001", "issue_type": "damaged"}
```

调用前必须知道订单号和明确的问题类型。若用户只说“有问题”或“想售后”，应先追问，不能任选一个 `issue_type`。

###4.3 `create_after_sales_request`

用途：创建退款、退货退款或换货申请。

必填参数：

```json
{
  "order_id": "EC-1001",
  "request_type": "exchange",
  "reason": "damaged"
}
```

调用前置条件：

1. 订单存在且身份已核验。
2. 问题类型已经明确。
3. `check_return_policy` 的结果允许该 `request_type`。
4. 用户已经明确选择或确认该售后动作。
5. 当前订单不存在重复售后申请。

模型不得把“我帮你看看”解释成用户已经授权创建售后单。

##5. 工具 observation 协议

成功响应：

```json
{
  "ok": true,
  "tool": "query_order_status",
  "data": {}
}
```

失败响应：

```json
{
  "ok": false,
  "tool": "query_order_status",
  "error": {
    "code": "ORDER_NOT_FOUND",
    "message": "未找到该订单。",
    "retryable": false
  }
}
```

模型只能依据响应中出现的事实作答。`retryable=true` 时可以说明服务暂时异常并建议稍后重试；`retryable=false` 时不得承诺重试一定成功。

首版错误代码：

| 错误代码 | 可重试 | 推荐行为 |
|---|---:|---|
| `INVALID_ARGUMENTS` | 否 | 修正参数或向用户追问 |
| `UNKNOWN_TOOL` | 否 | 修正工具选择 |
| `ORDER_NOT_FOUND` | 否 | 请用户核对订单号 |
| `ORDER_NOT_PAID` | 否 | 解释当前不能申请售后 |
| `ORDER_NOT_DELIVERED` | 否 | 说明尚未满足当前政策前置条件 |
| `RETURN_WINDOW_EXPIRED` | 否 | 解释超期，不承诺例外赔付 |
| `IDENTITY_NOT_VERIFIED` | 否 | 引导完成身份核验 |
| `DUPLICATE_REQUEST` | 否 | 查询或解释已有申请 |
| `REQUEST_TYPE_NOT_ALLOWED` | 否 | 提供政策允许的选项 |
| `UPSTREAM_TIMEOUT` | 是 | 说明暂时失败并建议稍后重试 |

##6. 售后政策 v1

政策由 `policies_v1.json` 执行，核心规则如下：

- 无理由退货：签收后 7 天内，只允许 `return_refund`。
- 破损：签收后可申请 `refund_only`、`return_refund` 或 `exchange`，需要凭证。
- 错发：签收后可申请 `return_refund` 或 `exchange`，需要凭证。
- 漏发：签收后只允许 `refund_only`，需要凭证。
- 质量问题：签收后可申请 `return_refund` 或 `exchange`，需要凭证。
- 未签收订单不直接创建上述售后申请。
- 身份未核验时不得创建申请。
- 已有售后申请时不得重复创建。

本规则是用于可复现实验的抽象政策，不复制任何平台受版权保护的完整规则，也不代表真实平台承诺。

##7. 多轮状态机

```text
START
  -> COLLECT_ORDER_ID
  -> QUERY_ORDER
  -> CLARIFY_ISSUE
  -> CHECK_POLICY
  -> CONFIRM_ACTION
  -> CREATE_REQUEST
  -> RESOLVED
```

允许的恢复分支：

- 任一步缺少必要参数：留在当前状态并追问。
- `ORDER_NOT_FOUND`：返回 `COLLECT_ORDER_ID`。
- 可重试工具错误：进入 `TOOL_RETRY_PENDING`，不得编造结果。
- 政策不允许：进入 `POLICY_EXPLAINED`，给出允许的下一步但不创建申请。
- 身份未核验：进入 `IDENTITY_VERIFICATION_REQUIRED`。
- 重复申请：进入 `EXISTING_REQUEST_FOUND`。

状态机描述决策顺序，不要求模型输出内部状态名称。

##8. SFT 数据格式

SFT 保持 MedicalGPT 的 ShareGPT 兼容结构：

```json
{
  "conversations": [
    {"from": "human", "value": "订单 EC-1001 的耳机破损了，想换货。"},
    {
      "from": "function_call",
      "value": "{\"name\":\"query_order_status\",\"arguments\":{\"order_id\":\"EC-1001\"}}"
    },
    {
      "from": "observation",
      "value": "{\"ok\":true,\"tool\":\"query_order_status\",\"data\":{\"order_id\":\"EC-1001\",\"fulfillment_status\":\"delivered\"}}"
    },
    {"from": "gpt", "value": "订单已签收。我再为你核对破损换货政策。"}
  ],
  "tools": []
}
```

要求：

- `tools` 使用 `tools_v1.json` 中的定义，不在样本间随意改写。
- `function_call.value` 是包含 `name` 和 `arguments` 的 JSON 字符串。
- `observation.value` 必须来自确定性模拟器。
- 工具调用和最终答复都属于 assistant 训练目标。
- 用户文本不能包含真实 PII。

样本旁路 manifest 至少记录：

- `sample_id`
- `parent_id`
- `source_id`
- `scenario`
- `intent`
- `difficulty`
- `policy_version`
- `tool_schema_version`
- `generator_version`
- `review_status`

##9. DPO 数据格式与标签

DPO 样本使用：

```json
{
  "conversations": [{"from": "human", "value": "……"}],
  "chosen": "……",
  "rejected": "……",
  "tools": []
}
```

每个偏好对只突出一个主要行为差异，避免 chosen 同时更长、更礼貌、格式更整齐造成伪偏好。

主错误标签：

| 标签 | 定义 |
|---|---|
| `wrong_tool` | 选择了错误工具 |
| `missing_argument` | 必填参数缺失或格式错误 |
| `policy_violation` | 违反政策、权限或用户确认边界 |
| `hallucinated_state` | 编造订单、物流、退款或售后状态 |
| `unnecessary_tool` | 本可直接回答却调用工具 |
| `missed_tool` | 需要动态事实却未调用工具 |
| `wrong_escalation` | 错误决定是否升级处理 |
| `observation_misread` | 错读或篡改工具结果 |
| `incomplete_resolution` | 未给出必要下一步 |
| `tone_or_verbosity` | 语气或冗长度问题，只作低优先级标签 |

偏好 manifest 额外记录 `primary_error`、`secondary_errors`、`pair_source` 和 `review_reason`。

##10. 数据切分与泄漏控制

必须先按以下分组键切分，再生成或改写变体：

1. 原始会话 ID。
2. 订单实体 ID。
3. 场景模板族 ID。
4. 父样本 ID。

同一分组只能出现在 train、validation、test 之一。禁止先随机拆分，再从训练样本生成测试改写。冻结测试集建立后只允许新增版本，不原地替换失败样本。

首版冻结测试集 v1 先覆盖约 100 条合成案例：

- 30% 常规流程
- 15% 缺参追问
- 15% 工具选择与参数困难样本
- 15% 政策边界
- 10% 工具异常恢复
- 10% 反幻觉与 observation 忠实度
- 5% 多诉求或多轮状态冲突

正式实验再扩展至 500～1,000 条，并保留版本化 manifest 和内容哈希。

##11. 自动评测契约

每个测试样本必须提供可程序判定的期望：

- 是否需要工具。
- 允许的工具序列。
- 每次调用的必填参数和允许值。
- 确定性 observation。
- 是否允许创建售后申请。
- 禁止出现的状态或承诺。
- 期望终止状态。

规则评测器至少输出：

- `tool_selection_accuracy`
- `argument_completeness`
- `tool_json_valid_rate`
- `observation_faithfulness`
- `policy_violation_rate`
- `task_success_rate`
- 逐样本错误标签和证据

所有聚合指标都必须能追溯到逐样本记录，不以单一 LLM judge 替代规则判定。

##12. Phase 1 验收条件

Phase 1 完成需同时满足：

1. 三个工具的 schema 可由程序加载并校验。
2. 模拟器对相同初始状态和调用序列返回相同结果。
3. 正常、缺参、越权、超期、重复申请和工具超时均有测试。
4. 冻结测试集 v1 的生成器、manifest 和哈希可复现。
5. 不训练模型也能说明每条测试样本的规则、期望工具轨迹和成功判据。
6. 仓库不包含真实或生成训练数据、密钥、模型和 checkpoint。
