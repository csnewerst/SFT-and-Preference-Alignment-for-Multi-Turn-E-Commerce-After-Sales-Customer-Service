#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "configs" / "ecommerce"


class SimulatorConfigError(ValueError):
    """Raised when versioned simulator configuration is inconsistent."""


class EcommerceToolSimulator:
    """Deterministic, in-memory implementation of the three v1 after-sales tools."""

    EXPECTED_TOOLS = {
        "query_order_status",
        "check_return_policy",
        "create_after_sales_request",
    }

    def __init__(
        self,
        tools_config: Mapping[str, Any],
        policies_config: Mapping[str, Any],
        scenarios_config: Mapping[str, Any],
    ) -> None:
        self.tools_config = copy.deepcopy(dict(tools_config))
        self.policies = copy.deepcopy(dict(policies_config))
        self.scenarios = copy.deepcopy(dict(scenarios_config))

        tools = self.tools_config.get("tools")
        if not isinstance(tools, list):
            raise SimulatorConfigError("tools_v1.json must contain a tools list")
        self.tool_schemas = {tool.get("name"): tool for tool in tools if isinstance(tool, dict)}
        if set(self.tool_schemas) != self.EXPECTED_TOOLS:
            raise SimulatorConfigError(
                f"v1 must define exactly {sorted(self.EXPECTED_TOOLS)}, got {sorted(self.tool_schemas)}"
            )

        rules = self.policies.get("rules")
        if not isinstance(rules, dict) or not rules:
            raise SimulatorConfigError("policies_v1.json must contain non-empty rules")

        orders = self.scenarios.get("orders")
        if not isinstance(orders, list):
            raise SimulatorConfigError("scenarios_v1.json must contain an orders list")
        self.orders: Dict[str, Dict[str, Any]] = {}
        for order in orders:
            if not isinstance(order, dict) or not isinstance(order.get("order_id"), str):
                raise SimulatorConfigError("each scenario order must contain a string order_id")
            order_id = order["order_id"]
            if order_id in self.orders:
                raise SimulatorConfigError(f"duplicate scenario order_id: {order_id}")
            self.orders[order_id] = copy.deepcopy(order)

    @classmethod
    def from_config_dir(cls, config_dir: Path = DEFAULT_CONFIG_DIR) -> "EcommerceToolSimulator":
        return cls(
            _load_json(config_dir / "tools_v1.json"),
            _load_json(config_dir / "policies_v1.json"),
            _load_json(config_dir / "scenarios_v1.json"),
        )

    def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tool_schemas:
            return self._error(tool_name, "UNKNOWN_TOOL", f"未定义工具：{tool_name}")
        if not isinstance(arguments, Mapping):
            return self._error(tool_name, "INVALID_ARGUMENTS", "arguments 必须是 JSON object。")

        validation_error = self._validate_arguments(tool_name, arguments)
        if validation_error:
            return self._error(tool_name, "INVALID_ARGUMENTS", validation_error)

        order_id = arguments.get("order_id")
        injected_failure = self._injected_failure(tool_name, order_id)
        if injected_failure:
            return self._error(
                tool_name,
                injected_failure["code"],
                injected_failure["message"],
                retryable=bool(injected_failure.get("retryable", False)),
            )

        handlers = {
            "query_order_status": self._query_order_status,
            "check_return_policy": self._check_return_policy,
            "create_after_sales_request": self._create_after_sales_request,
        }
        return handlers[tool_name](dict(arguments))

    def snapshot(self) -> Dict[str, Any]:
        """Return a defensive copy for evaluation and reproducibility checks."""
        return copy.deepcopy(self.orders)

    def _validate_arguments(self, tool_name: str, arguments: Mapping[str, Any]) -> str | None:
        parameters = self.tool_schemas[tool_name].get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        missing = [name for name in required if name not in arguments]
        if missing:
            return f"缺少必填参数：{', '.join(sorted(missing))}。"

        if parameters.get("additionalProperties") is False:
            unexpected = sorted(set(arguments) - set(properties))
            if unexpected:
                return f"包含未定义参数：{', '.join(unexpected)}。"

        for name, value in arguments.items():
            schema = properties.get(name, {})
            if schema.get("type") == "string" and not isinstance(value, str):
                return f"参数 {name} 必须是字符串。"
            if isinstance(value, str) and not value.strip():
                return f"参数 {name} 不能为空。"
            if "enum" in schema and value not in schema["enum"]:
                return f"参数 {name} 不在允许值中：{value!r}。"
            if "pattern" in schema and isinstance(value, str) and not re.fullmatch(schema["pattern"], value):
                return f"参数 {name} 格式不合法：{value!r}。"
        return None

    def _injected_failure(self, tool_name: str, order_id: Any) -> Dict[str, Any] | None:
        if not isinstance(order_id, str) or order_id not in self.orders:
            return None
        failures = self.orders[order_id].get("tool_failures") or {}
        failure = failures.get(tool_name)
        return copy.deepcopy(failure) if isinstance(failure, dict) else None

    def _query_order_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        order = self.orders.get(arguments["order_id"])
        if order is None:
            return self._error("query_order_status", "ORDER_NOT_FOUND", "未找到该订单。")
        data = {
            "order_id": order["order_id"],
            "payment_status": order["payment_status"],
            "fulfillment_status": order["fulfillment_status"],
            "days_since_delivery": order.get("days_since_delivery"),
            "identity_verification_status": "verified" if order.get("identity_verified", False) else "required",
            "item_summary": order["item_summary"],
            "after_sales_request": copy.deepcopy(order.get("after_sales_request")),
        }
        return self._success("query_order_status", data)

    def _check_return_policy(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        order = self.orders.get(arguments["order_id"])
        if order is None:
            return self._error("check_return_policy", "ORDER_NOT_FOUND", "未找到该订单。")
        decision = self._evaluate_policy(order, arguments["issue_type"])
        return self._success("check_return_policy", decision)

    def _create_after_sales_request(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        order = self.orders.get(arguments["order_id"])
        if order is None:
            return self._error("create_after_sales_request", "ORDER_NOT_FOUND", "未找到该订单。")
        if not order.get("identity_verified", False):
            return self._error(
                "create_after_sales_request",
                "IDENTITY_NOT_VERIFIED",
                "用户身份尚未核验，不能创建售后申请。",
            )
        if order.get("after_sales_request") is not None:
            return self._error(
                "create_after_sales_request",
                "DUPLICATE_REQUEST",
                "该订单已有售后申请，不能重复创建。",
            )

        decision = self._evaluate_policy(order, arguments["reason"])
        if not decision["eligible"]:
            return self._error(
                "create_after_sales_request",
                decision["reason_code"],
                decision["message"],
            )
        if arguments["request_type"] not in decision["allowed_request_types"]:
            return self._error(
                "create_after_sales_request",
                "REQUEST_TYPE_NOT_ALLOWED",
                "当前问题不支持所选售后类型。",
            )

        request = {
            "request_id": f"ASR-{order['order_id']}-001",
            "order_id": order["order_id"],
            "request_type": arguments["request_type"],
            "reason": arguments["reason"],
            "status": "submitted",
            "evidence_required": decision["evidence_required"],
        }
        order["after_sales_request"] = copy.deepcopy(request)
        return self._success("create_after_sales_request", request)

    def _evaluate_policy(self, order: Mapping[str, Any], issue_type: str) -> Dict[str, Any]:
        rule = self.policies["rules"].get(issue_type)
        base = {
            "order_id": order["order_id"],
            "issue_type": issue_type,
            "policy_version": self.policies["policy_version"],
            "eligible": False,
            "allowed_request_types": [],
            "evidence_required": False,
        }
        if rule is None:
            return {
                **base,
                "reason_code": "UNSUPPORTED_ISSUE_TYPE",
                "message": "未配置该问题类型的售后政策。",
            }
        if order["payment_status"] not in self.policies["eligible_payment_statuses"]:
            return {**base, "reason_code": "ORDER_NOT_PAID", "message": "订单尚未支付，不能申请售后。"}
        if rule.get("requires_delivery") and order["fulfillment_status"] != "delivered":
            return {**base, "reason_code": "ORDER_NOT_DELIVERED", "message": "订单尚未签收。"}
        if rule.get("enforce_return_window"):
            days = order.get("days_since_delivery")
            window = self.policies["return_window_days"]
            if not isinstance(days, int) or days > window:
                return {
                    **base,
                    "reason_code": "RETURN_WINDOW_EXPIRED",
                    "message": f"已超过 {window} 天无理由退货时效。",
                }
        return {
            **base,
            "eligible": True,
            "allowed_request_types": list(rule["allowed_request_types"]),
            "evidence_required": bool(rule.get("evidence_required", False)),
            "reason_code": "ELIGIBLE",
            "message": "符合当前售后政策。",
        }

    @staticmethod
    def _success(tool_name: str, data: Mapping[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "tool": tool_name, "data": copy.deepcopy(dict(data))}

    @staticmethod
    def _error(
        tool_name: str,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "tool": tool_name,
            "error": {"code": code, "message": message, "retryable": retryable},
        }


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise SimulatorConfigError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic ecommerce tool call.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--arguments", required=True, help="JSON object with tool arguments")
    args = parser.parse_args()

    try:
        arguments = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        parser.error(f"--arguments is not valid JSON: {exc}")
    result = EcommerceToolSimulator.from_config_dir(args.config_dir).call(args.tool, arguments)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
