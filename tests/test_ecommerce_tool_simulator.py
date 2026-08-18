import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from tool_simulator import EcommerceToolSimulator


@pytest.fixture()
def simulator():
    return EcommerceToolSimulator.from_config_dir(ROOT / "configs" / "ecommerce")


def test_v1_defines_exactly_three_tools(simulator):
    assert set(simulator.tool_schemas) == {
        "query_order_status",
        "check_return_policy",
        "create_after_sales_request",
    }


def test_query_order_status_returns_only_configured_state(simulator):
    result = simulator.call("query_order_status", {"order_id": "EC-1001"})

    assert result["ok"] is True
    assert result["data"]["fulfillment_status"] == "delivered"
    assert result["data"]["after_sales_request"] is None
    assert result["data"]["identity_verification_status"] == "verified"
    assert "identity_verified" not in result["data"]


def test_query_exposes_actionable_identity_status_without_internal_boolean(simulator):
    result = simulator.call("query_order_status", {"order_id": "EC-1004"})

    assert result["ok"] is True
    assert result["data"]["identity_verification_status"] == "required"
    assert "identity_verified" not in result["data"]


def test_unknown_order_does_not_fabricate_state(simulator):
    result = simulator.call("query_order_status", {"order_id": "EC-UNKNOWN"})

    assert result["ok"] is False
    assert result["error"]["code"] == "ORDER_NOT_FOUND"
    assert "data" not in result


@pytest.mark.parametrize(
    ("arguments", "message_fragment"),
    [
        ({}, "order_id"),
        ({"order_id": "1001"}, "格式"),
        ({"order_id": "EC-1001", "extra": "x"}, "未定义参数"),
        ({"order_id": 1001}, "字符串"),
    ],
)
def test_argument_validation_rejects_invalid_calls(simulator, arguments, message_fragment):
    result = simulator.call("query_order_status", arguments)

    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_ARGUMENTS"
    assert message_fragment in result["error"]["message"]


def test_policy_allows_in_window_no_reason_return(simulator):
    result = simulator.call(
        "check_return_policy",
        {"order_id": "EC-1001", "issue_type": "no_reason"},
    )

    assert result["data"]["eligible"] is True
    assert result["data"]["allowed_request_types"] == ["return_refund"]
    assert result["data"]["policy_version"] == "2026-08-v1"


def test_policy_rejects_expired_no_reason_return(simulator):
    result = simulator.call(
        "check_return_policy",
        {"order_id": "EC-1002", "issue_type": "no_reason"},
    )

    assert result["data"]["eligible"] is False
    assert result["data"]["reason_code"] == "RETURN_WINDOW_EXPIRED"


def test_policy_requires_delivery(simulator):
    result = simulator.call(
        "check_return_policy",
        {"order_id": "EC-1003", "issue_type": "damaged"},
    )

    assert result["data"]["eligible"] is False
    assert result["data"]["reason_code"] == "ORDER_NOT_DELIVERED"


def test_create_request_mutates_state_once(simulator):
    arguments = {
        "order_id": "EC-1001",
        "request_type": "exchange",
        "reason": "damaged",
    }
    created = simulator.call("create_after_sales_request", arguments)
    duplicate = simulator.call("create_after_sales_request", arguments)
    queried = simulator.call("query_order_status", {"order_id": "EC-1001"})

    assert created["ok"] is True
    assert created["data"]["request_id"] == "ASR-EC-1001-001"
    assert created["data"]["evidence_required"] is True
    assert duplicate["error"]["code"] == "DUPLICATE_REQUEST"
    assert queried["data"]["after_sales_request"]["status"] == "submitted"


def test_create_request_enforces_identity_and_request_type(simulator):
    identity_error = simulator.call(
        "create_after_sales_request",
        {"order_id": "EC-1004", "request_type": "exchange", "reason": "damaged"},
    )
    type_error = simulator.call(
        "create_after_sales_request",
        {"order_id": "EC-1001", "request_type": "refund_only", "reason": "no_reason"},
    )

    assert identity_error["error"]["code"] == "IDENTITY_NOT_VERIFIED"
    assert type_error["error"]["code"] == "REQUEST_TYPE_NOT_ALLOWED"


def test_injected_tool_failure_is_retryable(simulator):
    result = simulator.call("query_order_status", {"order_id": "EC-FAIL-001"})

    assert result["ok"] is False
    assert result["error"] == {
        "code": "UPSTREAM_TIMEOUT",
        "message": "订单服务暂时超时，请稍后重试。",
        "retryable": True,
    }


def test_instances_start_from_same_deterministic_state(simulator):
    simulator.call(
        "create_after_sales_request",
        {"order_id": "EC-1001", "request_type": "exchange", "reason": "damaged"},
    )
    fresh = EcommerceToolSimulator.from_config_dir(ROOT / "configs" / "ecommerce")

    assert fresh.snapshot()["EC-1001"]["after_sales_request"] is None


def test_unknown_tool_is_rejected(simulator):
    result = simulator.call("transfer_human", {"order_id": "EC-1001"})

    assert result["ok"] is False
    assert result["error"]["code"] == "UNKNOWN_TOOL"


def test_cli_emits_json_result():
    script = ROOT / "scripts" / "ecommerce" / "tool_simulator.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--tool",
            "query_order_status",
            "--arguments",
            json.dumps({"order_id": "EC-1001"}),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["order_id"] == "EC-1001"
