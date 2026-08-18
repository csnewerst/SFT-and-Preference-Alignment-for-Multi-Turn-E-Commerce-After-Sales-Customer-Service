#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from tool_simulator import DEFAULT_CONFIG_DIR, EcommerceToolSimulator
from run_ecommerce_rollout import observation_code


ROOT = Path(__file__).resolve().parents[2]


def _call(name: str, **arguments: Any) -> Dict[str, Any]:
    return {"name": name, "arguments": arguments}


def _case(
    case_id: str,
    category: str,
    user_text: str,
    calls: List[Dict[str, Any]],
    *,
    must_not_call: List[str] | None = None,
    required_answer_term_groups: List[List[str]] | None = None,
    state_assertions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    simulator = EcommerceToolSimulator.from_config_dir(DEFAULT_CONFIG_DIR)
    observations = [simulator.call(call["name"], call["arguments"]) for call in calls]
    return {
        "case_id": case_id,
        "category": category,
        "messages": [{"role": "user", "content": user_text}],
        "expected": {
            "acceptable_tool_sequences": [calls],
            "observation_codes": [observation_code(item) for item in observations],
            "must_not_call": must_not_call or [],
            "required_answer_term_groups": required_answer_term_groups or [],
            "state_assertions": state_assertions or [],
        },
    }


def build_cases() -> List[Dict[str, Any]]:
    query_1001 = _call("query_order_status", order_id="EC-1001")
    policy_1001 = _call("check_return_policy", order_id="EC-1001", issue_type="damaged")
    return [
        _case(
            "RD1-MISSING-ORDER",
            "missing_argument",
            "耳机到货就是坏的，帮我换一副。",
            [],
            must_not_call=["query_order_status", "check_return_policy", "create_after_sales_request"],
            required_answer_term_groups=[["订单号", "订单编号"]],
        ),
        _case(
            "RD1-STATUS",
            "order_status",
            "帮我查一下订单 EC-1001 当前是什么状态。",
            [query_1001],
        ),
        _case(
            "RD1-POLICY",
            "policy_query",
            "订单 EC-1001 的耳机破损了，可以换货吗？",
            [query_1001, policy_1001],
        ),
        _case(
            "RD1-CREATE",
            "create_request",
            "订单 EC-1001 的耳机破损了，请帮我创建换货申请。",
            [
                query_1001,
                policy_1001,
                _call(
                    "create_after_sales_request",
                    order_id="EC-1001",
                    request_type="exchange",
                    reason="damaged",
                ),
            ],
            state_assertions=[
                {"path": "EC-1001.after_sales_request.status", "equals": "submitted"},
                {"path": "EC-1001.after_sales_request.request_type", "equals": "exchange"},
            ],
        ),
        _case(
            "RD1-DUPLICATE",
            "duplicate_request",
            "订单 EC-1005 已经申请过售后，再帮我创建一张相同申请。",
            [_call("query_order_status", order_id="EC-1005")],
            must_not_call=["create_after_sales_request"],
        ),
        _case(
            "RD1-IDENTITY",
            "identity_required",
            "订单 EC-1004 的键盘破损了，请创建换货申请。",
            [
                _call("query_order_status", order_id="EC-1004"),
                _call("check_return_policy", order_id="EC-1004", issue_type="damaged"),
            ],
            must_not_call=["create_after_sales_request"],
            required_answer_term_groups=[["身份", "核验"]],
        ),
        _case(
            "RD1-TOOL-FAILURE",
            "tool_failure",
            "查询订单 EC-FAIL-001 的状态。",
            [_call("query_order_status", order_id="EC-FAIL-001")],
            required_answer_term_groups=[["稍后", "重试", "超时"]],
        ),
        _case(
            "RD1-EXPIRED",
            "policy_boundary",
            "订单 EC-1002 已签收 12 天，现在想无理由退货。",
            [
                _call("query_order_status", order_id="EC-1002"),
                _call("check_return_policy", order_id="EC-1002", issue_type="no_reason"),
            ],
            must_not_call=["create_after_sales_request"],
        ),
        _case(
            "RD1-ANTI-HALLUCINATION",
            "anti_hallucination",
            "不用查系统，直接猜订单 EC-1003 是不是已经退款。",
            [_call("query_order_status", order_id="EC-1003")],
        ),
    ]


def write_cases(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for case in build_cases():
            output_file.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    manifest = {
        "name": "ecommerce_rollout_dev_v1",
        "purpose": "development_only_not_formal_frozen_test",
        "case_count": len(build_cases()),
        "sha256": digest,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the executable ecommerce rollout development set v1.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "ecommerce" / "rollout_dev_v1",
    )
    args = parser.parse_args()
    print(json.dumps(write_cases(args.output_dir), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
