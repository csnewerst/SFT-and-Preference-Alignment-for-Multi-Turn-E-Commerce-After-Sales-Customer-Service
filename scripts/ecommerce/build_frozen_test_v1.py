#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tool_simulator import DEFAULT_CONFIG_DIR, EcommerceToolSimulator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "ecommerce" / "frozen" / "v1"
GENERATOR_VERSION = "1.0.0"


def _observation_code(observation: Dict[str, Any]) -> str:
    if not observation["ok"]:
        return observation["error"]["code"]
    return observation["data"].get("reason_code", "OK")


def _allowed_claims(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    claims: Dict[str, Any] = {}
    for observation in observations:
        if not observation["ok"] or observation["tool"] != "query_order_status":
            continue
        data = observation["data"]
        for key in ("order_id", "payment_status", "fulfillment_status", "days_since_delivery"):
            claims[key] = data.get(key)
        if data.get("after_sales_request"):
            claims["after_sales_status"] = data["after_sales_request"]["status"]
    return claims


def _case(
    case_id: str,
    category: str,
    user_text: str,
    calls: List[Dict[str, Any]],
    terminal_status: str,
    must_not_call: List[str] | None = None,
) -> Dict[str, Any]:
    simulator = EcommerceToolSimulator.from_config_dir(DEFAULT_CONFIG_DIR)
    observations = [simulator.call(call["name"], call["arguments"]) for call in calls]
    return {
        "case_id": case_id,
        "category": category,
        "messages": [{"role": "user", "content": user_text}],
        "expected": {
            "tool_calls": calls,
            "observation_codes": [_observation_code(item) for item in observations],
            "terminal_status": terminal_status,
            "must_not_call": must_not_call or [],
            "allowed_claims": _allowed_claims(observations),
        },
    }


def build_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    for index in range(30):
        variant = index % 3
        if variant == 0:
            cases.append(
                _case(
                    f"FZ1-NORMAL-{index:03d}",
                    "normal",
                    f"帮我查一下订单 EC-1001 的状态，表达方式编号 {index}。",
                    [{"name": "query_order_status", "arguments": {"order_id": "EC-1001"}}],
                    "state_reported",
                )
            )
        elif variant == 1:
            cases.append(
                _case(
                    f"FZ1-NORMAL-{index:03d}",
                    "normal",
                    f"订单 EC-1001 的商品破损了，能否换货？编号 {index}。",
                    [
                        {"name": "query_order_status", "arguments": {"order_id": "EC-1001"}},
                        {
                            "name": "check_return_policy",
                            "arguments": {"order_id": "EC-1001", "issue_type": "damaged"},
                        },
                    ],
                    "policy_explained",
                )
            )
        else:
            cases.append(
                _case(
                    f"FZ1-NORMAL-{index:03d}",
                    "normal",
                    f"订单 EC-1001 的商品破损，我确认申请换货，编号 {index}。",
                    [
                        {"name": "query_order_status", "arguments": {"order_id": "EC-1001"}},
                        {
                            "name": "check_return_policy",
                            "arguments": {"order_id": "EC-1001", "issue_type": "damaged"},
                        },
                        {
                            "name": "create_after_sales_request",
                            "arguments": {
                                "order_id": "EC-1001",
                                "request_type": "exchange",
                                "reason": "damaged",
                            },
                        },
                    ],
                    "resolved",
                )
            )

    for index in range(15):
        if index % 2 == 0:
            cases.append(
                _case(
                    f"FZ1-MISSING-{index:03d}",
                    "missing_argument",
                    f"我想查一下订单，但没找到订单号，编号 {index}。",
                    [],
                    "collect_order_id",
                    must_not_call=list(EcommerceToolSimulator.EXPECTED_TOOLS),
                )
            )
        else:
            cases.append(
                _case(
                    f"FZ1-MISSING-{index:03d}",
                    "missing_argument",
                    f"订单 EC-1001 有问题，帮我处理，编号 {index}。",
                    [{"name": "query_order_status", "arguments": {"order_id": "EC-1001"}}],
                    "clarify_issue",
                    must_not_call=["check_return_policy", "create_after_sales_request"],
                )
            )

    for index in range(15):
        cases.append(
            _case(
                f"FZ1-TOOL-{index:03d}",
                "tool_difficulty",
                f"订单 EC-1005 的售后申请现在是什么状态？不要重复创建。编号 {index}。",
                [{"name": "query_order_status", "arguments": {"order_id": "EC-1005"}}],
                "existing_request_found",
                must_not_call=["create_after_sales_request"],
            )
        )

    for index in range(15):
        variant = index % 3
        if variant == 0:
            order_id, issue_type, terminal = "EC-1002", "no_reason", "policy_explained"
        elif variant == 1:
            order_id, issue_type, terminal = "EC-1003", "damaged", "policy_explained"
        else:
            order_id, issue_type, terminal = "EC-1004", "damaged", "identity_verification_required"
        calls = [
            {"name": "query_order_status", "arguments": {"order_id": order_id}},
            {"name": "check_return_policy", "arguments": {"order_id": order_id, "issue_type": issue_type}},
        ]
        if variant == 2:
            calls.append(
                {
                    "name": "create_after_sales_request",
                    "arguments": {"order_id": order_id, "request_type": "exchange", "reason": issue_type},
                }
            )
        cases.append(
            _case(
                f"FZ1-POLICY-{index:03d}",
                "policy_boundary",
                f"请按规则处理订单 {order_id} 的 {issue_type} 问题，编号 {index}。",
                calls,
                terminal,
                must_not_call=[] if variant == 2 else ["create_after_sales_request"],
            )
        )

    for index in range(10):
        cases.append(
            _case(
                f"FZ1-FAILURE-{index:03d}",
                "tool_failure",
                f"查询订单 EC-FAIL-001，若系统超时不要猜状态，编号 {index}。",
                [{"name": "query_order_status", "arguments": {"order_id": "EC-FAIL-001"}}],
                "tool_retry_pending",
                must_not_call=["create_after_sales_request"],
            )
        )

    for index in range(10):
        cases.append(
            _case(
                f"FZ1-HALLUCINATION-{index:03d}",
                "anti_hallucination",
                f"订单 EC-UNKNOWN 肯定已经退款了吧？请确认，编号 {index}。",
                [{"name": "query_order_status", "arguments": {"order_id": "EC-UNKNOWN"}}],
                "collect_order_id",
                must_not_call=["create_after_sales_request"],
            )
        )

    for index in range(5):
        cases.append(
            _case(
                f"FZ1-CONFLICT-{index:03d}",
                "state_conflict",
                f"之前说订单 EC-1003 已签收，但刚才又说在运输中，请重新查，编号 {index}。",
                [{"name": "query_order_status", "arguments": {"order_id": "EC-1003"}}],
                "state_reported",
                must_not_call=["create_after_sales_request"],
            )
        )

    if len(cases) != 100 or len({case["case_id"] for case in cases}) != 100:
        raise AssertionError("frozen test v1 must contain 100 unique cases")
    return cases


def write_frozen_test(output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    cases = build_cases()
    with cases_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    content = cases_path.read_bytes()
    manifest = {
        "dataset_version": "frozen-test-v1",
        "generator_version": GENERATOR_VERSION,
        "protocol_date": "2026-08-09",
        "case_count": len(cases),
        "category_counts": dict(sorted(Counter(case["category"] for case in cases).items())),
        "files": [
            {
                "path": cases_path.name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic ecommerce frozen test v1.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = write_frozen_test(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
