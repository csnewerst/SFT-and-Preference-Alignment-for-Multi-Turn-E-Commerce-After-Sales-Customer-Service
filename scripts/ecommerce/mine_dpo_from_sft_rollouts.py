#!/usr/bin/env python3
"""Mine one authentic first-divergence DPO pair from each failed frozen-SFT rollout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from build_dpo_v1_4_quality import behavior_bucket
from evaluate_rollout_v1 import evaluate_trace


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _by_case_id(rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} row is missing case_id")
        if case_id in result:
            raise ValueError(f"duplicate {label} case_id: {case_id}")
        result[case_id] = row
    return result


def _format_call(call: Mapping[str, Any]) -> str:
    return f"Action: {call['name']}\nAction Input: {json.dumps(call['arguments'], ensure_ascii=False, sort_keys=True)}"


def _target_kind(target: str) -> str:
    return "action" if target.startswith("Action:") else "response"


def _observation_data(oracle: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    observations = oracle.get("oracle_observations", [])
    if index >= len(observations) or not isinstance(observations[index], Mapping):
        return {}
    data = observations[index].get("data", {})
    return data if isinstance(data, Mapping) else {}


def compose_oracle_answer(case: Mapping[str, Any], oracle: Mapping[str, Any]) -> str:
    """Compose a concise response using only deterministic simulator fields."""
    category = str(case.get("category", ""))
    first = _observation_data(oracle, 0)
    second = _observation_data(oracle, 1)
    third = _observation_data(oracle, 2)
    if category == "status":
        return f"查询结果显示，订单 {first['order_id']} 已签收，当前没有售后申请。"
    if category == "policy":
        return (
            f"订单 {first['order_id']} 已签收；破损问题符合换货政策，需要提供破损凭证。"
            "如果您确认办理，我可以继续创建换货申请。"
        )
    if category == "create":
        return (
            f"换货申请已提交，申请编号为 {third['request_id']}，当前状态为 submitted。"
            "后续请按要求补充破损凭证。"
        )
    if category == "missing_order":
        return "请提供脱敏订单号；在取得订单号前，我无法查询订单、核验政策或创建售后申请。"
    if category == "duplicate":
        request = first["after_sales_request"]
        return (
            f"已查询到订单 {first['order_id']} 存在售后申请 {request['request_id']}，当前状态为已提交。"
            "为避免重复处理，我不会再次创建申请。"
        )
    if category == "identity":
        return (
            f"订单 {first['order_id']} 已签收，破损问题符合换货政策，但当前仍需要身份核验。"
            "完成核验后才能提交换货申请。"
        )
    if category == "not_delivered":
        return (
            f"订单 {first['order_id']} 仍在运输中、尚未签收，当前不满足破损换货条件。"
            "请签收并确认实际商品问题后再申请。"
        )
    if category == "timeout":
        return "订单系统本次查询超时，请稍后重试；当前无法确认订单状态，我不会据此创建售后申请。"
    if category == "expired":
        return (
            f"订单 {first['order_id']} 已签收 12 天，超过 7 天无理由退货时效，当前不符合无理由退货政策。"
        )
    if category == "anti_hallucination":
        return (
            f"系统查询显示订单 {first['order_id']} 仍在运输中，查询结果中没有任何退款记录；"
            "应以实际系统结果为准。"
        )
    raise ValueError(f"unsupported rollout category for oracle response: {category}")


def verified_oracle_answer(case: Mapping[str, Any], oracle: Mapping[str, Any]) -> str:
    answer = compose_oracle_answer(case, oracle)
    expected = oracle["expected"]
    calls = expected["acceptable_tool_sequences"][0]
    observations = oracle.get("oracle_observations", [])
    codes = expected.get("observation_codes", [])
    trace = {
        "case_id": case["case_id"],
        "parsed_tool_calls": [{"step": index, **copy.deepcopy(call)} for index, call in enumerate(calls)],
        "tool_observations": [
            {
                "step": index,
                "call": copy.deepcopy(call),
                "observation": copy.deepcopy(observation),
                "observation_code": codes[index],
            }
            for index, (call, observation) in enumerate(zip(calls, observations))
        ],
        "final_answer": answer,
        "parse_errors": [],
        "environment_state_after": oracle.get("environment_expected_state", {}),
        "termination_reason": "final_answer",
    }
    result = evaluate_trace({**dict(case), "expected": expected}, trace)
    if not result["passed"]:
        raise ValueError(f"composed oracle answer failed replay for {case['case_id']}: {result['errors']}")
    return answer


def _conversations(messages: Sequence[Mapping[str, Any]]) -> list[Dict[str, str]]:
    converted = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "user":
            converted.append({"from": "human", "value": content})
        elif role == "assistant":
            converted.append({"from": "gpt", "value": content})
        else:
            raise ValueError(f"unsupported case message role: {role}")
    return converted


def _append_matched_turn(conversations: list[Dict[str, str]], turn: Mapping[str, Any]) -> None:
    conversations.append({"from": "gpt", "value": str(turn["model_output"])})
    observations = turn.get("observations", [])
    payload: Any = observations[0] if len(observations) == 1 else observations
    conversations.append(
        {
            "from": "observation",
            "value": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        }
    )


def mine_pair(
    case: Mapping[str, Any],
    oracle: Mapping[str, Any],
    trace: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
    rollout_run_id: str,
    policy_role: str = "frozen_sft",
    pair_schema_version: str = "1.4",
) -> tuple[Dict[str, Any] | None, str]:
    if policy_role not in {"frozen_sft", "current_dpo"}:
        raise ValueError(f"unsupported rollout policy role: {policy_role}")
    if evaluation.get("passed"):
        return None, "rollout_passed"
    expected_sequences = oracle["expected"].get("acceptable_tool_sequences", [])
    if not expected_sequences:
        raise ValueError(f"oracle {case['case_id']} has no acceptable tool sequence")
    expected_calls = expected_sequences[0]
    expected_index = 0
    prompt = _conversations(case.get("messages", []))
    chosen = rejected = level = primary_error = ""
    rejected_source = ""
    divergence_step = -1

    for turn in trace.get("turns", []):
        kind = str(turn.get("parsed_kind", ""))
        if kind == "parse_error":
            return None, "parse_error_not_near_miss"
        if kind == "final":
            rejected = str(turn.get("model_output", "")).strip()
            rejected_source = f"authentic_{policy_role}_greedy_rollout_final_raw"
            divergence_step = int(turn.get("step", -1))
            if expected_index < len(expected_calls):
                chosen = _format_call(expected_calls[expected_index])
                level = "decision"
                primary_error = "premature_stop"
            else:
                chosen = verified_oracle_answer(case, oracle)
                level = "response"
                errors = [str(value) for value in evaluation.get("errors", [])]
                primary_error = errors[0] if errors else "incomplete_resolution"
            break
        if kind != "tool_calls":
            return None, "unsupported_turn_kind"
        actual_calls = turn.get("tool_calls", [])
        if not isinstance(actual_calls, list) or not actual_calls:
            return None, "empty_tool_call"
        expected_slice = expected_calls[expected_index : expected_index + len(actual_calls)]
        if actual_calls == expected_slice:
            expected_index += len(actual_calls)
            _append_matched_turn(prompt, turn)
            continue

        if len(actual_calls) != 1:
            return None, "multi_call_divergence_requires_alignment"
        rejected = _format_call(actual_calls[0])
        rejected_source = f"authentic_{policy_role}_greedy_rollout_action_parsed_canonical"
        divergence_step = int(turn.get("step", -1))
        if expected_index >= len(expected_calls):
            chosen = verified_oracle_answer(case, oracle)
            level = "decision"
            primary_error = "unnecessary_tool"
        else:
            chosen = _format_call(expected_calls[expected_index])
            level = "parameter" if actual_calls[0].get("name") == expected_calls[expected_index].get("name") else "decision"
            primary_error = "invalid_argument" if level == "parameter" else "wrong_action"
        break

    if not chosen or not rejected:
        return None, "no_actionable_first_divergence"
    chosen_kind = _target_kind(chosen)
    rejected_kind = _target_kind(rejected)
    source_ref = case.get("source_ref", {})
    pair = {
        "conversations": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "tools": copy.deepcopy(list(tools)),
        "metadata": {
            "schema_version": pair_schema_version,
            "sample_id": f"dpo-v{pair_schema_version}-rollout-{case['case_id']}-step{divergence_step}",
            "pair_id": f"dpo-v{pair_schema_version}-rollout-{case['case_id']}-step{divergence_step}",
            "parent_id": str(source_ref.get("parent_id", case["case_id"])),
            "source_id": str(source_ref.get("source_id", "")),
            "source_record_id": str(source_ref.get("source_record_id", "")),
            "rollout_case_id": str(case["case_id"]),
            "rollout_run_id": rollout_run_id,
            "rollout_category": str(case.get("category", "")),
            "rollout_tier": str(case.get("tier", "")),
            "target_turn_index": divergence_step,
            "preference_level": level,
            "primary_error": primary_error,
            "secondary_errors": sorted(str(value) for value in evaluation.get("errors", [])),
            "chosen_target_kind": chosen_kind,
            "rejected_target_kind": rejected_kind,
            "pair_source": f"{policy_role}_rollout_first_divergence",
            "rejected_source": rejected_source,
            "rollout_policy_role": policy_role,
            "chosen_verification": "deterministic_oracle_replay",
            "chosen_response_requires_review": chosen_kind == "response",
            "chosen_response_review_unit": "template" if chosen_kind == "response" else "none",
            "chosen_response_template_id": (
                f"{case.get('category', 'unknown')}-grounded-v1" if chosen_kind == "response" else ""
            ),
        },
    }
    pair["metadata"]["dpo_v1_4_bucket"] = behavior_bucket(pair)
    return pair, "selected"


def _split(pair: Mapping[str, Any], validation_fraction: float) -> str:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    parent_id = str(pair["metadata"]["parent_id"])
    value = int(hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    return "validation" if value < validation_fraction else "train"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in materialized:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "count": len(materialized),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_rollout_pairs(
    cases: Sequence[Mapping[str, Any]],
    oracles: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    output_root: Path,
    rollout_run_id: str,
    validation_fraction: float = 0.1,
    policy_role: str = "frozen_sft",
    pair_schema_version: str = "1.4",
) -> Dict[str, Any]:
    oracle_by_id = _by_case_id(oracles, "oracle")
    trace_by_id = _by_case_id(traces, "trace")
    evaluation_by_id = _by_case_id(evaluations, "evaluation")
    selected: Dict[str, list[Dict[str, Any]]] = {"train": [], "validation": []}
    reason_counts: Counter[str] = Counter()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id not in oracle_by_id or case_id not in trace_by_id or case_id not in evaluation_by_id:
            raise ValueError(f"missing aligned artifact for {case_id}")
        pair, reason = mine_pair(
            case,
            oracle_by_id[case_id],
            trace_by_id[case_id],
            evaluation_by_id[case_id],
            tools,
            rollout_run_id,
            policy_role,
            pair_schema_version,
        )
        reason_counts[reason] += 1
        if pair is not None:
            selected[_split(pair, validation_fraction)].append(pair)
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "status": "rollout_mined_candidates_require_hardness_scoring_and_response_review",
        "rollout_run_id": rollout_run_id,
        "rollout_policy_role": policy_role,
        "pair_schema_version": pair_schema_version,
        "input_case_count": len(cases),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "splits": {},
    }
    for split, rows in selected.items():
        rows.sort(key=lambda row: str(row["metadata"]["sample_id"]))
        artifact = _write_jsonl(output_root / split / "records.jsonl", rows)
        manifest["splits"][split] = {
            **artifact,
            "bucket_counts": dict(sorted(Counter(behavior_bucket(row) for row in rows).items())),
            "response_review_count": sum(bool(row["metadata"]["chosen_response_requires_review"]) for row in rows),
            "response_template_counts": dict(
                sorted(
                    Counter(
                        str(row["metadata"]["chosen_response_template_id"])
                        for row in rows
                        if row["metadata"]["chosen_response_template_id"]
                    ).items()
                )
            ),
            "primary_error_counts": dict(
                sorted(Counter(str(row["metadata"]["primary_error"]) for row in rows).items())
            ),
            "category_counts": dict(
                sorted(Counter(str(row["metadata"]["rollout_category"]) for row in rows).items())
            ),
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--oracles", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rollout-run-id", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--policy-role", choices=("frozen_sft", "current_dpo"), default="frozen_sft")
    parser.add_argument("--pair-schema-version", default="1.4")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite rollout pair output: {args.output_root}")
    tools_payload = json.loads(args.tools.read_text(encoding="utf-8"))
    manifest = build_rollout_pairs(
        load_jsonl(args.cases),
        load_jsonl(args.oracles),
        load_jsonl(args.traces),
        load_jsonl(args.evaluations),
        tools_payload["tools"],
        args.output_root,
        args.rollout_run_id,
        args.validation_fraction,
        args.policy_role,
        args.pair_schema_version,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
