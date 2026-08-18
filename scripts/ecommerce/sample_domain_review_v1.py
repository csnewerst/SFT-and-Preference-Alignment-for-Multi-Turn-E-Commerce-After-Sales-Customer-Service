#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
SAMPLER_VERSION = "2.0.0"

HIGH_RISK_SCENARIOS = {
    "duplicate_request",
    "identity_required",
    "no_reason_expired",
    "tool_timeout",
}
HIGH_RISK_ERRORS = {
    "hallucinated_state",
    "policy_violation",
    "premature_stop",
}
ALLOWED_TOOLS = {
    "query_order_status",
    "check_return_policy",
    "create_after_sales_request",
}


def _load_task_rows(task_root: Path, task: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(task_root.rglob("*.jsonl")):
        split = path.parent.name
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                rows.append({"task": task, "split": split, "row": row})
    return rows


def _bucket_key(item: Mapping[str, Any]) -> Tuple[str, ...]:
    metadata = item["row"]["metadata"]
    key = (metadata["source_id"], metadata["scenario"], item["split"])
    if item["task"] == "dpo":
        key += (
            metadata.get("preference_level", "unknown"),
            metadata.get("primary_error", "unknown"),
        )
    return key


def stratified_sample(items: Sequence[Dict[str, Any]], limit: int, salt: str) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[_bucket_key(item)].append(item)
    for rows in buckets.values():
        rows.sort(
            key=lambda item: hashlib.sha256(
                f"{salt}:{item['row']['metadata']['sample_id']}".encode("utf-8")
            ).hexdigest()
        )
    selected = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return selected


def _risk_score(item: Mapping[str, Any]) -> int:
    metadata = item["row"]["metadata"]
    score = 0
    if metadata.get("scenario") in HIGH_RISK_SCENARIOS:
        score += 3
    if metadata.get("primary_error") in HIGH_RISK_ERRORS:
        score += 2
    if metadata.get("preference_level") in {"decision", "parameter"}:
        score += 2
    if item.get("split") == "validation":
        score += 1
    return score


def select_human_review_items(
    items: Sequence[Dict[str, Any]],
    limit: int,
    salt: str,
    risk_fraction: float = 0.6,
) -> List[Dict[str, Any]]:
    if limit < 0:
        raise ValueError("review limit must be non-negative")
    if not 0 <= risk_fraction <= 1:
        raise ValueError("risk_fraction must be between 0 and 1")
    risk_candidates = [item for item in items if _risk_score(item) >= 3]
    risk_limit = min(len(risk_candidates), round(limit * risk_fraction))
    selected = stratified_sample(risk_candidates, risk_limit, f"{salt}-risk")
    selected_ids = {item["row"]["metadata"]["sample_id"] for item in selected}
    controls = [
        item
        for item in items
        if item["row"]["metadata"]["sample_id"] not in selected_ids and _risk_score(item) < 3
    ]
    selected.extend(stratified_sample(controls, limit - len(selected), f"{salt}-control"))
    if len(selected) < limit:
        selected_ids = {item["row"]["metadata"]["sample_id"] for item in selected}
        fallback = [
            item
            for item in items
            if item["row"]["metadata"]["sample_id"] not in selected_ids
        ]
        selected.extend(stratified_sample(fallback, limit - len(selected), f"{salt}-fallback"))
    return selected


def _machine_precheck(row: Mapping[str, Any], task: str) -> Tuple[str, str]:
    issues = []
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        issues.append("missing_conversation")
        conversations = []
    function_calls = 0
    observations = 0
    for message in conversations:
        if not isinstance(message, Mapping):
            issues.append("invalid_message")
            continue
        role = message.get("from")
        value = message.get("value")
        if not isinstance(value, str) or not value.strip():
            issues.append("empty_message")
        if role == "function_call":
            function_calls += 1
            try:
                call = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                issues.append("invalid_function_call_json")
                continue
            if call.get("name") not in ALLOWED_TOOLS or not isinstance(call.get("arguments"), dict):
                issues.append("invalid_function_call")
        elif role == "observation":
            observations += 1
    if observations != function_calls:
        issues.append("unpaired_tool_observation")
    if task == "dpo" and row.get("chosen") == row.get("rejected"):
        issues.append("identical_preference_pair")
    if issues:
        return "不通过", ",".join(sorted(set(issues)))
    return "通过", "结构、工具定义、参数 JSON 与 observation 配对检查通过；业务状态来自版本化模拟器"


def _conversation_text(conversations: Iterable[Mapping[str, Any]]) -> str:
    role_names = {
        "system": "系统",
        "human": "用户",
        "user": "用户",
        "gpt": "客服",
        "assistant": "客服",
        "function_call": "工具调用",
        "observation": "工具结果",
    }
    return "\n".join(
        f"[{role_names.get(str(message.get('from')), str(message.get('from')))}] {message.get('value', '')}"
        for message in conversations
    )


def build_review_pack(
    domain_root: Path,
    output_dir: Path,
    sft_count: int = 60,
    dpo_count: int = 40,
    risk_fraction: float = 0.6,
) -> Dict[str, Any]:
    sft_items = _load_task_rows(domain_root / "sft", "sft")
    dpo_items = _load_task_rows(domain_root / "dpo", "dpo")
    selected = select_human_review_items(sft_items, sft_count, "sft-review-v2", risk_fraction)
    selected += select_human_review_items(dpo_items, dpo_count, "dpo-review-v2", risk_fraction)
    review_rows = []
    for index, item in enumerate(selected, start=1):
        row = item["row"]
        metadata = row["metadata"]
        machine_status, machine_evidence = _machine_precheck(row, item["task"])
        if machine_status != "通过":
            raise ValueError(f"machine precheck failed for {metadata['sample_id']}: {machine_evidence}")
        risk_score = _risk_score(item)
        review_rows.append(
            {
                "review_id": f"R{index:03d}",
                "task": item["task"],
                "split": item["split"],
                "sample_id": metadata["sample_id"],
                "parent_id": metadata["parent_id"],
                "source_id": metadata["source_id"],
                "scenario": metadata["scenario"],
                "intent": metadata["intent"],
                "preference_level": metadata.get("preference_level", ""),
                "primary_error": metadata.get("primary_error", ""),
                "selection_type": "高风险复核" if risk_score >= 3 else "随机分层控制",
                "risk_score": risk_score,
                "conversation": _conversation_text(row["conversations"]),
                "chosen": row.get("chosen", ""),
                "rejected": row.get("rejected", ""),
                "machine_gate_status": machine_status,
                "machine_gate_evidence": machine_evidence,
                "natural_expression": "",
                "fact_grounded": "",
                "chosen_clearly_preferred": "",
                "reviewer": "",
                "notes": "",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "review_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in review_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "sampler_version": SAMPLER_VERSION,
        "requested": {"sft": sft_count, "dpo": dpo_count},
        "risk_fraction": risk_fraction,
        "selected": {
            "sft": sum(row["task"] == "sft" for row in review_rows),
            "dpo": sum(row["task"] == "dpo" for row in review_rows),
        },
        "selected_strata": {
            "source": dict(
                sorted(
                    (source, sum(row["source_id"] == source for row in review_rows))
                    for source in {row["source_id"] for row in review_rows}
                )
            ),
            "scenario": dict(
                sorted(
                    (scenario, sum(row["scenario"] == scenario for row in review_rows))
                    for scenario in {row["scenario"] for row in review_rows}
                )
            ),
            "preference_level": dict(
                sorted(
                    (level, sum(row["preference_level"] == level for row in review_rows))
                    for level in {row["preference_level"] for row in review_rows if row["preference_level"]}
                )
            ),
        },
        "selection_type_counts": dict(
            sorted(
                (selection_type, sum(row["selection_type"] == selection_type for row in review_rows))
                for selection_type in {row["selection_type"] for row in review_rows}
            )
        ),
        "machine_precheck_passed": all(row["machine_gate_status"] == "通过" for row in review_rows),
        "human_review_fields": ["natural_expression", "fact_grounded", "chosen_clearly_preferred"],
        "review_rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a stratified human-review pack for domain pilot v1.")
    parser.add_argument("--domain-root", type=Path, default=ROOT / "data" / "ecommerce" / "domain_pilot_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "ecommerce" / "reviews" / "domain_pilot_v1")
    parser.add_argument("--sft-count", type=int, default=60)
    parser.add_argument("--dpo-count", type=int, default=40)
    parser.add_argument("--risk-fraction", type=float, default=0.6)
    args = parser.parse_args()
    manifest = build_review_pack(
        args.domain_root,
        args.output_dir,
        args.sft_count,
        args.dpo_count,
        args.risk_fraction,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
