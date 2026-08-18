#!/usr/bin/env python3
"""Build a small deterministic human-review queue from frozen 7B formal results."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _row_map(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate case_id: {case_id}")
        result[case_id] = row
    return result


def attach_trace_evidence(
    items: Iterable[Mapping[str, Any]],
    sft_traces: Iterable[Mapping[str, Any]],
    dpo_traces: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach the observations that the reviewed answers were conditioned on."""
    sft_trace_map = _row_map(sft_traces)
    dpo_trace_map = _row_map(dpo_traces)
    enriched: List[Dict[str, Any]] = []
    for source_item in items:
        item = copy.deepcopy(dict(source_item))
        case_id = str(item["case_id"])
        if case_id not in sft_trace_map or case_id not in dpo_trace_map:
            raise ValueError(f"missing rollout trace for review case: {case_id}")
        for label, trace_map in (("sft", sft_trace_map), ("dpo", dpo_trace_map)):
            result = dict(item[label])
            trace = trace_map[case_id]
            result["tool_observations"] = trace.get("tool_observations", [])
            result["termination_reason"] = trace.get("termination_reason", "")
            result["parse_errors"] = trace.get("parse_errors", [])
            item[label] = result
        item["review_evidence_version"] = "1.1"
        enriched.append(item)
    return enriched


def _stratified_take(
    ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    limit: int,
    seed: int,
) -> List[str]:
    groups: Dict[tuple[str, str], List[str]] = defaultdict(list)
    for case_id in ids:
        case = cases[case_id]
        groups[(str(case.get("tier", "unknown")), str(case.get("category", "unknown")))].append(case_id)
    generator = random.Random(seed)
    for values in groups.values():
        generator.shuffle(values)
    selected: List[str] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected


def select_review_items(
    cases: Iterable[Mapping[str, Any]],
    evaluator_cases: Iterable[Mapping[str, Any]],
    sft_rows: Iterable[Mapping[str, Any]],
    dpo_rows: Iterable[Mapping[str, Any]],
    *,
    count: int = 40,
    seed: int = 20260810,
) -> List[Dict[str, Any]]:
    case_map = _row_map(cases)
    evaluator_map = _row_map(evaluator_cases)
    sft_map = _row_map(sft_rows)
    dpo_map = _row_map(dpo_rows)
    expected_ids = set(case_map)
    for label, rows in (("evaluator", evaluator_map), ("sft", sft_map), ("dpo", dpo_map)):
        if set(rows) != expected_ids:
            raise ValueError(f"{label} rows do not contain exactly the fixed case set")
    if count <= 0 or count > len(expected_ids):
        raise ValueError("count must be between 1 and the number of cases")

    regressions = [case_id for case_id in expected_ids if sft_map[case_id]["passed"] and not dpo_map[case_id]["passed"]]
    recoveries = [case_id for case_id in expected_ids if not sft_map[case_id]["passed"] and dpo_map[case_id]["passed"]]
    weak_categories = {"expired", "identity", "not_delivered", "timeout"}
    weak_failures = [
        case_id
        for case_id in expected_ids
        if not dpo_map[case_id]["passed"] and str(case_map[case_id].get("category")) in weak_categories
    ]
    stable_success = [case_id for case_id in expected_ids if sft_map[case_id]["passed"] and dpo_map[case_id]["passed"]]

    regression_quota = min(20, len(regressions), count)
    recovery_quota = min(10, len(recoveries), max(0, count - regression_quota))
    stable_quota = min(5, len(stable_success), max(0, count - regression_quota - recovery_quota))
    weak_quota = min(
        len(weak_failures),
        max(0, count - regression_quota - recovery_quota - stable_quota),
    )
    quotas = {
        "sft_pass_dpo_fail": regression_quota,
        "sft_fail_dpo_pass": recovery_quota,
        "weak_category_dpo_fail": weak_quota,
        "stable_success_control": stable_quota,
    }
    pools = {
        "sft_pass_dpo_fail": regressions,
        "sft_fail_dpo_pass": recoveries,
        "weak_category_dpo_fail": weak_failures,
        "stable_success_control": stable_success,
    }
    selected_reasons: Dict[str, str] = {}
    for index, (reason, pool) in enumerate(pools.items()):
        remaining = [case_id for case_id in pool if case_id not in selected_reasons]
        for case_id in _stratified_take(remaining, case_map, quotas[reason], seed + index):
            selected_reasons[case_id] = reason
    if len(selected_reasons) < count:
        remaining = [case_id for case_id in expected_ids if case_id not in selected_reasons]
        for case_id in _stratified_take(remaining, case_map, count - len(selected_reasons), seed + 100):
            selected_reasons[case_id] = "stratified_fill"

    def result_view(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "passed": bool(row["passed"]),
            "tool_calls": row.get("actual_tool_calls", []),
            "final_answer": row.get("final_answer", ""),
            "errors": row.get("errors", []),
            "checks": row.get("checks", {}),
        }

    items = []
    for case_id, reason in selected_reasons.items():
        case = case_map[case_id]
        evaluator = evaluator_map[case_id]
        items.append(
            {
                "case_id": case_id,
                "selection_reason": reason,
                "tier": case.get("tier", "unknown"),
                "category": case.get("category", "unknown"),
                "messages": case.get("messages", case.get("conversation", [])),
                "expected_evidence": evaluator.get("expected", {}),
                "sft": result_view(sft_map[case_id]),
                "dpo": result_view(dpo_map[case_id]),
                "human_review": {
                    "evaluator_correct": "",
                    "dpo_preferred": "",
                    "factual_grounding": "",
                    "safe_action": "",
                    "notes": "",
                },
            }
        )
    return sorted(items, key=lambda item: (item["selection_reason"], item["tier"], item["category"], item["case_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-queue", type=Path, help="Enrich an already selected review queue")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--evaluator-cases", type=Path)
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--dpo", type=Path)
    parser.add_argument("--sft-traces", type=Path)
    parser.add_argument("--dpo-traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    generation_paths = (args.cases, args.evaluator_cases, args.sft, args.dpo)
    if args.input_queue:
        if any(generation_paths):
            parser.error("--input-queue cannot be combined with --cases/--evaluator-cases/--sft/--dpo")
        items = load_jsonl(args.input_queue)
    else:
        if not all(generation_paths):
            parser.error("queue generation requires --cases, --evaluator-cases, --sft, and --dpo")
        items = select_review_items(
            load_jsonl(args.cases),
            load_jsonl(args.evaluator_cases),
            load_jsonl(args.sft),
            load_jsonl(args.dpo),
            count=args.count,
            seed=args.seed,
        )
    if bool(args.sft_traces) != bool(args.dpo_traces):
        parser.error("--sft-traces and --dpo-traces must be provided together")
    if args.sft_traces and args.dpo_traces:
        items = attach_trace_evidence(items, load_jsonl(args.sft_traces), load_jsonl(args.dpo_traces))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for item in items:
            output_file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": "1.1" if args.sft_traces else "1.0",
        "count": len(items),
        "seed": args.seed,
        "selection_reason_counts": dict(sorted(Counter(item["selection_reason"] for item in items).items())),
        "tier_counts": dict(sorted(Counter(str(item["tier"]) for item in items).items())),
        "category_counts": dict(sorted(Counter(str(item["category"]) for item in items).items())),
        "review_fields": ["evaluator_correct", "dpo_preferred", "factual_grounding", "safe_action", "notes"],
        "trace_evidence_attached": bool(args.sft_traces),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
