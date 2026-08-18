#!/usr/bin/env python3
"""Analyze per-case success transitions from an SFT baseline to DPO candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


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


def _tool_sequence(row: Mapping[str, Any]) -> str:
    calls = row.get("actual_tool_calls", [])
    names = [str(call.get("name", "<missing>")) for call in calls if isinstance(call, Mapping)]
    return " -> ".join(names) if names else "<no_tool>"


def _tool_calls(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    calls = row.get("actual_tool_calls", [])
    return [
        {
            "name": str(call.get("name", "<missing>")),
            "arguments": call.get("arguments", {}),
        }
        for call in calls
        if isinstance(call, Mapping)
    ]


def _case_context(case: Mapping[str, Any]) -> Dict[str, Any]:
    source_ref = case.get("source_ref", {})
    if not isinstance(source_ref, Mapping):
        source_ref = {}
    return {
        "tier": str(case.get("tier", "unknown")),
        "category": str(case.get("category", "unknown")),
        "parent_id": str(source_ref.get("parent_id", "")),
    }


def _transition_detail(
    case: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "case_id": str(case["case_id"]),
        **_case_context(case),
        "baseline": {
            "tool_calls": _tool_calls(baseline),
            "errors": [str(error) for error in baseline.get("errors", [])],
            "final_answer": str(baseline.get("final_answer", "")),
        },
        "candidate": {
            "tool_calls": _tool_calls(candidate),
            "errors": [str(error) for error in candidate.get("errors", [])],
            "final_answer": str(candidate.get("final_answer", "")),
        },
    }


def analyze_failures(
    cases: Iterable[Mapping[str, Any]],
    baseline_rows: Iterable[Mapping[str, Any]],
    candidate_runs: Sequence[Tuple[str, Iterable[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    case_list = list(cases)
    case_ids = [str(case["case_id"]) for case in case_list]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cases contain duplicate case_id values")
    case_map = {str(case["case_id"]): case for case in case_list}
    categories = {case_id: str(case.get("category", "unknown")) for case_id, case in case_map.items()}
    baseline = _row_map(baseline_rows)
    if set(baseline) != set(case_ids):
        raise ValueError("baseline does not contain exactly the fixed case set")

    baseline_passed = sum(bool(baseline[case_id]["passed"]) for case_id in case_ids)
    report: Dict[str, Any] = {
        "schema_version": "1.1",
        "case_count": len(case_ids),
        "baseline": {
            "passed": baseline_passed,
            "task_success_rate": baseline_passed / len(case_ids) if case_ids else 0.0,
        },
        "candidates": {},
    }
    for label, rows in candidate_runs:
        candidate = _row_map(rows)
        if set(candidate) != set(case_ids):
            raise ValueError(f"candidate {label} does not contain exactly the fixed case set")
        transitions = Counter()
        regression_errors: Counter[str] = Counter()
        regression_categories: Counter[str] = Counter()
        recovery_categories: Counter[str] = Counter()
        regression_sequences: Counter[str] = Counter()
        regression_ids = []
        recovery_ids = []
        regression_details = []
        recovery_details = []
        changed_sequences = 0
        for case_id in case_ids:
            before = bool(baseline[case_id]["passed"])
            after = bool(candidate[case_id]["passed"])
            transition = ("pass" if before else "fail") + "_to_" + ("pass" if after else "fail")
            transitions[transition] += 1
            if _tool_sequence(baseline[case_id]) != _tool_sequence(candidate[case_id]):
                changed_sequences += 1
            if before and not after:
                regression_ids.append(case_id)
                regression_details.append(
                    _transition_detail(case_map[case_id], baseline[case_id], candidate[case_id])
                )
                regression_categories[categories[case_id]] += 1
                regression_errors.update(str(error) for error in candidate[case_id].get("errors", []))
                regression_sequences[_tool_sequence(candidate[case_id])] += 1
            elif not before and after:
                recovery_ids.append(case_id)
                recovery_details.append(
                    _transition_detail(case_map[case_id], baseline[case_id], candidate[case_id])
                )
                recovery_categories[categories[case_id]] += 1

        candidate_passed = transitions["pass_to_pass"] + transitions["fail_to_pass"]
        report["candidates"][label] = {
            "passed": candidate_passed,
            "task_success_rate": candidate_passed / len(case_ids) if case_ids else 0.0,
            "paired_delta": (candidate_passed - baseline_passed) / len(case_ids) if case_ids else 0.0,
            "transitions": dict(sorted(transitions.items())),
            "tool_sequence_changed": changed_sequences,
            "regression_by_category": dict(sorted(regression_categories.items())),
            "recovery_by_category": dict(sorted(recovery_categories.items())),
            "regression_error_counts": dict(regression_errors.most_common()),
            "regression_tool_sequences": dict(regression_sequences.most_common(10)),
            "regression_case_ids": sorted(regression_ids),
            "recovery_case_ids": sorted(recovery_ids),
            "regression_details": sorted(regression_details, key=lambda item: item["case_id"]),
            "recovery_details": sorted(recovery_details, key=lambda item: item["case_id"]),
        }
    return report


def _parse_run(spec: str) -> Tuple[str, List[Dict[str, Any]]]:
    label, separator, path = spec.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"invalid --run value: {spec}")
    return label, load_jsonl(Path(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=per_sample.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_failures(
        load_jsonl(args.cases),
        load_jsonl(args.baseline),
        [_parse_run(spec) for spec in args.run],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(report["candidates"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
