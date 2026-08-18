#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _valid_tool_calls(tool_calls: Any) -> bool:
    return isinstance(tool_calls, list) and all(
        isinstance(call, dict)
        and isinstance(call.get("name"), str)
        and isinstance(call.get("arguments"), dict)
        for call in tool_calls
    )


def evaluate_case(case: Dict[str, Any], prediction: Dict[str, Any] | None) -> Dict[str, Any]:
    expected = case["expected"]
    errors: List[str] = []
    if prediction is None:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "passed": False,
            "checks": {
                "tool_json_valid": False,
                "tool_selection_exact": False,
                "arguments_exact": False,
                "forbidden_tool_absent": False,
                "observation_codes_exact": False,
                "terminal_status_exact": False,
                "claims_faithful": False,
            },
            "errors": ["missing_prediction"],
        }

    tool_calls = prediction.get("tool_calls")
    tool_json_valid = _valid_tool_calls(tool_calls)
    if not tool_json_valid:
        errors.append("wrong_tool")
        tool_calls = []

    expected_calls = expected["tool_calls"]
    predicted_names = [call["name"] for call in tool_calls]
    expected_names = [call["name"] for call in expected_calls]
    tool_selection_exact = predicted_names == expected_names
    if not tool_selection_exact and "wrong_tool" not in errors:
        errors.append("wrong_tool")

    arguments_exact = tool_json_valid and len(tool_calls) == len(expected_calls) and all(
        predicted["arguments"] == wanted["arguments"]
        for predicted, wanted in zip(tool_calls, expected_calls)
    )
    if not arguments_exact:
        errors.append("missing_argument")

    forbidden = set(expected.get("must_not_call", []))
    forbidden_tool_absent = not any(name in forbidden for name in predicted_names)
    if not forbidden_tool_absent:
        errors.append("policy_violation")

    observation_codes_exact = prediction.get("observation_codes", []) == expected["observation_codes"]
    if not observation_codes_exact:
        errors.append("observation_misread")

    terminal_status_exact = prediction.get("terminal_status") == expected["terminal_status"]
    if not terminal_status_exact:
        errors.append("incomplete_resolution")

    claimed_facts = prediction.get("claimed_facts", {})
    allowed_claims = expected.get("allowed_claims", {})
    claims_faithful = isinstance(claimed_facts, dict) and all(
        key in allowed_claims and allowed_claims[key] == value for key, value in claimed_facts.items()
    )
    if not claims_faithful:
        errors.append("hallucinated_state")

    checks = {
        "tool_json_valid": tool_json_valid,
        "tool_selection_exact": tool_selection_exact,
        "arguments_exact": arguments_exact,
        "forbidden_tool_absent": forbidden_tool_absent,
        "observation_codes_exact": observation_codes_exact,
        "terminal_status_exact": terminal_status_exact,
        "claims_faithful": claims_faithful,
    }
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "passed": all(checks.values()),
        "checks": checks,
        "errors": sorted(set(errors)),
    }


def evaluate_predictions(
    cases: Iterable[Dict[str, Any]],
    predictions: Iterable[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    case_list = list(cases)
    prediction_by_id: Dict[str, Dict[str, Any]] = {}
    for prediction in predictions:
        case_id = prediction.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("every prediction must contain a string case_id")
        if case_id in prediction_by_id:
            raise ValueError(f"duplicate prediction case_id: {case_id}")
        prediction_by_id[case_id] = prediction

    known_ids = {case["case_id"] for case in case_list}
    unknown_ids = sorted(set(prediction_by_id) - known_ids)
    if unknown_ids:
        raise ValueError(f"predictions contain unknown case_ids: {unknown_ids[:5]}")

    results = [evaluate_case(case, prediction_by_id.get(case["case_id"])) for case in case_list]
    total = len(results)
    check_names = list(results[0]["checks"]) if results else []
    metrics = {
        name: sum(result["checks"][name] for result in results) / total if total else 0.0
        for name in check_names
    }
    metrics["task_success_rate"] = sum(result["passed"] for result in results) / total if total else 0.0

    error_counts: Dict[str, int] = {}
    category_success: Dict[str, Dict[str, int]] = {}
    for result in results:
        for error in result["errors"]:
            error_counts[error] = error_counts.get(error, 0) + 1
        category = category_success.setdefault(result["category"], {"passed": 0, "total": 0})
        category["total"] += 1
        category["passed"] += int(result["passed"])

    summary = {
        "case_count": total,
        "prediction_count": len(prediction_by_id),
        "metrics": metrics,
        "error_counts": dict(sorted(error_counts.items())),
        "category_metrics": {
            name: {
                **counts,
                "task_success_rate": counts["passed"] / counts["total"],
            }
            for name, counts in sorted(category_success.items())
        },
    }
    return results, summary


def write_evaluation(
    cases_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    results, summary = evaluate_predictions(load_jsonl(cases_path), load_jsonl(predictions_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule-evaluate ecommerce frozen test v1 predictions.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_evaluation(args.cases, args.predictions, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
