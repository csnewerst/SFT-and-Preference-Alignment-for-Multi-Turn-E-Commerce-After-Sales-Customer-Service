#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from run_ecommerce_rollout import load_jsonl


IDENTIFIER_RE = re.compile(r"\b(?:EC|ASR)-[A-Z0-9-]+\b")
STATE_CLAIM_PATTERNS = (
    (re.compile(r"(?:已退款|已经退款|退款成功)"), "refunded"),
    (re.compile(r"(?<!未)已提交"), "submitted"),
    (re.compile(r"(?<!未)已审核"), "approved"),
    (re.compile(r"(?:已签收|完成交付|已完成交付)"), "delivered"),
    (re.compile(r"(?:已发货|运输中)"), "shipped"),
    (re.compile(r"(?:已支付|完成支付|已完成支付)"), "paid"),
)
NEGATED_STATE_PREFIX_RE = re.compile(
    r"(?:尚未|并未|没有|并没有|不是|并非|不代表|无法确认|不能确认|尚无法确认|未能确认)"
    r"[^，。；！？\n]{0,6}$"
)
AUTO_RESOLUTION_INELIGIBLE_CATEGORIES = frozenset({"identity", "timeout"})


def _tool_calls(trace: Mapping[str, Any]) -> List[Dict[str, Any]]:
    calls = []
    for call in trace.get("parsed_tool_calls", []):
        if isinstance(call, Mapping) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), Mapping):
            calls.append({"name": call["name"], "arguments": dict(call["arguments"])})
    return calls


def _acceptable_sequences(expected: Mapping[str, Any]) -> List[List[Dict[str, Any]]]:
    sequences = expected.get("acceptable_tool_sequences")
    if isinstance(sequences, list):
        return sequences
    calls = expected.get("tool_calls")
    return [calls] if isinstance(calls, list) else [[]]


def _lookup(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _fact_issues(case: Mapping[str, Any], trace: Mapping[str, Any]) -> List[str]:
    final_answer = str(trace.get("final_answer", ""))
    observations = [
        item.get("observation")
        for item in trace.get("tool_observations", [])
        if isinstance(item, Mapping) and isinstance(item.get("observation"), Mapping)
    ]
    evidence_text = json.dumps(observations, ensure_ascii=False, sort_keys=True)
    input_text = json.dumps(case.get("messages", []), ensure_ascii=False, sort_keys=True)
    allowed_identifiers = set(IDENTIFIER_RE.findall(input_text + " " + evidence_text))
    issues = [
        f"unsupported_identifier:{identifier}"
        for identifier in sorted(set(IDENTIFIER_RE.findall(final_answer)) - allowed_identifiers)
    ]
    evidence_values = set(re.findall(r'"(?:status|payment_status|fulfillment_status)":\s*"([^"]+)"', evidence_text))
    stated_values = set(
        re.findall(r'["\'](?:status|payment_status|fulfillment_status)["\']\s*:\s*["\']([^"\']+)["\']', final_answer)
    )
    for canonical in stated_values:
        if canonical not in evidence_values:
            issues.append(f"unsupported_state:{canonical}")
    for pattern, canonical in STATE_CLAIM_PATTERNS:
        for match in pattern.finditer(final_answer):
            prefix = final_answer[max(0, match.start() - 12) : match.start()]
            if NEGATED_STATE_PREFIX_RE.search(prefix):
                continue
            if canonical not in evidence_values:
                issues.append(f"unsupported_state:{canonical}")
                break
    return issues


def evaluate_trace(case: Mapping[str, Any], trace: Mapping[str, Any] | None) -> Dict[str, Any]:
    expected = case["expected"]
    auto_resolution_eligible = bool(
        case.get(
            "auto_resolution_eligible",
            str(case.get("category", "")) not in AUTO_RESOLUTION_INELIGIBLE_CATEGORIES,
        )
    )
    if trace is None:
        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "passed": False,
            "checks": {},
            "errors": ["missing_trace"],
            "auto_resolution_eligible": auto_resolution_eligible,
            "auto_resolved": False,
        }

    calls = _tool_calls(trace)
    names = [call["name"] for call in calls]
    acceptable = _acceptable_sequences(expected)
    tool_selection_valid = any(names == [call["name"] for call in sequence] for sequence in acceptable)
    arguments_valid = any(calls == sequence for sequence in acceptable)
    forbidden_tool_absent = not any(name in set(expected.get("must_not_call", [])) for name in names)
    actual_codes = [
        item.get("observation_code")
        for item in trace.get("tool_observations", [])
        if isinstance(item, Mapping)
    ]
    observation_outcomes_valid = actual_codes == expected.get("observation_codes", [])
    final_answer = str(trace.get("final_answer", "")).strip()
    final_answer_present = bool(final_answer)
    parse_success = not trace.get("parse_errors")
    within_step_limit = trace.get("termination_reason") != "max_steps"

    term_groups = expected.get("required_answer_term_groups", [])
    answer_requirements_met = all(any(term in final_answer for term in group) for group in term_groups)
    state_assertions_met = True
    for assertion in expected.get("state_assertions", []):
        try:
            actual = _lookup(trace.get("environment_state_after", {}), assertion["path"])
        except (KeyError, TypeError):
            state_assertions_met = False
            break
        if actual != assertion.get("equals"):
            state_assertions_met = False
            break
    fact_issues = _fact_issues(case, trace)
    facts_faithful = not fact_issues

    checks = {
        "parse_success": parse_success,
        "tool_selection_valid": tool_selection_valid,
        "arguments_valid": arguments_valid,
        "forbidden_tool_absent": forbidden_tool_absent,
        "observation_outcomes_valid": observation_outcomes_valid,
        "final_answer_present": final_answer_present,
        "answer_requirements_met": answer_requirements_met,
        "state_assertions_met": state_assertions_met,
        "facts_faithful": facts_faithful,
        "within_step_limit": within_step_limit,
    }
    error_names = {
        "parse_success": "parse_error",
        "tool_selection_valid": "wrong_tool",
        "arguments_valid": "wrong_argument",
        "forbidden_tool_absent": "forbidden_tool",
        "observation_outcomes_valid": "unexpected_tool_outcome",
        "final_answer_present": "missing_final_answer",
        "answer_requirements_met": "incomplete_resolution",
        "state_assertions_met": "goal_state_not_reached",
        "facts_faithful": "hallucinated_state",
        "within_step_limit": "max_steps",
    }
    errors = [error_names[name] for name, passed in checks.items() if not passed]
    errors.extend(fact_issues)
    passed = all(checks.values())
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "passed": passed,
        "auto_resolution_eligible": auto_resolution_eligible,
        "auto_resolved": auto_resolution_eligible and passed,
        "checks": checks,
        "errors": sorted(set(errors)),
        "actual_tool_calls": calls,
        "actual_observation_codes": actual_codes,
        "final_answer": final_answer,
    }


def evaluate_traces(
    cases: Iterable[Mapping[str, Any]],
    traces: Iterable[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    case_list = list(cases)
    trace_by_id: Dict[str, Mapping[str, Any]] = {}
    for trace in traces:
        case_id = trace.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("every trace must contain a string case_id")
        if case_id in trace_by_id:
            raise ValueError(f"duplicate trace case_id: {case_id}")
        trace_by_id[case_id] = trace
    known_ids = {str(case["case_id"]) for case in case_list}
    unknown = sorted(set(trace_by_id) - known_ids)
    if unknown:
        raise ValueError(f"traces contain unknown case_ids: {unknown[:5]}")

    results = [evaluate_trace(case, trace_by_id.get(str(case["case_id"]))) for case in case_list]
    total = len(results)
    check_names = sorted({name for result in results for name in result["checks"]})
    metrics = {
        name: sum(bool(result["checks"].get(name)) for result in results) / total if total else 0.0
        for name in check_names
    }
    metrics["task_success_rate"] = sum(result["passed"] for result in results) / total if total else 0.0
    eligible_results = [result for result in results if result["auto_resolution_eligible"]]
    metrics["eligible_auto_resolution_rate"] = (
        sum(result["auto_resolved"] for result in eligible_results) / len(eligible_results)
        if eligible_results
        else 0.0
    )
    error_counts: Dict[str, int] = {}
    category_counts: Dict[str, Dict[str, int]] = {}
    for result in results:
        for error in result["errors"]:
            error_counts[error] = error_counts.get(error, 0) + 1
        counts = category_counts.setdefault(result["category"], {"passed": 0, "total": 0})
        counts["passed"] += int(result["passed"])
        counts["total"] += 1
    return results, {
        "case_count": total,
        "trace_count": len(trace_by_id),
        "eligible_auto_resolution_count": len(eligible_results),
        "metrics": metrics,
        "error_counts": dict(sorted(error_counts.items())),
        "category_metrics": {
            name: {**counts, "task_success_rate": counts["passed"] / counts["total"]}
            for name, counts in sorted(category_counts.items())
        },
    }


def write_evaluation(cases_path: Path, traces_path: Path, output_dir: Path) -> Dict[str, Any]:
    results, summary = evaluate_traces(load_jsonl(cases_path), load_jsonl(traces_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8", newline="\n") as output_file:
        for result in results:
            output_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate executable ecommerce rollout traces.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_evaluation(args.cases, args.traces, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
