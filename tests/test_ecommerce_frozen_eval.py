import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from build_frozen_test_v1 import build_cases, write_frozen_test
from evaluate_frozen_test_v1 import evaluate_case, evaluate_predictions, load_jsonl, write_evaluation


def _perfect_prediction(case):
    expected = case["expected"]
    return {
        "case_id": case["case_id"],
        "tool_calls": expected["tool_calls"],
        "observation_codes": expected["observation_codes"],
        "terminal_status": expected["terminal_status"],
        "claimed_facts": expected["allowed_claims"],
    }


def test_frozen_v1_has_fixed_size_and_category_distribution():
    cases = build_cases()

    counts = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1

    assert len(cases) == 100
    assert len({case["case_id"] for case in cases}) == 100
    assert counts == {
        "normal": 30,
        "missing_argument": 15,
        "tool_difficulty": 15,
        "policy_boundary": 15,
        "tool_failure": 10,
        "anti_hallucination": 10,
        "state_conflict": 5,
    }


def test_frozen_v1_generation_is_byte_deterministic(tmp_path):
    first = write_frozen_test(tmp_path / "first")
    second = write_frozen_test(tmp_path / "second")
    first_bytes = (tmp_path / "first" / "cases.jsonl").read_bytes()
    second_bytes = (tmp_path / "second" / "cases.jsonl").read_bytes()

    assert first_bytes == second_bytes
    assert first["files"][0]["sha256"] == second["files"][0]["sha256"]
    assert first["files"][0]["sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert len(load_jsonl(tmp_path / "first" / "cases.jsonl")) == 100


def test_perfect_structured_predictions_pass_all_rules():
    cases = build_cases()
    predictions = [_perfect_prediction(case) for case in cases]

    results, summary = evaluate_predictions(cases, predictions)

    assert all(result["passed"] for result in results)
    assert summary["metrics"]["task_success_rate"] == 1.0
    assert summary["error_counts"] == {}


def test_rule_evaluator_reports_behavior_errors():
    case = build_cases()[0]
    bad_prediction = {
        "case_id": case["case_id"],
        "tool_calls": [
            {
                "name": "create_after_sales_request",
                "arguments": {
                    "order_id": "EC-1001",
                    "request_type": "refund_only",
                    "reason": "no_reason",
                },
            }
        ],
        "observation_codes": [],
        "terminal_status": "resolved",
        "claimed_facts": {"fulfillment_status": "refunded"},
    }

    result = evaluate_case(case, bad_prediction)

    assert result["passed"] is False
    assert set(result["errors"]) == {
        "wrong_tool",
        "missing_argument",
        "observation_misread",
        "incomplete_resolution",
        "hallucinated_state",
    }


def test_evaluation_writes_summary_and_per_sample_results(tmp_path):
    frozen_dir = tmp_path / "frozen"
    write_frozen_test(frozen_dir)
    cases = load_jsonl(frozen_dir / "cases.jsonl")
    predictions_path = tmp_path / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for case in cases:
            output.write(json.dumps(_perfect_prediction(case), ensure_ascii=False) + "\n")

    summary = write_evaluation(frozen_dir / "cases.jsonl", predictions_path, tmp_path / "evaluation")

    assert summary["case_count"] == 100
    assert summary["metrics"]["task_success_rate"] == 1.0
    assert (tmp_path / "evaluation" / "summary.json").is_file()
    assert len(load_jsonl(tmp_path / "evaluation" / "per_sample.jsonl")) == 100
