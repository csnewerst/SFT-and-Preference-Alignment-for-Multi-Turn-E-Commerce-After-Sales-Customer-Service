import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from build_7b_formal_review_queue import attach_trace_evidence, select_review_items


def test_review_queue_prioritizes_regressions_and_is_deterministic():
    cases = [
        {"case_id": f"c{i}", "tier": "iid" if i % 2 else "challenge", "category": "timeout" if i < 4 else "create", "messages": []}
        for i in range(12)
    ]
    evaluator = [{**case, "expected": {"acceptable_tool_sequences": []}} for case in cases]
    sft = []
    dpo = []
    for i, case in enumerate(cases):
        sft_passed = i in {0, 1, 4, 5, 8, 9}
        dpo_passed = i in {2, 3, 4, 5, 6, 7, 8, 9}
        base = {"case_id": case["case_id"], "actual_tool_calls": [], "final_answer": "ok", "errors": [], "checks": {}}
        sft.append({**base, "passed": sft_passed})
        dpo.append({**base, "passed": dpo_passed})

    first = select_review_items(cases, evaluator, sft, dpo, count=8, seed=7)
    second = select_review_items(cases, evaluator, sft, dpo, count=8, seed=7)

    assert first == second
    assert len(first) == 8
    reasons = {item["case_id"]: item["selection_reason"] for item in first}
    assert reasons["c0"] == "sft_pass_dpo_fail"
    assert reasons["c1"] == "sft_pass_dpo_fail"
    assert {"c2", "c3"}.issubset(reasons)
    assert all(item["expected_evidence"] for item in first)
    assert all(set(item["human_review"]) == {"evaluator_correct", "dpo_preferred", "factual_grounding", "safe_action", "notes"} for item in first)


def test_review_queue_attaches_model_visible_tool_observations():
    item = {
        "case_id": "c1",
        "sft": {"passed": True},
        "dpo": {"passed": False},
    }
    sft_trace = {
        "case_id": "c1",
        "tool_observations": [{"observation_code": "OK", "observation": {"data": {"payment_status": "paid"}}}],
        "termination_reason": "final_answer",
        "parse_errors": [],
    }
    dpo_trace = {
        "case_id": "c1",
        "tool_observations": [{"observation_code": "UPSTREAM_TIMEOUT", "observation": {"ok": False}}],
        "termination_reason": "final_answer",
        "parse_errors": ["bad tool call"],
    }

    enriched = attach_trace_evidence([item], [sft_trace], [dpo_trace])

    assert enriched[0]["review_evidence_version"] == "1.1"
    assert enriched[0]["sft"]["tool_observations"][0]["observation_code"] == "OK"
    assert enriched[0]["dpo"]["tool_observations"][0]["observation_code"] == "UPSTREAM_TIMEOUT"
    assert enriched[0]["dpo"]["parse_errors"] == ["bad tool call"]
    assert "tool_observations" not in item["sft"]
