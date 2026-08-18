import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from build_rollout_dev_v1 import build_cases, write_cases
from evaluate_rollout_v1 import evaluate_trace, evaluate_traces
from run_ecommerce_rollout import parse_assistant_output, run_case, summarize_generation_metrics
from tool_simulator import EcommerceToolSimulator


class ScriptedGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages_seen = []

    def __call__(self, messages):
        self.messages_seen.append(copy.deepcopy(messages))
        return self.outputs.pop(0)


def _case(case_id="CASE-1"):
    return {
        "case_id": case_id,
        "category": "test",
        "messages": [{"role": "user", "content": "查询订单 EC-1001。"}],
    }


def test_parser_distinguishes_tools_final_answers_and_malformed_actions():
    parsed = parse_assistant_output(
        'Action: query_order_status\nAction Input: {"order_id": "EC-1001"}'
    )

    assert parsed == {
        "kind": "tool_calls",
        "tool_calls": [{"name": "query_order_status", "arguments": {"order_id": "EC-1001"}}],
    }
    assert parse_assistant_output("订单当前已签收。")["kind"] == "final"
    assert parse_assistant_output("Action: query_order_status\nAction Input: not-json")["kind"] == "parse_error"


def test_generation_metrics_separate_cold_and_steady_state_latency():
    summary = summarize_generation_metrics(
        [
            {"latency_seconds": 3.0, "input_tokens": 10, "output_tokens": 6},
            {"latency_seconds": 1.0, "input_tokens": 20, "output_tokens": 4},
            {"latency_seconds": 2.0, "input_tokens": 30, "output_tokens": 8},
        ],
        case_count=2,
        model_load_seconds=5.0,
        end_to_end_seconds=8.0,
        peak_allocated_mib=100.0,
        peak_reserved_mib=120.0,
    )

    assert summary["first_call_latency_seconds"] == 3.0
    assert summary["generation_latency_seconds"]["steady_state_p50"] == 1.5
    assert summary["tokens"]["output_tokens_per_second"] == 3.0
    assert summary["cases_per_second"] == 0.25


def test_rollout_executes_tool_and_records_environment_owned_observation():
    generator = ScriptedGenerator(
        [
            'Action: query_order_status\nAction Input: {"order_id": "EC-1001"}',
            "订单已经签收。",
        ]
    )
    trace = run_case(
        _case(),
        generator,
        simulator=EcommerceToolSimulator.from_config_dir(ROOT / "configs" / "ecommerce"),
    )

    assert trace["termination_reason"] == "final_answer"
    assert trace["parsed_tool_calls"][0]["name"] == "query_order_status"
    observation = trace["tool_observations"][0]["observation"]
    assert observation["data"]["order_id"] == "EC-1001"
    assert observation["data"]["identity_verification_status"] == "verified"
    assert "Observation:" in generator.messages_seen[1][-1]["content"]
    assert trace["environment_state_before"] == trace["environment_state_after"]


def test_rollout_records_state_mutation_and_stops_at_max_steps():
    create = (
        'Action: create_after_sales_request\nAction Input: '
        '{"order_id":"EC-1001","request_type":"exchange","reason":"damaged"}'
    )
    trace = run_case(
        _case(),
        ScriptedGenerator([create, create]),
        simulator=EcommerceToolSimulator.from_config_dir(ROOT / "configs" / "ecommerce"),
        max_steps=2,
    )

    assert trace["termination_reason"] == "max_steps"
    assert trace["environment_state_after"]["EC-1001"]["after_sales_request"]["status"] == "submitted"
    assert [item["observation_code"] for item in trace["tool_observations"]] == ["OK", "DUPLICATE_REQUEST"]


def test_rollout_dev_set_covers_all_tools_and_is_marked_development_only(tmp_path):
    cases = build_cases()
    names = {
        call["name"]
        for case in cases
        for sequence in case["expected"]["acceptable_tool_sequences"]
        for call in sequence
    }
    manifest = write_cases(tmp_path)

    assert len(cases) == 9
    assert names == {
        "query_order_status",
        "check_return_policy",
        "create_after_sales_request",
    }
    assert manifest["purpose"] == "development_only_not_formal_frozen_test"
    assert (tmp_path / "cases.jsonl").is_file()


def test_trace_evaluator_uses_executed_results_and_detects_unsupported_claims():
    case = next(item for item in build_cases() if item["case_id"] == "RD1-STATUS")
    good = run_case(
        case,
        ScriptedGenerator(
            [
                'Action: query_order_status\nAction Input: {"order_id":"EC-1001"}',
                "订单 EC-1001 已签收。",
            ]
        ),
        simulator=EcommerceToolSimulator.from_config_dir(ROOT / "configs" / "ecommerce"),
    )
    bad = copy.deepcopy(good)
    bad["final_answer"] = "订单 EC-1001 已退款，售后单 ASR-FAKE-001 已审核。"

    assert evaluate_trace(case, good)["passed"] is True
    bad_result = evaluate_trace(case, bad)
    assert bad_result["passed"] is False
    assert "hallucinated_state" in bad_result["errors"]
    assert "unsupported_identifier:ASR-FAKE-001" in bad_result["errors"]

    guessed_refund = copy.deepcopy(good)
    guessed_refund["final_answer"] = "我猜订单 EC-1001 已经退款了。"
    guessed_result = evaluate_trace(case, guessed_refund)
    assert guessed_result["passed"] is False
    assert "unsupported_state:refunded" in guessed_result["errors"]

    negated_refund = copy.deepcopy(good)
    negated_refund["final_answer"] = "查询结果中没有证据表明已经退款，当前只能确认订单已签收。"
    negated_result = evaluate_trace(case, negated_refund)
    assert "unsupported_state:refunded" not in negated_result["errors"]
    assert negated_result["checks"]["facts_faithful"] is True

    uncertain_refund = copy.deepcopy(good)
    uncertain_refund["final_answer"] = "目前无法确认是否已退款，能够确认的是订单已签收。"
    uncertain_result = evaluate_trace(case, uncertain_refund)
    assert "unsupported_state:refunded" not in uncertain_result["errors"]

    explicit_refund = copy.deepcopy(good)
    explicit_refund["final_answer"] = "订单不是运输中，而是已经退款。"
    explicit_result = evaluate_trace(case, explicit_refund)
    assert "unsupported_state:refunded" in explicit_result["errors"]

    json_state = copy.deepcopy(good)
    json_state["final_answer"] = '{"order_id":"EC-1001","status":"pending"}'
    json_result = evaluate_trace(case, json_state)
    assert json_result["passed"] is False
    assert "unsupported_state:pending" in json_result["errors"]


def test_missing_order_case_requires_question_without_any_tool_call():
    case = next(item for item in build_cases() if item["case_id"] == "RD1-MISSING-ORDER")
    trace = run_case(
        case,
        ScriptedGenerator(["请提供订单号，我再为你查询并处理。"]),
        simulator=EcommerceToolSimulator.from_config_dir(ROOT / "configs" / "ecommerce"),
    )
    results, summary = evaluate_traces([case], [trace])

    assert results[0]["passed"] is True
    assert summary["metrics"]["task_success_rate"] == 1.0
