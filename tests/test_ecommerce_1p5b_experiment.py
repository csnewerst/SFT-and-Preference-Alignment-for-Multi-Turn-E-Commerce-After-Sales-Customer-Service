import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from build_1p5b_dpo_variants import build_variants
from analyze_1p5b_dpo_failures import analyze_failures
from build_dpo_v1_4_quality import behavior_bucket, quotas, select_quality_rows
from audit_dpo_v1_4_hardness import audit_dataset, percentile, summarize_rows
from score_dpo_pair_hardness import completion_token_ids
from shard_rollout_cases import merge_traces, prepare_shards
from mine_dpo_from_sft_rollouts import compose_oracle_answer, mine_pair, verified_oracle_answer
from build_rollout_prefreeze_v1 import _scenario_oracle
from evaluate_rollout_v1 import evaluate_traces
from audit_1p5b_token_lengths import _encoded_length, summarize
from capture_experiment_manifest import _files
from prepare_1p5b_eval_split import stratified_case_ids, write_split
from summarize_1p5b_runs import hardware_summary
from compare_1p5b_screen_runs import bootstrap_mean, compare_runs
from validate_formal_test_v2 import validate as validate_formal_test


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prefreeze_screen_gate_split_is_deterministic_and_stratified(tmp_path):
    cases = []
    for index in range(80):
        tier = "iid" if index < 40 else "compositional" if index < 60 else "challenge"
        category = f"scenario-{index % 4}"
        cases.append(
            {
                "case_id": f"case-{index}",
                "tier": tier,
                "category": category,
                "source_ref": {"parent_id": f"parent-{index}"},
            }
        )
    input_dir = tmp_path / "input"
    for name in ("cases", "private_oracle", "evaluator_cases", "source_manifest"):
        _write_jsonl(input_dir / f"{name}.jsonl", cases)

    first = write_split(input_dir, tmp_path / "first", screen_count=20, seed=20260809)
    second = write_split(input_dir, tmp_path / "second", screen_count=20, seed=20260809)

    assert first["splits"]["screen"]["case_count"] == 20
    assert first["splits"]["gate"]["case_count"] == 60
    assert first["splits"]["screen"]["tier_counts"] == {"challenge": 5, "compositional": 5, "iid": 10}
    assert (
        first["splits"]["screen"]["artifacts"]["cases"]["sha256"]
        == second["splits"]["screen"]["artifacts"]["cases"]["sha256"]
    )


def test_dpo_failure_analysis_tracks_regressions_and_recoveries():
    cases = [
        {"case_id": "a", "category": "create"},
        {"case_id": "b", "category": "status"},
        {"case_id": "c", "category": "status"},
    ]
    baseline = [
        {"case_id": "a", "passed": True, "actual_tool_calls": [{"name": "create"}]},
        {"case_id": "b", "passed": False, "errors": ["wrong_tool"], "actual_tool_calls": []},
        {"case_id": "c", "passed": True, "actual_tool_calls": [{"name": "query"}]},
    ]
    candidate = [
        {"case_id": "a", "passed": False, "errors": ["wrong_tool"], "actual_tool_calls": []},
        {"case_id": "b", "passed": True, "actual_tool_calls": [{"name": "query"}]},
        {"case_id": "c", "passed": True, "actual_tool_calls": [{"name": "query"}]},
    ]
    report = analyze_failures(cases, baseline, [("dpo", candidate)])
    result = report["candidates"]["dpo"]
    assert result["paired_delta"] == 0.0
    assert result["transitions"] == {"fail_to_pass": 1, "pass_to_fail": 1, "pass_to_pass": 1}
    assert result["regression_by_category"] == {"create": 1}
    assert result["recovery_by_category"] == {"status": 1}
    assert result["regression_error_counts"] == {"wrong_tool": 1}
    assert result["tool_sequence_changed"] == 2
    assert result["regression_details"] == [
        {
            "case_id": "a",
            "tier": "unknown",
            "category": "create",
            "parent_id": "",
            "baseline": {
                "tool_calls": [{"name": "create", "arguments": {}}],
                "errors": [],
                "final_answer": "",
            },
            "candidate": {
                "tool_calls": [],
                "errors": ["wrong_tool"],
                "final_answer": "",
            },
        }
    ]
    assert result["recovery_details"][0]["case_id"] == "b"


def _quality_pair(index, bucket, margin, parent=None):
    kinds = {
        "must_continue": ("decision", "action", "response"),
        "must_stop": ("decision", "response", "action"),
        "wrong_action": ("decision", "action", "action"),
        "parameter": ("parameter", "action", "action"),
        "response": ("response", "response", "response"),
    }
    level, chosen_kind, rejected_kind = kinds[bucket]
    return {
        "conversations": [{"from": "human", "value": f"case {index}"}],
        "chosen": f"chosen {index}",
        "rejected": f"rejected {index}",
        "metadata": {
            "sample_id": f"pair-{index}",
            "parent_id": parent or f"parent-{index}",
            "preference_level": level,
            "chosen_target_kind": chosen_kind,
            "rejected_target_kind": rejected_kind,
            "sft_hardness": {
                "chosen_mean_logp": -1.0,
                "rejected_mean_logp": -1.0 - margin,
                "mean_logp_margin": margin,
            },
        },
    }


def test_quality_dpo_selection_enforces_buckets_hardness_and_parent_cap():
    fractions = {
        "must_continue": 0.25,
        "must_stop": 0.10,
        "wrong_action": 0.10,
        "parameter": 0.25,
        "response": 0.30,
    }
    rows = []
    index = 0
    for bucket in fractions:
        for offset in range(10):
            rows.append(_quality_pair(index, bucket, margin=float(offset - 2)))
            index += 1
    selected, audit = select_quality_rows(rows, 20, fractions, max_pairs_per_parent=2)
    assert len(selected) == 20
    assert audit["bucket_counts"] == quotas(20, fractions)
    assert audit["max_pairs_for_one_parent"] == 1
    assert all("dpo_v1_4_bucket" in row["metadata"] for row in selected)
    assert {behavior_bucket(row) for row in selected} == set(fractions)


def test_quality_dpo_selection_allows_zero_quota_for_unmined_bucket():
    fractions = {
        "must_continue": 0.36,
        "must_stop": 0.36,
        "wrong_action": 0.0,
        "parameter": 0.04,
        "response": 0.24,
    }
    rows = []
    index = 0
    for bucket in ("must_continue", "must_stop", "parameter", "response"):
        for offset in range(40):
            rows.append(_quality_pair(index, bucket, margin=float(offset)))
            index += 1
    selected, audit = select_quality_rows(rows, 100, fractions, max_pairs_per_parent=1)
    assert audit["bucket_counts"] == {
        "must_continue": 36,
        "must_stop": 36,
        "parameter": 4,
        "response": 24,
    }
    assert all(behavior_bucket(row) != "wrong_action" for row in selected)


def test_quality_dpo_selection_rejects_unscored_pairs():
    row = _quality_pair(1, "response", margin=0.1)
    del row["metadata"]["sft_hardness"]
    try:
        select_quality_rows(
            [row],
            1,
            {"must_continue": 0.0, "must_stop": 0.0, "wrong_action": 0.0, "parameter": 0.0, "response": 1.0},
            max_pairs_per_parent=1,
        )
    except ValueError as exc:
        assert "sft_hardness" in str(exc)
    else:
        raise AssertionError("expected unscored DPO pair to be rejected")


def test_dpo_hardness_audit_reports_bucket_quantiles_and_thresholds(tmp_path):
    rows = []
    index = 0
    for bucket in ("must_continue", "must_stop", "wrong_action", "parameter", "response"):
        for margin in (-1.0, 0.0, 0.25, 1.0):
            row = _quality_pair(index, bucket, margin=margin)
            row["metadata"]["primary_error"] = f"error-{bucket}"
            rows.append(row)
            index += 1
    for split in ("train", "validation"):
        _write_jsonl(tmp_path / split / "records.jsonl", rows)

    report = audit_dataset(tmp_path)
    overall = report["splits"]["train"]["overall"]
    assert overall["count"] == 20
    assert overall["hard_candidate_counts"] == {
        "margin_le_0": 10,
        "margin_le_0p25": 15,
        "margin_le_0p5": 15,
    }
    assert overall["margin"]["p50"] == 0.125
    assert report["splits"]["train"]["by_bucket"]["response"]["count"] == 4
    assert percentile([0.0, 10.0], 0.25) == 2.5


def test_dpo_hardness_audit_rejects_missing_score():
    row = _quality_pair(1, "response", margin=0.1)
    del row["metadata"]["sft_hardness"]
    try:
        summarize_rows([row])
    except ValueError as exc:
        assert "mean_logp_margin" in str(exc)
    else:
        raise AssertionError("expected missing hardness score to be rejected")


def test_rollout_shards_are_deterministic_and_merge_losslessly(tmp_path):
    cases = [{"case_id": f"case-{index}", "value": index} for index in range(11)]
    cases_path = tmp_path / "cases.jsonl"
    _write_jsonl(cases_path, cases)
    manifest = prepare_shards(cases_path, tmp_path / "shards", shard_count=4)
    assert [item["count"] for item in manifest["shards"]] == [3, 3, 3, 2]

    trace_paths = []
    for shard in manifest["shards"]:
        shard_cases = [json.loads(line) for line in Path(shard["path"]).read_text(encoding="utf-8").splitlines()]
        trace_path = tmp_path / f"trace-{shard['index']}.jsonl"
        _write_jsonl(trace_path, [{"case_id": row["case_id"], "trace": row["value"]} for row in shard_cases])
        trace_paths.append(trace_path)
    output = tmp_path / "merged.jsonl"
    report = merge_traces(cases_path, list(reversed(trace_paths)), output)
    merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["trace_count"] == len(cases)
    assert [row["case_id"] for row in merged] == [row["case_id"] for row in cases]


def test_rollout_shard_merge_rejects_incomplete_coverage(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    traces_path = tmp_path / "traces.jsonl"
    _write_jsonl(cases_path, [{"case_id": "a"}, {"case_id": "b"}])
    _write_jsonl(traces_path, [{"case_id": "a"}])
    try:
        merge_traces(cases_path, [traces_path], tmp_path / "merged.jsonl")
    except ValueError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("expected incomplete trace coverage to be rejected")


def _mining_artifacts(actual_turn, expected_calls, *, passed=False, errors=None, category="create"):
    case = {
        "case_id": "RM1-IID-example",
        "category": category,
        "tier": "iid",
        "messages": [{"role": "user", "content": "请处理订单"}],
        "source_ref": {"parent_id": "parent-1", "source_id": "csds", "source_record_id": "record-1"},
    }
    oracle = {
        "expected": {
            "acceptable_tool_sequences": [expected_calls],
            "observation_codes": [],
            "required_answer_term_groups": [],
            "state_assertions": [],
        },
        "oracle_observations": [],
        "environment_expected_state": {},
    }
    trace = {"turns": [actual_turn]}
    evaluation = {"passed": passed, "errors": errors or ["wrong_tool"]}
    tools = [{"name": "query_order_status"}]
    return case, oracle, trace, evaluation, tools


def test_rollout_mining_captures_authentic_premature_stop():
    expected = [{"name": "query_order_status", "arguments": {"order_id": "EC-1"}}]
    artifacts = _mining_artifacts(
        {"step": 0, "parsed_kind": "final", "model_output": "无需查询，已经处理。"},
        expected,
        errors=["wrong_tool"],
    )
    pair, reason = mine_pair(*artifacts, rollout_run_id="sft-r4")
    assert reason == "selected"
    assert pair["metadata"]["dpo_v1_4_bucket"] == "must_continue"
    assert pair["chosen"].startswith("Action: query_order_status")
    assert pair["rejected"] == "无需查询，已经处理。"
    assert pair["metadata"]["chosen_response_requires_review"] is False


def test_rollout_mining_separates_wrong_arguments_from_wrong_action():
    expected = [{"name": "query_order_status", "arguments": {"order_id": "EC-1"}}]
    artifacts = _mining_artifacts(
        {
            "step": 0,
            "parsed_kind": "tool_calls",
            "model_output": 'Action: query_order_status\nAction Input: {"order_id": "EC-2"}',
            "tool_calls": [{"name": "query_order_status", "arguments": {"order_id": "EC-2"}}],
            "observations": [],
        },
        expected,
        errors=["wrong_argument"],
    )
    pair, reason = mine_pair(*artifacts, rollout_run_id="sft-r4")
    assert reason == "selected"
    assert pair["metadata"]["dpo_v1_4_bucket"] == "parameter"
    assert pair["metadata"]["primary_error"] == "invalid_argument"


def test_rollout_mining_marks_oracle_response_for_review():
    artifacts = _mining_artifacts(
        {"step": 0, "parsed_kind": "final", "model_output": "我猜已经完成。"},
        [],
        errors=["hallucinated_state"],
        category="missing_order",
    )
    pair, reason = mine_pair(*artifacts, rollout_run_id="sft-r4")
    assert reason == "selected"
    assert pair["metadata"]["dpo_v1_4_bucket"] == "response"
    assert pair["metadata"]["chosen_response_requires_review"] is True
    assert pair["metadata"]["rejected_source"] == "authentic_frozen_sft_greedy_rollout_final_raw"


def test_rollout_mining_records_current_dpo_negative_provenance():
    artifacts = _mining_artifacts(
        {"step": 0, "parsed_kind": "final", "model_output": "当前不能继续。"},
        [{"name": "query_order_status", "arguments": {"order_id": "EC-1"}}],
        errors=["premature_stop"],
    )

    pair, reason = mine_pair(
        *artifacts,
        rollout_run_id="dpo-v1p4-step5",
        policy_role="current_dpo",
        pair_schema_version="1.5",
    )

    assert reason == "selected"
    assert pair is not None
    assert pair["metadata"]["schema_version"] == "1.5"
    assert pair["metadata"]["pair_source"] == "current_dpo_rollout_first_divergence"
    assert pair["metadata"]["rejected_source"] == "authentic_current_dpo_greedy_rollout_final_raw"
    assert pair["metadata"]["rollout_policy_role"] == "current_dpo"


def test_rollout_mining_reconstructs_matched_prefix_before_divergence():
    query = {"name": "query_order_status", "arguments": {"order_id": "EC-1"}}
    policy = {"name": "check_return_policy", "arguments": {"order_id": "EC-1", "issue_type": "damaged"}}
    case, oracle, _, evaluation, tools = _mining_artifacts({}, [query, policy], errors=["wrong_tool"])
    trace = {
        "turns": [
            {
                "step": 0,
                "parsed_kind": "tool_calls",
                "model_output": 'Action: query_order_status\nAction Input: {"order_id": "EC-1"}',
                "tool_calls": [query],
                "observations": [{"ok": True, "data": {"status": "delivered"}}],
            },
            {"step": 1, "parsed_kind": "final", "model_output": "不用再查政策。"},
        ]
    }
    pair, reason = mine_pair(case, oracle, trace, evaluation, tools, rollout_run_id="sft-r4")
    assert reason == "selected"
    assert pair["metadata"]["dpo_v1_4_bucket"] == "must_continue"
    assert pair["chosen"].startswith("Action: check_return_policy")
    assert [turn["from"] for turn in pair["conversations"]] == ["human", "gpt", "observation"]


def test_rollout_mining_captures_unnecessary_tool_as_must_stop():
    artifacts = _mining_artifacts(
        {
            "step": 0,
            "parsed_kind": "tool_calls",
            "model_output": 'Action: query_order_status\nAction Input: {"order_id": "EC-1"}',
            "tool_calls": [{"name": "query_order_status", "arguments": {"order_id": "EC-1"}}],
            "observations": [],
        },
        [],
        errors=["forbidden_tool"],
        category="missing_order",
    )
    pair, reason = mine_pair(*artifacts, rollout_run_id="sft-r4")
    assert reason == "selected"
    assert pair["metadata"]["dpo_v1_4_bucket"] == "must_stop"
    assert pair["metadata"]["chosen_response_requires_review"] is True
    assert pair["rejected"].startswith("Action: query_order_status")


def test_all_oracle_response_templates_pass_trace_evaluator():
    for category in (
        "status",
        "policy",
        "create",
        "missing_order",
        "duplicate",
        "identity",
        "not_delivered",
        "timeout",
        "expired",
        "anti_hallucination",
    ):
        case = {"case_id": f"case-{category}", "category": category, "messages": []}
        oracle = _scenario_oracle(category)
        answer = verified_oracle_answer(case, oracle)
        assert answer == compose_oracle_answer(case, oracle)


def test_hardness_completion_tokens_append_one_eos():
    class Tokenizer:
        eos_token_id = 9

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert add_special_tokens is False
            return [1, 2, 9] if text == "closed" else [3, 4]

    assert completion_token_ids(Tokenizer(), {"chosen": "closed"}, "chosen") == [1, 2, 9]
    assert completion_token_ids(Tokenizer(), {"chosen": "open"}, "chosen") == [3, 4, 9]


def test_prefreeze_split_rejects_parent_leakage():
    cases = [
        {"case_id": "a", "tier": "iid", "category": "x", "source_ref": {"parent_id": "same"}},
        {"case_id": "b", "tier": "iid", "category": "x", "source_ref": {"parent_id": "same"}},
    ]
    try:
        stratified_case_ids(cases, screen_count=1, seed=42)
    except ValueError as exc:
        assert "parent_id" in str(exc)
    else:
        raise AssertionError("expected duplicate parents to be rejected")


def _dpo_row(index, level):
    return {
        "conversations": [{"from": "human", "value": f"问题 {index}"}],
        "chosen": f"正确 {index}",
        "rejected": f"错误 {index}",
        "metadata": {"sample_id": f"sample-{index}", "preference_level": level},
    }


def test_dpo_variants_control_data_count_and_preserve_full_set(tmp_path):
    input_root = tmp_path / "input"
    for split, offset in (("train", 0), ("validation", 100)):
        rows = []
        rows.extend(_dpo_row(offset + index, "decision") for index in range(8))
        rows.extend(_dpo_row(offset + 20 + index, "parameter") for index in range(5))
        rows.extend(_dpo_row(offset + 40 + index, "response") for index in range(7))
        _write_jsonl(input_root / split / "records.jsonl", rows)

    manifest = build_variants(input_root, tmp_path / "output", seed=42)

    for split in ("train", "validation"):
        assert manifest["variants"]["response_only_matched"][split]["count"] == 7
        assert manifest["variants"]["multigranularity_matched"][split]["count"] == 7
        assert manifest["variants"]["multigranularity_full"][split]["count"] == 20
        assert manifest["variants"]["response_only_matched"][split]["level_counts"] == {"response": 7}


def test_experiment_plan_and_seed_wiring_are_explicit(tmp_path):
    plan = json.loads((ROOT / "configs" / "ecommerce" / "experiments_1p5b_v1.json").read_text(encoding="utf-8"))
    assert plan["status"] == "dpo_v1p4_direction_gate_passed_checkpoint10_selected"
    assert plan["sft"]["rank_candidates"] == [4, 16, 64]
    assert plan["dpo"]["beta_candidates"] == [0.05, 0.1, 0.3]
    assert plan["distributed"]["deepspeed"] is False
    assert plan["sft"]["model_max_length"] == 1024
    assert plan["dpo"]["max_source_length"] == 1024
    assert plan["dpo"]["max_target_length"] == 128
    assert plan["dpo"]["warmup_schedule"]["steps_by_training_budget"] == {"99": 3, "278": 9}
    assert plan["dpo"]["training_budget"]["multigranularity_full"] == 278
    assert plan["dpo"]["reference_policy"] == "frozen_copy_of_loaded_sft_adapter"

    plan_7b = json.loads((ROOT / "configs" / "ecommerce" / "experiments_7b_v1.json").read_text(encoding="utf-8"))
    assert plan_7b["model"]["precision"] == "bfloat16"
    assert plan_7b["model"]["quantization"] == "none"
    assert plan_7b["sft_calibration"]["rank_candidates"] == [8, 16]
    assert plan_7b["sft_calibration"]["max_steps"] == 100
    assert plan_7b["sft_main"]["maximum_epochs"] == 2.0
    assert plan_7b["sft_main"]["selected_rank"] == 8
    assert plan_7b["dpo_calibration"]["checkpoint_steps"] == [5, 10, 20]
    assert plan_7b["dpo_calibration"]["learning_rate_fallback"] == 1e-06
    assert plan_7b["dpo_calibration"]["drift_mitigation_order"][-1] == "increase_beta_to_0.2_if_needed"
    assert plan_7b["data"]["formal_test_v2_policy"].startswith("sealed_unopened")

    composition_plan = json.loads(
        (ROOT / "configs" / "ecommerce" / "dpo_v1_4_composition_ablation.json").read_text(encoding="utf-8")
    )
    assert composition_plan["variants"]["response_only_matched"]["train_count"] == 173
    assert composition_plan["variants"]["multigranularity_matched"]["train_count"] == 173
    assert composition_plan["training"]["checkpoint_steps"] == [5, 10, 20]
    assert composition_plan["evaluation"]["formal_test_v2"] == "sealed"

    source = (ROOT / "training" / "supervised_finetuning.py").read_text(encoding="utf-8")
    assert "raw_datasets['train'].shuffle(seed=training_args.data_seed)" in source
    assert 'raw_datasets["train"].shuffle(seed=42)' not in source

    dpo_source = (ROOT / "training" / "dpo_training.py").read_text(encoding="utf-8")
    assert "train_dataset.shuffle(seed=args.seed)" in dpo_source
    assert "len(x['prompt'] + x['chosen'])" not in dpo_source
    assert "tokenizer.encode(example['prompt']" in dpo_source
    assert "weight_decay=args.weight_decay" in dpo_source
    assert "verify_reference_matches_initial_policy(" in dpo_source
    assert '"ref" not in model.peft_config' in dpo_source
    assert "trainer.compute_ref_log_probs(trainer.model, batch)" in dpo_source

    sft_script = (ROOT / "scripts" / "ecommerce" / "run_1p5b_sft.sh").read_text(encoding="utf-8")
    assert 'MAX_STEPS="${MAX_STEPS:--1}"' in sft_script
    assert '--max_steps "$MAX_STEPS"' in sft_script
    assert '--warmup_steps "$WARMUP_STEPS"' in sft_script
    assert '--learning_rate "$LEARNING_RATE"' in sft_script
    assert '--num_train_epochs "$NUM_TRAIN_EPOCHS"' in sft_script
    assert '--save_steps "$SAVE_STEPS"' in sft_script
    assert "--warmup_ratio" not in sft_script

    sft_7b_matrix = (ROOT / "scripts" / "ecommerce" / "run_7b_sft_calibration_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "ranks=(8 16)" in sft_7b_matrix
    assert "MICRO_BATCH=2 GRAD_ACCUM=16" in sft_7b_matrix
    assert "MAX_STEPS=\"$MAX_STEPS\"" in sft_7b_matrix
    assert "Refusing to overwrite immutable run directory" in sft_7b_matrix

    monitor_script = (ROOT / "scripts" / "ecommerce" / "monitor_gpu.sh").read_text(encoding="utf-8")
    assert 'command+=(-i "$gpu_index")' in monitor_script

    matrix_script = (ROOT / "scripts" / "ecommerce" / "run_1p5b_sft_matrix.sh").read_text(encoding="utf-8")
    for run_name in ("sft-r4-all", "sft-r16-all", "sft-r64-all", "sft-r16-qv"):
        assert run_name in matrix_script
    assert 'Refusing to overwrite immutable run directory' in matrix_script

    screen_script = (ROOT / "scripts" / "ecommerce" / "run_1p5b_sft_screen_eval.sh").read_text(encoding="utf-8")
    assert '--cases "$CASES_ROOT/cases.jsonl"' in screen_script
    assert '--max-new-tokens 512' in screen_script
    assert '--max-steps 6' in screen_script
    assert 'Refusing to overwrite screen evaluation' in screen_script

    initial_script = (ROOT / "scripts" / "ecommerce" / "run_1p5b_initial_screen.sh").read_text(encoding="utf-8")
    assert '--base-model "$MODEL_PATH"' in initial_script
    assert "--adapter" not in initial_script
    assert '--max-new-tokens 512' in initial_script
    assert '--max-steps 6' in initial_script
    assert 'Refusing to overwrite existing run directory' in initial_script
    assert 'EXPERIMENT_SCALE="${EXPERIMENT_SCALE:-1p5b}"' in initial_script

    initial_7b = (ROOT / "scripts" / "ecommerce" / "run_7b_initial_screen.sh").read_text(encoding="utf-8")
    assert "Qwen2.5-7B-Instruct" in initial_7b
    assert "EXPERIMENT_SCALE=7b" in initial_7b

    eval_7b = (ROOT / "scripts" / "ecommerce" / "run_7b_sft_calibration_eval.sh").read_text(encoding="utf-8")
    assert "ranks=(8 16)" in eval_7b
    assert "calibration_screen_eval" in eval_7b
    assert "Refusing to overwrite calibration evaluation" in eval_7b

    analysis_7b = (ROOT / "scripts" / "ecommerce" / "analyze_7b_sft_calibration.sh").read_text(
        encoding="utf-8"
    )
    for label in ("initial", "sft_r8", "sft_r16"):
        assert f'--run "{label}=' in analysis_7b
    assert "--resamples 10000" in analysis_7b
    assert "Refusing to overwrite immutable calibration analysis" in analysis_7b

    gate_script = (ROOT / "scripts" / "ecommerce" / "run_1p5b_sft_gate_eval.sh").read_text(encoding="utf-8")
    assert "sft-r4-all-full" in gate_script
    assert "sft-r16-all-full" in gate_script
    assert "sft-r64-all-full" not in gate_script
    assert '--cases "$CASES_ROOT/cases.jsonl"' in gate_script
    assert '--max-new-tokens 512' in gate_script
    assert '--max-steps 6' in gate_script
    assert 'Refusing to overwrite gate evaluation' in gate_script

    dpo_matrix = (ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_composition_matrix.sh").read_text(
        encoding="utf-8"
    )
    for variant in ("response_only_matched", "multigranularity_matched", "multigranularity_full"):
        assert variant in dpo_matrix
    assert "max_steps=(99 99 278)" in dpo_matrix
    assert "BETA=0.1" in dpo_matrix
    assert 'SFT_RUN_ID="${SFT_RUN_ID:?' in dpo_matrix
    assert 'Refusing to overwrite immutable run directory' in dpo_matrix

    dpo_screen = (
        ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_composition_screen_eval.sh"
    ).read_text(encoding="utf-8")
    assert "dpo-response-only-matched-beta0p1" in dpo_screen
    assert "dpo-multigranularity-matched-beta0p1" in dpo_screen
    assert "dpo-multigranularity-full-beta0p1" in dpo_screen
    assert '--cases "$CASES_ROOT/cases.jsonl"' in dpo_screen
    assert '--max-new-tokens 512' in dpo_screen
    assert '--max-steps 6' in dpo_screen
    assert "Refusing to overwrite DPO screen evaluation" in dpo_screen

    dpo_beta = (ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_beta_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "betas=(0.05 0.3)" in dpo_beta
    assert "beta_tags=(0p05 0p3)" in dpo_beta
    assert "MAX_STEPS=278" in dpo_beta
    assert "multigranularity_full" in dpo_beta
    assert "Refusing to overwrite immutable run directory" in dpo_beta

    dpo_script = (ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo.sh").read_text(encoding="utf-8")
    assert "--weight_decay 0.01" in dpo_script
    assert '--learning_rate "$LEARNING_RATE"' in dpo_script
    assert "--verify_reference_logps True" in dpo_script
    assert "--reference_logps_tolerance 1e-4" in dpo_script
    assert 'SAVE_STEPS="${SAVE_STEPS:-50}"' in dpo_script
    assert '--save_steps "$SAVE_STEPS"' in dpo_script

    preflight_7b = (ROOT / "scripts" / "ecommerce" / "run_7b_preflight.sh").read_text(encoding="utf-8")
    assert "--max-length 1024 --batch-size 2" in preflight_7b
    assert "Refusing to overwrite immutable preflight" in preflight_7b

    main_sft_7b = (ROOT / "scripts" / "ecommerce" / "run_7b_sft_main.sh").read_text(encoding="utf-8")
    assert "SAVE_STEPS=100" in main_sft_7b
    assert "NUM_TRAIN_EPOCHS=1" in main_sft_7b

    dpo_7b = (ROOT / "scripts" / "ecommerce" / "run_7b_dpo_calibration.sh").read_text(encoding="utf-8")
    assert "dpo_v1_4_rollout_quality_screen_800_v2" in dpo_7b
    assert "MAX_STEPS=20 SAVE_STEPS=5 EVAL_STEPS=5" in dpo_7b
    assert "LEARNING_RATE=2e-6" in dpo_7b
    assert '--eval_steps "$EVAL_STEPS"' in dpo_script

    v1p4_runner = (ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_v1p4_screen.sh").read_text(
        encoding="utf-8"
    )
    assert 'MAX_STEPS="${MAX_STEPS:-45}"' in v1p4_runner
    assert 'SAVE_STEPS="${SAVE_STEPS:-5}"' in v1p4_runner
    assert "dpo_v1_4_rollout_quality_screen_800_v2" in v1p4_runner

    v1p4_eval = (
        ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_v1p4_checkpoint_eval.sh"
    ).read_text(encoding="utf-8")
    assert 'CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-10 25 45}"' in v1p4_eval
    assert "Refusing to overwrite checkpoint-$step evaluation" in v1p4_eval

    v1p4_analysis = (
        ROOT / "scripts" / "ecommerce" / "analyze_1p5b_dpo_v1p4_screen.sh"
    ).read_text(encoding="utf-8")
    for label in ("dpo_step10", "dpo_step25", "dpo_step45"):
        assert label in v1p4_analysis
    assert "--resamples 10000" in v1p4_analysis
    assert "Refusing to overwrite immutable analysis" in v1p4_analysis
    assert 'EVAL_SUFFIX="${EVAL_SUFFIX:-}"' in v1p4_analysis
    assert 'ANALYSIS_SUFFIX="${ANALYSIS_SUFFIX:-$EVAL_SUFFIX}"' in v1p4_analysis

    v1p4_postprocess = (
        ROOT / "scripts" / "ecommerce" / "postprocess_1p5b_dpo_v1p4_eval.sh"
    ).read_text(encoding="utf-8")
    assert "POSTPROCESS_NEGATION_V2_COMPLETED" in v1p4_postprocess
    assert "Refusing to overwrite evaluator-v2 output" in v1p4_postprocess
    assert 'EVAL_SUFFIX="_negation_v2"' in v1p4_postprocess

    composition_runner = (
        ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_v1p4_composition_ablation.sh"
    ).read_text(encoding="utf-8")
    assert "response_only_matched multigranularity_matched" in composition_runner
    assert "MAX_STEPS=20 SAVE_STEPS=5 EVAL_STEPS=5" in composition_runner
    assert "Refusing to overwrite immutable run directory" in composition_runner

    composition_eval = (
        ROOT / "scripts" / "ecommerce" / "run_1p5b_dpo_v1p4_composition_eval.sh"
    ).read_text(encoding="utf-8")
    assert 'CHECKPOINT_STEPS="5 10 20"' in composition_eval

    composition_analysis = (
        ROOT / "scripts" / "ecommerce" / "analyze_1p5b_dpo_v1p4_composition.sh"
    ).read_text(encoding="utf-8")
    for step in (5, 10, 20):
        assert f"composition_transitions_step${{step}}.json" in composition_analysis
    assert "--resamples 10000" in composition_analysis

    data_dir = tmp_path / "data"
    _write_jsonl(data_dir / "a.jsonl", [{"value": 1}])
    first = _files([data_dir])[0]
    second = _files([data_dir])[0]
    assert first["type"] == "directory"
    assert first["sha256"] == second["sha256"]


def test_token_length_summary_reports_tail_and_exceedance():
    summary = summarize([10, 20, 30, 40, 100], limits=[32, 128])
    assert summary["p50"] == 30
    assert summary["p99"] == 100
    assert summary["limit_exceedance"]["32"] == {"count": 2, "rate": 0.4}
    assert summary["limit_exceedance"]["128"] == {"count": 0, "rate": 0.0}


def test_token_audit_adds_project_root_for_training_imports():
    source = (ROOT / "scripts" / "ecommerce" / "audit_1p5b_token_lengths.py").read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(ROOT))" in source


def test_token_audit_counts_batch_encoding_input_ids_not_mapping_keys():
    assert _encoded_length({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3
    assert _encoded_length({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}) == 3


def test_run_summary_filters_the_assigned_gpu(tmp_path):
    path = tmp_path / "hardware.csv"
    path.write_text(
        "timestamp,index,name,uuid,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_c\n"
        "t0, 0, A800, u0, 50, 1000, 81920, 200, 40\n"
        "t0, 1, A800, u1, 90, 2000, 81920, 300, 50\n"
        "t1, 0, A800, u0, 100, 1500, 81920, 250, 45\n",
        encoding="utf-8",
    )
    summary = hardware_summary(path, "0")
    assert summary["sample_count"] == 2
    assert summary["peak_memory_mib"] == 1500
    assert summary["peak_utilization_pct"] == 100


def test_screen_comparison_uses_fixed_cases_and_paired_deltas():
    cases = [
        {"case_id": "a", "tier": "iid"},
        {"case_id": "b", "tier": "iid"},
        {"case_id": "c", "tier": "challenge"},
        {"case_id": "d", "tier": "challenge"},
    ]
    checks = {
        "parse_success": True,
        "tool_selection_valid": True,
        "arguments_valid": True,
        "forbidden_tool_absent": True,
        "observation_outcomes_valid": True,
        "answer_requirements_met": True,
        "state_assertions_met": True,
        "facts_faithful": True,
        "within_step_limit": True,
    }
    left = [
        {
            "case_id": case["case_id"],
            "passed": index == 0,
            "checks": checks,
            "auto_resolution_eligible": index < 2,
            "auto_resolved": index == 0,
        }
        for index, case in enumerate(cases)
    ]
    right = [
        {
            "case_id": case["case_id"],
            "passed": index < 3,
            "checks": checks,
            "auto_resolution_eligible": index < 2,
            "auto_resolved": True,
        }
        for index, case in enumerate(cases)
    ]
    report = compare_runs(cases, [("left", left), ("right", right)], seed=42, resamples=200)

    assert report["runs"]["left"]["metrics"]["task_success_rate"]["estimate"] == 0.25
    assert report["runs"]["right"]["metrics"]["task_success_rate"]["estimate"] == 0.75
    assert report["paired_task_success_deltas"]["right_minus_left"]["estimate"] == 0.5
    assert report["paired_metric_deltas"]["right_minus_left"]["task_success_rate"]["estimate"] == 0.5
    assert report["paired_metric_deltas"]["right_minus_left"]["eligible_auto_resolution_rate"] == {
        "case_count": 2,
        "estimate": 0.5,
        "ci95_low": 0.0,
        "ci95_high": 1.0,
    }
    assert report["runs"]["right"]["metrics"]["eligible_auto_resolution_rate"]["case_count"] == 2
    assert report["runs"]["right"]["tier_metrics"]["iid"]["case_count"] == 2
    assert bootstrap_mean([0.0, 1.0], seed=7, resamples=100) == bootstrap_mean(
        [0.0, 1.0], seed=7, resamples=100
    )


def test_rollout_summary_reports_eligible_auto_resolution_rate():
    expected = {
        "acceptable_tool_sequences": [[]],
        "observation_codes": [],
        "must_not_call": [],
        "required_answer_term_groups": [],
        "state_assertions": [],
    }
    cases = [
        {"case_id": "eligible", "category": "status", "expected": expected},
        {"case_id": "ineligible", "category": "timeout", "expected": expected},
    ]
    traces = [
        {"case_id": "eligible", "final_answer": "已说明。", "termination_reason": "final"},
        {"case_id": "ineligible", "final_answer": "请稍后重试。", "termination_reason": "final"},
    ]
    results, summary = evaluate_traces(cases, traces)
    assert summary["eligible_auto_resolution_count"] == 1
    assert summary["metrics"]["eligible_auto_resolution_rate"] == 1.0
    assert results[0]["auto_resolved"] is True
    assert results[1]["auto_resolution_eligible"] is False


def _write_formal_candidate(path, cases):
    artifacts = {}
    for name in ("cases", "evaluator_cases", "private_oracle", "source_manifest"):
        artifact = path / f"{name}.jsonl"
        _write_jsonl(artifact, cases)
        import hashlib

        artifacts[name] = {"sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "status": "formal_frozen_test_unopened",
                "sealed": True,
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def test_formal_test_validator_accepts_disjoint_frozen_artifacts(tmp_path):
    candidate = tmp_path / "candidate"
    development = tmp_path / "development"
    _write_formal_candidate(
        candidate,
        [{"case_id": "formal-1", "source_ref": {"parent_id": "formal-parent"}}],
    )
    _write_jsonl(
        development / "cases.jsonl",
        [{"case_id": "dev-1", "source_ref": {"parent_id": "dev-parent"}}],
    )
    report = validate_formal_test(candidate, [development], minimum_cases=1)
    assert report["status"] == "passed"


def test_formal_test_validator_rejects_parent_leakage(tmp_path):
    candidate = tmp_path / "candidate"
    development = tmp_path / "development"
    _write_formal_candidate(
        candidate,
        [{"case_id": "formal-1", "source_ref": {"parent_id": "shared-parent"}}],
    )
    _write_jsonl(
        development / "cases.jsonl",
        [{"case_id": "dev-1", "source_ref": {"parent_id": "shared-parent"}}],
    )
    try:
        validate_formal_test(candidate, [development], minimum_cases=1)
    except ValueError as exc:
        assert "parent_id leakage" in str(exc)
    else:
        raise AssertionError("expected formal-test parent leakage to be rejected")


def test_formal_test_validator_rejects_training_metadata_leakage(tmp_path):
    candidate = tmp_path / "candidate"
    development = tmp_path / "development"
    training = tmp_path / "train.jsonl"
    _write_formal_candidate(
        candidate,
        [
            {
                "case_id": "formal-1",
                "source_ref": {
                    "parent_id": "formal-parent",
                    "source_record_id": "shared-source-record",
                },
            }
        ],
    )
    _write_jsonl(
        development / "cases.jsonl",
        [{"case_id": "dev-1", "source_ref": {"parent_id": "dev-parent"}}],
    )
    _write_jsonl(
        training,
        [{"metadata": {"parent_id": "train-parent", "source_record_id": "shared-source-record"}}],
    )
    try:
        validate_formal_test(candidate, [development], minimum_cases=1, reference_jsonl=[training])
    except ValueError as exc:
        assert "source_record_id leakage" in str(exc)
    else:
        raise AssertionError("expected formal-test training-source leakage to be rejected")


def test_formal_test_validator_rejects_exact_message_leakage(tmp_path):
    candidate = tmp_path / "candidate"
    development = tmp_path / "development"
    shared_messages = [{"role": "user", "content": "相同测试问题"}]
    _write_formal_candidate(
        candidate,
        [
            {
                "case_id": "formal-1",
                "messages": shared_messages,
                "source_ref": {"parent_id": "formal-parent"},
            }
        ],
    )
    _write_jsonl(
        development / "cases.jsonl",
        [
            {
                "case_id": "dev-1",
                "messages": shared_messages,
                "source_ref": {"parent_id": "dev-parent"},
            }
        ],
    )
    try:
        validate_formal_test(candidate, [development], minimum_cases=1)
    except ValueError as exc:
        assert "message_sha256 leakage" in str(exc)
    else:
        raise AssertionError("expected formal-test message leakage to be rejected")


def test_formal_builder_uses_unused_parent_offset_and_sealed_schema():
    source = (ROOT / "scripts" / "ecommerce" / "build_formal_test_v2.py").read_text(encoding="utf-8")
    assert "evidence_offset: int = 800" in source
    assert '"status": "formal_frozen_test_unopened"' in source
    assert '("csds-emnlp21", "dch2-dialeval2")' in source


def test_7b_formal_runner_creates_manifest_before_validation_output():
    source = (ROOT / "scripts" / "ecommerce" / "run_7b_formal_test_v2.sh").read_text(encoding="utf-8")
    assert source.index("capture_experiment_manifest.py") < source.index("validate_formal_test_v2.py")
    assert 'if [[ -e "$RUN_DIR" ]]' in source
    assert "Refusing to overwrite formal evaluation" in source
