import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from build_rollout_prefreeze_v1 import _realize_prompt, build_candidates, write_prefreeze
from evaluate_rollout_v1 import evaluate_trace
from run_ecommerce_rollout import run_case


def _write_evidence(root, source_id, split, index, *, multi_turn=False):
    path = root / source_id / "normalized" / split / "records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    turns = [{"role": "user", "text": f"用户表达 {index}"}, {"role": "assistant", "text": "客服回复"}]
    if multi_turn:
        turns.extend([{"role": "user", "text": "继续追问"}, {"role": "assistant", "text": "继续回复"}])
    row = {
        "source_id": source_id,
        "source_record_id": f"{source_id}-{split}-{index}",
        "group_id": f"{source_id}:{split}:parent-{index}",
        "source_content_sha256": f"hash-{split}-{index}",
        "usage": "structure_only",
        "turns": turns,
        "labels": {"trajectory_type": "multi_turn" if multi_turn else "dialogue_pair"},
    }
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fixture(root):
    for index in range(16):
        _write_evidence(root, "source-a", "train", index)
    for index in range(4):
        _write_evidence(root, "source-a", "validation", index)
    for index in range(40):
        _write_evidence(root, "source-a", "test", index, multi_turn=index % 2 == 0)
    for index in range(40):
        _write_evidence(root, "source-b", "test", index, multi_turn=index % 3 == 0)


def test_prefreeze_builds_traceable_executable_artifacts(tmp_path):
    public_root = tmp_path / "public"
    _fixture(public_root)
    manifest = write_prefreeze(public_root, tmp_path / "output", limit=60)

    assert manifest["dataset_status"] == "development_candidate_not_frozen"
    assert manifest["case_count"] == 60
    assert manifest["tier_counts"] == {"challenge": 15, "compositional": 15, "iid": 30}
    assert manifest["audit_passed"] is True
    audit = json.loads((tmp_path / "output" / "audit" / "report.json").read_text(encoding="utf-8"))
    assert audit["oracle_replay_pass_rate"] == 1.0
    for name in ("cases", "private_oracle", "source_manifest", "evaluator_cases", "review_queue", "audit"):
        assert name in manifest["artifacts"]
        assert (tmp_path / "output" / manifest["artifacts"][name]["path"]).exists()


def test_prefreeze_supports_recommended_evaluation_scale(tmp_path):
    public_root = tmp_path / "public"
    for index in range(800):
        _write_evidence(public_root, "authorized-zh", "test", index, multi_turn=index % 3 == 0)

    manifest = write_prefreeze(public_root, tmp_path / "output", limit=800)

    assert manifest["case_count"] == 800
    assert manifest["tier_counts"] == {"challenge": 200, "compositional": 200, "iid": 400}
    assert manifest["audit_passed"] is True
    audit = json.loads((tmp_path / "output" / "audit" / "report.json").read_text(encoding="utf-8"))
    assert audit["issue_counts"] == {}
    assert audit["oracle_replay_pass_rate"] == 1.0


def test_prompt_realization_supports_large_mining_pool_without_id_suffixes():
    prompts = [_realize_prompt("create", index) for index in range(512)]
    assert len(prompts) == len(set(prompts))
    assert all("RM1-" not in prompt and "PF1-" not in prompt for prompt in prompts)


def test_prefreeze_can_restrict_authorized_sources(tmp_path):
    public_root = tmp_path / "public"
    for index in range(40):
        _write_evidence(public_root, "authorized-zh", "test", index)
        _write_evidence(public_root, "legacy-en", "test", index)

    manifest = write_prefreeze(
        public_root,
        tmp_path / "output",
        limit=20,
        source_ids=["authorized-zh"],
    )

    assert manifest["source_counts"] == {"authorized-zh": 20}


def test_rollout_mining_uses_train_split_and_cannot_be_misused_as_evaluation(tmp_path):
    public_root = tmp_path / "public"
    for index in range(60):
        _write_evidence(public_root, "authorized-zh", "train", index, multi_turn=index % 2 == 0)
    for index in range(20):
        _write_evidence(public_root, "authorized-zh", "test", index)

    manifest = write_prefreeze(
        public_root,
        tmp_path / "mining",
        limit=40,
        source_ids=["authorized-zh"],
        evidence_split="train",
        dataset_purpose="training_mining",
    )
    assert manifest["dataset_name"] == "ecommerce_rollout_mining_v1"
    assert manifest["dataset_status"] == "training_mining_only_forbidden_for_evaluation"
    assert manifest["evidence_split"] == "train"
    case = json.loads((tmp_path / "mining" / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert case["case_id"].startswith("RM1-")
    assert case["source_ref"]["evidence_split"] == "train"

    try:
        build_candidates(
            public_root,
            limit=40,
            evidence_split="train",
            dataset_purpose="evaluation",
        )
    except ValueError as exc:
        assert "evaluation must use test" in str(exc)
    else:
        raise AssertionError("expected evaluation/train split misuse to be rejected")


def test_prefreeze_keeps_oracle_private_and_is_deterministic(tmp_path):
    public_root = tmp_path / "public"
    _fixture(public_root)
    first = write_prefreeze(public_root, tmp_path / "first", limit=40)
    second = write_prefreeze(public_root, tmp_path / "second", limit=40)

    assert first["artifacts"]["cases"]["sha256"] == second["artifacts"]["cases"]["sha256"]
    case = json.loads((tmp_path / "first" / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    oracle = json.loads((tmp_path / "first" / "private_oracle.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "expected" not in case
    assert "environment_initial_state" not in case
    assert "expected" in oracle
    assert "environment_initial_state" in oracle


def test_build_candidates_supports_preregistered_targeted_strata(tmp_path):
    public_root = tmp_path / "public"
    for index in range(20):
        _write_evidence(public_root, "authorized-zh", "test", index)
    counts = {
        ("iid", "status"): 4,
        ("compositional", "not_delivered"): 4,
        ("challenge", "expired"): 4,
    }

    cases, oracles, sources = build_candidates(
        public_root,
        limit=12,
        stratum_counts=counts,
    )

    assert len(cases) == len(oracles) == len(sources) == 12
    assert {(case["tier"], case["category"]) for case in cases} == set(counts)
    assert Counter((case["tier"], case["category"]) for case in cases) == Counter(counts)


def test_oracle_policy_passes_existing_runner_and_evaluator(tmp_path):
    public_root = tmp_path / "public"
    _fixture(public_root)
    cases, oracles, _ = build_candidates(public_root, limit=40)
    index = next(i for i, case in enumerate(cases) if case["category"] == "policy")
    case = {**cases[index], "expected": oracles[index]["expected"]}
    outputs = iter(
        [
            'Action: query_order_status\nAction Input: {"order_id": "EC-1001"}',
            'Action: check_return_policy\nAction Input: {"order_id": "EC-1001", "issue_type": "damaged"}',
            "根据系统返回，这笔订单符合破损换货政策。",
        ]
    )
    trace = run_case(case, lambda _: next(outputs))
    result = evaluate_trace(case, trace)
    assert result["passed"] is True


def test_prefreeze_rejects_public_parent_leakage(tmp_path):
    public_root = tmp_path / "public"
    _fixture(public_root)
    leaked = {
        "source_id": "source-a",
        "source_record_id": "leak",
        "group_id": "source-a:test:parent-0",
        "turns": [{"role": "user", "text": "问题"}, {"role": "assistant", "text": "回复"}],
    }
    path = public_root / "source-a" / "normalized" / "train" / "records.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(leaked, ensure_ascii=False) + "\n")

    try:
        build_candidates(public_root, limit=40)
    except ValueError as exc:
        assert "parent leakage" in str(exc)
    else:
        raise AssertionError("expected parent leakage to be rejected")


def test_rollout_shell_can_consume_prefreeze_evaluator_cases():
    script = (ROOT / "scripts" / "ecommerce" / "run_05b_rollout_eval.sh").read_text(encoding="utf-8")
    assert 'CASES_FILE="${CASES_FILE:-$CASES_ROOT/cases.jsonl}"' in script
    assert 'BUILD_DEV_CASES="${BUILD_DEV_CASES:-1}"' in script
    assert script.count('--cases "$CASES_FILE"') == 2
