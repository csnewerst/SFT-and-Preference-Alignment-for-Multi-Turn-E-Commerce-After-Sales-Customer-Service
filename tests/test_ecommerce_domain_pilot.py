import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from build_domain_pilot_v1 import _missing_order_reply, build_domain_pilot, select_stratified_sft


def _write_evidence(root, source_id, split, index, trajectory="multi_call"):
    path = root / source_id / "normalized" / split / "records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "source_id": source_id,
        "source_record_id": f"{source_id}-{index}",
        "group_id": f"{source_id}:group-{index}",
        "split": split,
        "labels": {"trajectory_type": trajectory},
        "turns": [{"role": "user", "text": "source"}, {"role": "assistant", "text": "response"}],
    }
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_domain_pilot_builds_audited_sft_and_dpo(tmp_path):
    evidence_root = tmp_path / "public"
    for index in range(30):
        split = "train" if index < 24 else "validation" if index < 27 else "test"
        _write_evidence(evidence_root, "glaive-fc-v2", split, index)

    manifest = build_domain_pilot(evidence_root, tmp_path / "output", sft_limit=30, dpo_limit=12)

    assert 0 < manifest["sft_rows"] <= 30
    assert 0 < manifest["dpo_rows"] <= 12
    assert manifest["audits"]["sft"]["passed"] is True
    assert manifest["audits"]["dpo"]["passed"] is True
    sft_rows = []
    for path in (tmp_path / "output" / "sft").rglob("*.jsonl"):
        sft_rows.extend(_load_jsonl(path))
    assert all(len(row["tools"]) == 3 for row in sft_rows)
    assert all(row["metadata"]["parent_id"].startswith("glaive-fc-v2-") for row in sft_rows)
    observations = [
        json.loads(message["value"])
        for row in sft_rows
        for message in row["conversations"]
        if message["from"] == "observation"
    ]
    assert observations
    assert all("ok" in observation and "tool" in observation for observation in observations)


def test_domain_pilot_is_byte_deterministic(tmp_path):
    evidence_root = tmp_path / "public"
    for index in range(10):
        _write_evidence(evidence_root, "bitext-support-2024", "train", index, "dialogue_pair")

    first = build_domain_pilot(evidence_root, tmp_path / "first", sft_limit=10, dpo_limit=5)
    second = build_domain_pilot(evidence_root, tmp_path / "second", sft_limit=10, dpo_limit=5)

    assert first == second
    assert (tmp_path / "first" / "sft" / "train" / "data.jsonl").read_bytes() == (
        tmp_path / "second" / "sft" / "train" / "data.jsonl"
    ).read_bytes()


def test_domain_pilot_varies_language_and_builds_near_miss_preferences(tmp_path):
    evidence_root = tmp_path / "public"
    for index in range(90):
        split = ("train", "validation", "test")[index % 3]
        _write_evidence(evidence_root, "glaive-fc-v2", split, index)

    manifest = build_domain_pilot(evidence_root, tmp_path / "output", sft_limit=60, dpo_limit=30)
    sft_rows = []
    dpo_rows = []
    for path in (tmp_path / "output" / "sft").rglob("*.jsonl"):
        sft_rows.extend(_load_jsonl(path))
    for path in (tmp_path / "output" / "dpo").rglob("*.jsonl"):
        dpo_rows.extend(_load_jsonl(path))

    final_replies = [row["conversations"][-1]["value"] for row in sft_rows]
    assert manifest["generator_version"] == "1.3.2"
    assert len(set(final_replies)) >= 20
    assert any(len(row["conversations"]) >= 7 for row in sft_rows)
    assert all(row["metadata"]["counterfactual_strength"] == "near_miss" for row in dpo_rows)
    assert all(row["chosen"] != row["rejected"] for row in dpo_rows)
    assert set(manifest["dpo_preference_level_counts"]) == {"decision", "parameter", "response"}
    assert sum(manifest["dpo_preference_level_counts"].values()) == manifest["dpo_rows"]
    assert all(count > 0 for count in manifest["dpo_preference_level_counts"].values())

    parameter_rows = [row for row in dpo_rows if row["metadata"]["preference_level"] == "parameter"]
    decision_rows = [row for row in dpo_rows if row["metadata"]["preference_level"] == "decision"]
    response_rows = [row for row in dpo_rows if row["metadata"]["preference_level"] == "response"]
    assert all(row["chosen"].startswith("Action:") and row["rejected"].startswith("Action:") for row in parameter_rows)
    assert all("target_turn_index" in row["metadata"] for row in dpo_rows)
    assert any(row["chosen"].startswith("Action:") for row in decision_rows)
    decision_kind_counts = manifest["dpo_decision_chosen_target_kind_counts"]
    assert set(decision_kind_counts) == {"action", "response"}
    assert abs(decision_kind_counts["action"] - decision_kind_counts["response"]) <= 1
    assert all(
        row["metadata"]["chosen_target_kind"]
        == ("action" if row["chosen"].startswith("Action:") else "response")
        for row in dpo_rows
    )
    assert all(not row["chosen"].startswith("Action:") for row in response_rows)

    identity_rows = [row for row in sft_rows if row["metadata"]["scenario"] == "identity_required"]
    assert identity_rows
    assert all(
        not any(
            message["from"] == "function_call"
            and json.loads(message["value"])["name"] == "create_after_sales_request"
            for message in row["conversations"]
        )
        for row in identity_rows
    )
    assert all("身份" in row["conversations"][-1]["value"] for row in identity_rows)


def test_missing_order_replies_are_compositionally_diverse():
    replies = {_missing_order_reply(f"source-{index}") for index in range(100)}
    assert len(replies) >= 70


def test_domain_pilot_filters_sources_and_excludes_test(tmp_path):
    evidence_root = tmp_path / "public"
    for index in range(30):
        split = "train" if index < 20 else "validation" if index < 25 else "test"
        _write_evidence(evidence_root, "authorized-zh", split, index)
        _write_evidence(evidence_root, "legacy-en", split, index + 100)

    manifest = build_domain_pilot(
        evidence_root,
        tmp_path / "output",
        sft_limit=25,
        dpo_limit=12,
        source_ids=["authorized-zh"],
        source_splits=["train", "validation"],
    )

    assert manifest["source_counts"] == {"authorized-zh": manifest["sft_rows"]}
    assert manifest["sft_split_counts"]["test"] == 0
    assert manifest["dpo_split_counts"]["test"] == 0
    assert manifest["requested_source_ids"] == ["authorized-zh"]
    assert manifest["requested_source_splits"] == ["train", "validation"]


def test_multistep_preferences_cover_continue_and_cross_turn_parameters(tmp_path):
    evidence_root = tmp_path / "public"
    for index in range(120):
        _write_evidence(evidence_root, "authorized-zh", "train", index, "multi_call")

    build_domain_pilot(evidence_root, tmp_path / "output", sft_limit=120, dpo_limit=400)
    dpo_rows = []
    for path in (tmp_path / "output" / "dpo").rglob("*.jsonl"):
        dpo_rows.extend(_load_jsonl(path))

    continue_rows = [row for row in dpo_rows if row["metadata"]["primary_error"] == "premature_stop"]
    parameter_parents = {}
    for row in dpo_rows:
        if row["metadata"]["preference_level"] == "parameter":
            parameter_parents.setdefault(row["metadata"]["parent_id"], set()).add(row["metadata"]["target_turn_index"])

    assert continue_rows
    assert all(row["chosen"].startswith("Action:") for row in continue_rows)
    assert all(not row["rejected"].startswith("Action:") for row in continue_rows)
    assert any(len(indices) > 1 for indices in parameter_parents.values())


def test_domain_pilot_builds_distinct_variants_without_split_leakage(tmp_path):
    evidence_root = tmp_path / "public"
    for index in range(80):
        split = "train" if index < 64 else "validation"
        _write_evidence(evidence_root, "authorized-zh", split, index)

    manifest = build_domain_pilot(
        evidence_root,
        tmp_path / "output",
        sft_limit=120,
        dpo_limit=60,
        variants_per_parent=2,
    )
    sft_rows = []
    for path in (tmp_path / "output" / "sft").rglob("*.jsonl"):
        sft_rows.extend(_load_jsonl(path))

    variants_by_parent = {}
    for row in sft_rows:
        variants_by_parent.setdefault(row["metadata"]["parent_id"], set()).add(row["metadata"]["source_variant"])

    assert manifest["variants_per_parent"] == 2
    assert manifest["raw_sft_candidates"] == 160
    assert any(variants == {0, 1} for variants in variants_by_parent.values())
    assert manifest["audits"]["sft"]["passed"] is True


def test_stratified_sft_selection_balances_scenarios_and_split_targets():
    rows = []
    for split, count in (("train", 180), ("validation", 40)):
        for index in range(count):
            scenario = "common" if index < count - 20 else "rare"
            rows.append(
                (
                    split,
                    {
                        "metadata": {
                            "sample_id": f"{split}-{index}",
                            "scenario": scenario,
                            "source_id": "source-a" if index % 2 else "source-b",
                        }
                    },
                )
            )

    selected = select_stratified_sft(rows, 100)
    split_counts = {split: sum(item[0] == split for item in selected) for split in ("train", "validation")}
    scenario_counts = {
        scenario: sum(item[1]["metadata"]["scenario"] == scenario for item in selected)
        for scenario in ("common", "rare")
    }

    assert split_counts == {"train": 90, "validation": 10}
    assert scenario_counts["rare"] >= 24
