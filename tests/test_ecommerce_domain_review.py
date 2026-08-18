import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from sample_domain_review_v1 import build_review_pack


def _write_rows(root, task, count):
    scenarios = ("duplicate_request", "identity_required", "damaged_exchange", "missing_order_id")
    for index in range(count):
        split = "train" if index % 3 == 0 else "validation" if index % 3 == 1 else "test"
        path = root / task / split / "data.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "sample_id": f"{task}-{index}",
            "parent_id": f"parent-{index}",
            "source_id": "source-a" if index % 2 else "source-b",
            "scenario": scenarios[index % len(scenarios)],
            "intent": "damaged",
        }
        row = {
            "conversations": [{"from": "human", "value": f"question {index}"}],
            "metadata": metadata,
        }
        if task == "dpo":
            metadata["preference_level"] = ("decision", "parameter", "response")[index % 3]
            metadata["primary_error"] = "hallucinated_state" if index % 2 else "policy_violation"
            row.update({"chosen": "good", "rejected": "bad"})
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_review_pack_is_stratified_and_deterministic(tmp_path):
    domain_root = tmp_path / "domain"
    _write_rows(domain_root, "sft", 20)
    _write_rows(domain_root, "dpo", 20)

    first = build_review_pack(domain_root, tmp_path / "first", sft_count=8, dpo_count=6)
    second = build_review_pack(domain_root, tmp_path / "second", sft_count=8, dpo_count=6)

    assert first == second
    assert first["selected"] == {"sft": 8, "dpo": 6}
    first_bytes = (tmp_path / "first" / "review_rows.jsonl").read_bytes()
    assert first_bytes == (tmp_path / "second" / "review_rows.jsonl").read_bytes()
    rows = [json.loads(line) for line in first_bytes.decode("utf-8").splitlines()]
    assert len({row["scenario"] for row in rows}) >= 3
    assert {row["task"] for row in rows} == {"sft", "dpo"}
    assert set(first["selected_strata"]["preference_level"]) == {"decision", "parameter", "response"}
    assert first["machine_precheck_passed"] is True
    assert first["human_review_fields"] == [
        "natural_expression",
        "fact_grounded",
        "chosen_clearly_preferred",
    ]
    assert sum(first["selection_type_counts"].values()) == 14
    assert first["selection_type_counts"] == {"随机分层控制": 5, "高风险复核": 9}
    assert all(row["machine_gate_status"] == "通过" for row in rows)
    assert all("business_correct" not in row and "tool_trace_correct" not in row for row in rows)
    assert all("fact_grounded" in row and "selection_type" in row for row in rows)
