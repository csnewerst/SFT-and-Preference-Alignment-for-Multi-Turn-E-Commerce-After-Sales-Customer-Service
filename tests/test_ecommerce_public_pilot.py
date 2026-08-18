import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from prepare_public_pilot import _assign_group_splits, _parse_glaive_chat, prepare_pilot


def _read_all_normalized(source_dir):
    rows = []
    for path in sorted((source_dir / "normalized").rglob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def test_bitext_local_pilot_is_traceable_and_deterministic(tmp_path):
    input_path = tmp_path / "bitext.csv"
    with input_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["flags", "instruction", "category", "intent", "response"])
        writer.writeheader()
        writer.writerow(
            {
                "flags": "BQ",
                "instruction": "My order {{Order Number}} arrived damaged.",
                "category": "ORDER",
                "intent": "damaged_order",
                "response": "Please share the order reference so support can review it.",
            }
        )
        writer.writerow(
            {
                "flags": "BL",
                "instruction": "I want to return order {{Order Number}}.",
                "category": "ORDER",
                "intent": "return_order",
                "response": "I can explain the return process after checking the order.",
            }
        )

    first = prepare_pilot("bitext-support-2024", tmp_path / "first", input_path=input_path, limit=2)
    second = prepare_pilot("bitext-support-2024", tmp_path / "second", input_path=input_path, limit=2)
    first_rows = _read_all_normalized(tmp_path / "first" / "bitext-support-2024")
    second_rows = _read_all_normalized(tmp_path / "second" / "bitext-support-2024")

    assert first["accepted_rows"] == 2
    assert first["resolved_revision"] == second["resolved_revision"]
    assert first_rows == second_rows
    assert all(row["usage"] == "rewrite_only" for row in first_rows)
    assert {row["split"] for row in first_rows}.issubset({"train", "validation", "test"})
    assert (tmp_path / "first" / "bitext-support-2024" / "report.json").is_file()
    assert {artifact["path"] for artifact in first["artifacts"]} >= {
        "raw_selected.jsonl",
        "rejected.jsonl",
        "report.json",
    }


def test_pii_rows_are_rejected_without_copying_text_to_rejection_log(tmp_path):
    input_path = tmp_path / "bitext.csv"
    input_path.write_text(
        "flags,instruction,category,intent,response\n"
        "B,Call me at 13800138000,ORDER,return_order,Okay\n",
        encoding="utf-8",
    )

    manifest = prepare_pilot("bitext-support-2024", tmp_path / "out", input_path=input_path)
    rejection_text = (tmp_path / "out" / "bitext-support-2024" / "rejected.jsonl").read_text(
        encoding="utf-8"
    )

    assert manifest["accepted_rows"] == 0
    assert manifest["rejection_counts"] == {"pii_detected": 1}
    assert "13800138000" not in rejection_text


def test_glaive_parser_keeps_multiturn_and_observation_structure():
    chat = (
        "USER: Check order. ASSISTANT: Which order? <|endoftext|> "
        "USER: EC-1001 ASSISTANT: <functioncall> {\"name\":\"query\"} <|endoftext|> "
        "FUNCTION RESPONSE: {\"ok\":true} ASSISTANT: It is delivered."
    )

    turns = _parse_glaive_chat(chat)

    assert [turn["role"] for turn in turns] == ["user", "assistant", "user", "assistant", "observation", "assistant"]


def test_group_aware_split_is_deterministic_balanced_and_leak_free():
    rows = [{"group_id": f"group-{index}"} for index in range(100)]
    second = [dict(row) for row in rows]

    _assign_group_splits(rows, "source")
    _assign_group_splits(second, "source")

    counts = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    assert counts == {"train": 80, "validation": 10, "test": 10}
    assert rows == second

    repeated = [{"group_id": "shared"}, {"group_id": "shared"}, {"group_id": "other"}]
    _assign_group_splits(repeated, "source")
    assert len({row["split"] for row in repeated if row["group_id"] == "shared"}) == 1


def test_csds_adapter_requires_explicit_rights_acknowledgement(tmp_path):
    input_path = tmp_path / "csds.json"
    input_path.write_text("[]", encoding="utf-8")

    with pytest.raises(PermissionError, match="manual-license-review"):
        prepare_pilot("csds-emnlp21", tmp_path / "out", input_path=input_path)


def test_csds_and_dch2_authorized_local_adapters(tmp_path):
    csds_path = tmp_path / "csds.json"
    csds_path.write_text(
        json.dumps(
            [
                {
                    "DialogueID": "D1",
                    "Session_id": "S1",
                    "QRole": "用户",
                    "Dialogue": [
                        {"speaker": "Q", "turn": 0, "utterance": "商 品 破 损 了"},
                        {"speaker": "A", "turn": 1, "utterance": "请 提 供 订 单 号"},
                    ],
                    "QA": [{"Topic": "售后"}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dch2_path = tmp_path / "dch2.json"
    dch2_path.write_text(
        json.dumps(
            [
                {
                    "id": "H1",
                    "turns": [
                        {"sender": "customer", "utterances": ["商品坏了"]},
                        {"sender": "helpdesk", "utterances": ["请说明具体问题"]},
                    ],
                    "annotations": [{"quality": {"A": 1, "S": 1, "E": 1}}],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csds = prepare_pilot(
        "csds-emnlp21", tmp_path / "csds-out", input_path=csds_path, rights_acknowledged=True
    )
    dch2 = prepare_pilot(
        "dch2-dialeval2", tmp_path / "dch2-out", input_path=dch2_path, rights_acknowledged=True
    )

    assert csds["accepted_rows"] == 1
    assert dch2["accepted_rows"] == 1
    assert csds["rights_acknowledged"] is True
    assert dch2["rights_acknowledged"] is True
