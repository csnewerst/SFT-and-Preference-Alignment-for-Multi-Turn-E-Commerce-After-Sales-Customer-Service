import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from audit_ecommerce_data import AuditRow, _find_near_duplicate_pairs, audit_dataset, simhash64, simhash_distance


def _metadata(sample_id, group_id, source_id="synthetic-test"):
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "source_id": source_id,
        "scenario": "damaged_item",
        "intent": "exchange",
    }


def _sft_row(text, sample_id, group_id):
    return {
        "conversations": [
            {"from": "human", "value": text},
            {"from": "gpt", "value": "请提供脱敏订单号，我来核对。"},
        ],
        "metadata": _metadata(sample_id, group_id),
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _codes(issues):
    return {issue["code"] for issue in issues}


def test_clean_embedded_metadata_dataset_passes(tmp_path):
    dataset = tmp_path / "dataset"
    _write_jsonl(dataset / "train" / "sft.jsonl", [_sft_row("耳机破损，订单 EC-1001。", "s1", "g1")])
    _write_jsonl(
        dataset / "validation" / "sft.jsonl",
        [_sft_row("杯子有裂纹，订单 EC-2001。", "s2", "g2")],
    )

    report, issues = audit_dataset(dataset, require_metadata=True)

    assert report["passed"] is True
    assert report["row_count"] == 2
    assert report["split_counts"] == {"train": 1, "validation": 1}
    assert issues == []


def test_sidecar_manifest_supplies_required_metadata(tmp_path):
    dataset = tmp_path / "dataset"
    row = _sft_row("商品少了一件，订单 EC-3001。", "unused", "unused")
    row.pop("metadata")
    _write_jsonl(dataset / "train.jsonl", [row])
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [{"file": "train.jsonl", "row_index": 1, **_metadata("s1", "g1", "public-pilot")}],
    )

    report, issues = audit_dataset(dataset, metadata_manifest=manifest, require_metadata=True)

    assert report["passed"] is True
    assert report["source_counts"] == {"public-pilot": 1}
    assert issues == []


def test_pii_and_unknown_tool_are_blocking_errors(tmp_path):
    dataset = tmp_path / "dataset"
    row = _sft_row("手机号 13800138000，请查订单。", "s1", "g1")
    row["conversations"].insert(
        1,
        {
            "from": "function_call",
            "value": json.dumps({"name": "delete_order", "arguments": {}}, ensure_ascii=False),
        },
    )
    _write_jsonl(dataset / "train.jsonl", [row])

    report, issues = audit_dataset(dataset, require_metadata=True)

    assert report["passed"] is False
    assert {"pii_detected", "unknown_tool"}.issubset(_codes(issues))


def test_duplicate_id_group_and_content_split_leakage_are_detected(tmp_path):
    dataset = tmp_path / "dataset"
    shared_text = "订单 EC-4001 的商品破损了。"
    _write_jsonl(dataset / "train.jsonl", [_sft_row(shared_text, "same-id", "same-group")])
    _write_jsonl(dataset / "test.jsonl", [_sft_row(shared_text, "same-id", "same-group")])

    report, issues = audit_dataset(dataset, require_metadata=True)

    assert report["passed"] is False
    assert {
        "duplicate_sample_id",
        "group_split_leakage",
        "content_split_leakage",
    }.issubset(_codes(issues))


def test_unknown_split_and_empty_dataset_fail_closed(tmp_path):
    unknown = tmp_path / "unknown"
    _write_jsonl(unknown / "samples.jsonl", [_sft_row("需要售后帮助。", "s1", "g1")])

    unknown_report, unknown_issues = audit_dataset(unknown, require_metadata=True)
    empty_report, empty_issues = audit_dataset(tmp_path / "empty", require_metadata=True)

    assert unknown_report["passed"] is False
    assert "unknown_split" in _codes(unknown_issues)
    assert empty_report["passed"] is False
    assert "empty_dataset" in _codes(empty_issues)


def test_invalid_versioned_preference_metadata_is_blocking(tmp_path):
    dataset = tmp_path / "dataset"
    row = _sft_row("订单 EC-6001 需要售后。", "dpo-1", "g1")
    row["chosen"] = "正确回复"
    row["rejected"] = "错误回复"
    row["metadata"].update(
        {
            "preference_level": "parameter",
            "primary_error": "invalid_argument",
            "target_turn_index": -1,
        }
    )
    _write_jsonl(dataset / "train.jsonl", [row])

    report, issues = audit_dataset(dataset, require_metadata=True)

    assert report["passed"] is False
    assert "invalid_preference_schema" in _codes(issues)


def test_simhash_distance_and_near_duplicate_pair_detection():
    left = AuditRow("train.jsonl", 1, "train", {}, {}, "a", "a", "hash-a", 0)
    right = AuditRow("test.jsonl", 1, "test", {}, {}, "b", "b", "hash-b", 1)

    pairs, truncated = _find_near_duplicate_pairs([left, right], max_distance=3, max_pairs=10)

    assert simhash_distance(simhash64("同一段文本"), simhash64("同一段文本")) == 0
    assert pairs == [(0, 1, 1)]
    assert truncated is False


def test_cli_writes_machine_readable_outputs(tmp_path):
    dataset = tmp_path / "dataset"
    output_dir = tmp_path / "audit"
    _write_jsonl(dataset / "train.jsonl", [_sft_row("订单 EC-5001 需要换货。", "s1", "g1")])

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ecommerce" / "audit_ecommerce_data.py"),
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--require-metadata",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads((output_dir / "report.json").read_text(encoding="utf-8"))["passed"] is True
    assert (output_dir / "issues.jsonl").read_text(encoding="utf-8") == ""
