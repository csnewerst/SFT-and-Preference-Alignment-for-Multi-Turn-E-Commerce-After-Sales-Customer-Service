import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from open_sealed_formal_test import open_test


def test_open_test_pins_hash_and_refuses_second_open(tmp_path):
    candidate = tmp_path / "formal"
    candidate.mkdir()
    manifest = {
        "status": "formal_frozen_test_unopened",
        "case_count": 3,
        "artifacts": {"cases": {"sha256": "case-hash"}},
    }
    manifest_path = candidate / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output = tmp_path / "run" / "open.json"

    report = open_test(candidate, digest, output)

    assert report["status"] == "opened_once_for_frozen_comparison"
    assert report["manifest_sha256"] == digest
    with pytest.raises(FileExistsError):
        open_test(candidate, digest, output)


def test_open_test_rejects_wrong_hash(tmp_path):
    candidate = tmp_path / "formal"
    candidate.mkdir()
    (candidate / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        open_test(candidate, "0" * 64, tmp_path / "open.json")
