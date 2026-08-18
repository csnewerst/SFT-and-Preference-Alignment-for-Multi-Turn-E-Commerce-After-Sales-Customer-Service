#!/usr/bin/env python3
"""Record a one-time opening event for a hash-pinned sealed formal test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def open_test(candidate_dir: Path, expected_manifest_sha256: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"formal test was already opened for this run: {output}")
    manifest_path = candidate_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if actual != expected_manifest_sha256:
        raise ValueError(f"formal manifest hash mismatch: {actual} != {expected_manifest_sha256}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "formal_frozen_test_unopened":
        raise ValueError("formal test is not in the sealed unopened state")
    report = {
        "schema_version": "1.0",
        "status": "opened_once_for_frozen_comparison",
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir),
        "manifest_sha256": actual,
        "case_count": int(manifest["case_count"]),
        "artifact_hashes": {
            key: value["sha256"] for key, value in sorted(manifest["artifacts"].items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(open_test(args.candidate_dir, args.expected_manifest_sha256, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
