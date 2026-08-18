#!/usr/bin/env python3
"""Build and seal a disjoint formal rollout test from unused public-test parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from build_rollout_prefreeze_v1 import audit_candidates, build_candidates


ARTIFACTS = ("cases", "evaluator_cases", "private_oracle", "source_manifest")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_formal_test(
    public_root: Path,
    output_dir: Path,
    *,
    limit: int = 600,
    evidence_offset: int = 800,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite formal test: {output_dir}")
    cases, oracles, sources = build_candidates(
        public_root,
        limit=limit,
        source_ids=("csds-emnlp21", "dch2-dialeval2"),
        evidence_split="test",
        dataset_purpose="evaluation",
        evidence_offset=evidence_offset,
    )
    audit = audit_candidates(cases, oracles, sources)
    if not audit["passed"]:
        raise ValueError(f"formal test audit failed: {audit['issue_counts']}")
    output_dir.mkdir(parents=True)
    oracle_by_id = {row["case_id"]: row for row in oracles}
    rows_by_artifact = {
        "cases": cases,
        "private_oracle": oracles,
        "source_manifest": sources,
        "evaluator_cases": [
            {**case, "expected": oracle_by_id[case["case_id"]]["expected"]}
            for case in cases
        ],
    }
    artifacts: Dict[str, Any] = {}
    for name in ARTIFACTS:
        path = output_dir / f"{name}.jsonl"
        _write_jsonl(path, rows_by_artifact[name])
        artifacts[name] = {
            "count": len(rows_by_artifact[name]),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": "2.0",
        "dataset_id": "ecommerce-formal-test-v2",
        "status": "formal_frozen_test_unopened",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "evidence_split": "test",
        "evidence_offset": evidence_offset,
        "source_ids": ["csds-emnlp21", "dch2-dialeval2"],
        "tier_counts": audit["tier_counts"],
        "category_counts": audit["category_counts"],
        "source_counts": audit["source_counts"],
        "oracle_replay_pass_rate": audit["oracle_replay_pass_rate"],
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--evidence-offset", type=int, default=800)
    args = parser.parse_args()
    manifest = build_formal_test(
        args.public_root,
        args.output_dir,
        limit=args.limit,
        evidence_offset=args.evidence_offset,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
