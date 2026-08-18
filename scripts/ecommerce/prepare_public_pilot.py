#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from audit_ecommerce_data import PII_PATTERNS, normalize_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs" / "ecommerce" / "public_sources_v1.json"
TRANSFORM_VERSION = "1.0.0"
SPLIT_THRESHOLDS = (("train", 80), ("validation", 90), ("test", 100))


def load_registry(path: Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        registry = json.load(input_file)
    if not isinstance(registry.get("sources"), dict):
        raise ValueError("source registry requires a sources object")
    return registry


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in {".csv", ".json", ".jsonl"}
    )
    for file_path in files:
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_sha256(file_path).encode("ascii"))
    return digest.hexdigest()


def iter_local_rows(path: Path) -> Iterator[Dict[str, Any]]:
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    supported = [file_path for file_path in files if file_path.suffix.lower() in {".csv", ".json", ".jsonl"}]
    if not supported:
        raise ValueError(f"no CSV/JSON/JSONL files found under {path}")
    for file_path in supported:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            with file_path.open("r", encoding="utf-8-sig", newline="") as input_file:
                yield from csv.DictReader(input_file)
        elif suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{file_path}:{line_number} must contain an object")
                    yield value
        else:
            with file_path.open("r", encoding="utf-8") as input_file:
                value = json.load(input_file)
            if isinstance(value, list):
                records = value
            elif isinstance(value, dict) and isinstance(value.get("data"), list):
                records = value["data"]
            else:
                raise ValueError(f"{file_path} must contain a list or a data list")
            for row_index, row in enumerate(records, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"{file_path}: record {row_index} must be an object")
                yield row


def load_huggingface_rows(
    source: Mapping[str, Any], revision: str, cache_dir: Path, endpoint: str | None = None
) -> Tuple[Iterable[Dict[str, Any]], str]:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("automatic sources require datasets and huggingface_hub") from exc

    repo_id = str(source["dataset_repo"])
    resolved_revision = HfApi(endpoint=endpoint).dataset_info(repo_id, revision=revision).sha
    dataset = load_dataset(
        repo_id,
        split="train",
        streaming=True,
        revision=resolved_revision,
        cache_dir=str(cache_dir),
    )
    return (dict(row) for row in dataset), resolved_revision


def deterministic_sample(
    rows: Iterable[Dict[str, Any]], source_id: str, limit: int, seed: int
) -> Tuple[List[Tuple[int, Dict[str, Any], str]], int]:
    heap: List[Tuple[int, int, Dict[str, Any], str]] = []
    scanned = 0
    for source_index, row in enumerate(rows, start=1):
        scanned += 1
        raw_json = canonical_json(row)
        content_hash = sha256_text(raw_json)
        rank = int(sha256_text(f"{seed}:{source_id}:{content_hash}"), 16)
        candidate = (-rank, -source_index, row, content_hash)
        if len(heap) < limit:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
    selected = [(-rank, row, content_hash) for rank, _, row, content_hash in heap]
    selected.sort(key=lambda item: item[0])
    return selected, scanned


def _clean_text(value: Any, remove_word_spaces: bool = False) -> str:
    text = value if isinstance(value, str) else ""
    text = text.strip()
    if remove_word_spaces and re.search(r"[\u4e00-\u9fff]", text):
        text = re.sub(r"\s+", "", text)
    return text


def _parse_glaive_chat(chat: str) -> List[Dict[str, str]]:
    marker = re.compile(r"(?:^|\s)(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
    matches = list(marker.finditer(chat))
    turns: List[Dict[str, str]] = []
    role_map = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "observation"}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(chat)
        text = chat[match.end() : end].replace("<|endoftext|>", "").strip()
        if text:
            turns.append({"role": role_map[match.group(1)], "text": text})
    return turns


def _normalize_bitext(row: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    instruction = _clean_text(row.get("instruction"))
    response = _clean_text(row.get("response"))
    turns = [{"role": "user", "text": instruction}, {"role": "assistant", "text": response}]
    labels = {
        "category": _clean_text(row.get("category")),
        "intent": _clean_text(row.get("intent")),
        "flags": _clean_text(row.get("flags")),
        "trajectory_type": "dialogue_pair",
    }
    group = ":".join(value for value in (labels["intent"], labels["flags"]) if value)
    return turns, labels, group or "unknown"


def _normalize_glaive(row: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    system = _clean_text(row.get("system"))
    chat = _clean_text(row.get("chat"))
    turns = _parse_glaive_chat(chat)
    call_count = chat.count("<functioncall>")
    user_turns = sum(turn["role"] == "user" for turn in turns)
    if call_count == 0:
        trajectory_type = "no_call"
    elif call_count > 1:
        trajectory_type = "multi_call"
    elif user_turns > 1:
        trajectory_type = "multi_turn_call"
    else:
        trajectory_type = "single_call"
    labels = {
        "trajectory_type": trajectory_type,
        "source_system_sha256": sha256_text(system),
        "source_system": system,
    }
    return turns, labels, labels["source_system_sha256"][:16] or "unknown"


def _normalize_csds(row: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    dialogue = row.get("Dialogue")
    if not isinstance(dialogue, list):
        dialogue = row.get("dialogue")
    turns: List[Dict[str, str]] = []
    for utterance in dialogue if isinstance(dialogue, list) else []:
        if not isinstance(utterance, Mapping):
            continue
        role = "user" if str(utterance.get("speaker", "")).upper() == "Q" else "assistant"
        text = _clean_text(utterance.get("utterance"), remove_word_spaces=True)
        if text:
            turns.append({"role": role, "text": text})
    qa = row.get("QA") if isinstance(row.get("QA"), list) else []
    topics = sorted(
        {str(item.get("Topic")) for item in qa if isinstance(item, Mapping) and item.get("Topic")}
    )
    labels = {"trajectory_type": "multi_turn_dialogue", "topics": topics, "q_role": row.get("QRole")}
    group = str(row.get("Session_id") or row.get("DialogueID") or "unknown")
    return turns, labels, group


def _normalize_dch2(row: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
    turns: List[Dict[str, str]] = []
    for turn in row.get("turns", []) if isinstance(row.get("turns"), list) else []:
        if not isinstance(turn, Mapping):
            continue
        sender = str(turn.get("sender", "")).lower()
        role = "user" if sender == "customer" else "assistant"
        utterances = turn.get("utterances", [])
        text = "\n".join(_clean_text(value) for value in utterances if _clean_text(value)) if isinstance(utterances, list) else ""
        if text:
            turns.append({"role": role, "text": text})
    annotations = row.get("annotations") if isinstance(row.get("annotations"), list) else []
    labels = {
        "trajectory_type": "multi_turn_helpdesk",
        "annotation_count": len(annotations),
        "has_quality_scores": any(isinstance(item, Mapping) and "quality" in item for item in annotations),
    }
    return turns, labels, str(row.get("id") or "unknown")


NORMALIZERS = {
    "bitext": _normalize_bitext,
    "glaive": _normalize_glaive,
    "csds": _normalize_csds,
    "dch2": _normalize_dch2,
}


def _pii_types(text: str) -> List[str]:
    masked = text.replace("XXX@YYY.com", "").replace("xxx@yyy.com", "")
    return sorted(name for name, pattern in PII_PATTERNS.items() if pattern.search(masked))


def _assign_group_splits(rows: List[Dict[str, Any]], source_id: str) -> None:
    if not rows:
        return
    split_names = [split for split, _ in SPLIT_THRESHOLDS]
    ratios = {"train": 0.8, "validation": 0.1, "test": 0.1}
    targets = {split: len(rows) * ratios[split] for split in split_names}
    group_sizes = Counter(str(row["group_id"]) for row in rows)
    ordered_groups = sorted(
        group_sizes.items(),
        key=lambda item: (-item[1], sha256_text(f"{source_id}:{item[0]}")),
    )
    assigned_counts: Counter = Counter()
    group_splits: Dict[str, str] = {}
    for group_id, group_size in ordered_groups:
        candidates = []
        for split_index, split in enumerate(split_names):
            projected = dict(assigned_counts)
            projected[split] = projected.get(split, 0) + group_size
            deviation = sum(abs(projected.get(name, 0) - targets[name]) for name in split_names)
            candidates.append((deviation, split_index, split))
        _, _, selected_split = min(candidates)
        group_splits[group_id] = selected_split
        assigned_counts[selected_split] += group_size
    for row in rows:
        row["split"] = group_splits[str(row["group_id"])]


def normalize_selected(
    selected: Sequence[Tuple[int, Dict[str, Any], str]], source_id: str, source: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    normalizer = NORMALIZERS[str(source["adapter"])]
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    counts: Counter = Counter()
    seen_content: Dict[str, str] = {}
    for source_rank, (_, row, raw_hash) in enumerate(selected, start=1):
        try:
            turns, labels, natural_group = normalizer(row)
        except (TypeError, ValueError, KeyError) as exc:
            rejected.append({"source_content_sha256": raw_hash, "reason": "adapter_error", "detail": str(exc)})
            counts["adapter_error"] += 1
            continue
        user_turns = sum(turn["role"] == "user" for turn in turns)
        assistant_turns = sum(turn["role"] == "assistant" for turn in turns)
        if not turns or not user_turns or not assistant_turns:
            rejected.append({"source_content_sha256": raw_hash, "reason": "incomplete_dialogue"})
            counts["incomplete_dialogue"] += 1
            continue
        auditable_text = "\n".join(turn["text"] for turn in turns)
        if isinstance(labels.get("source_system"), str):
            auditable_text = f"{labels['source_system']}\n{auditable_text}"
        pii_types = _pii_types(auditable_text)
        if pii_types:
            rejected.append(
                {"source_content_sha256": raw_hash, "reason": "pii_detected", "pii_types": pii_types}
            )
            counts["pii_detected"] += 1
            continue
        normalized_hash = sha256_text(normalize_text(auditable_text))
        if normalized_hash in seen_content:
            rejected.append(
                {
                    "source_content_sha256": raw_hash,
                    "reason": "exact_duplicate",
                    "duplicate_of": seen_content[normalized_hash],
                }
            )
            counts["exact_duplicate"] += 1
            continue
        source_record_id = f"{source_id}-{raw_hash[:16]}"
        seen_content[normalized_hash] = source_record_id
        group_id = f"{source_id}:{natural_group}" if natural_group != "unknown" else source_record_id
        accepted.append(
            {
                "source_id": source_id,
                "source_record_id": source_record_id,
                "group_id": group_id,
                "language": source["language"],
                "license": source["license"],
                "usage": source["usage"],
                "source_rank": source_rank,
                "source_content_sha256": raw_hash,
                "normalized_content_sha256": normalized_hash,
                "turns": turns,
                "labels": labels,
            }
        )
        counts["accepted"] += 1
    _assign_group_splits(accepted, source_id)
    return accepted, rejected, counts


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_pilot(
    source_id: str,
    output_root: Path,
    input_path: Path | None = None,
    limit: int | None = None,
    seed: int = 20260809,
    revision: str | None = None,
    hf_endpoint: str | None = None,
    rights_acknowledged: bool = False,
    registry_path: Path = DEFAULT_REGISTRY,
) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    if source_id not in registry["sources"]:
        raise ValueError(f"unknown source_id: {source_id}")
    source = registry["sources"][source_id]
    access = source["access"]
    if access != "automatic" and not rights_acknowledged:
        raise PermissionError(
            f"{source_id} is {access}; provide local authorized files and --rights-acknowledged"
        )
    selected_limit = int(source["pilot_limit"]) if limit is None else limit
    if selected_limit < 1:
        raise ValueError("limit must be positive")

    source_output = output_root / source_id
    cache_dir = output_root / "_cache" / "huggingface"
    if input_path is not None:
        rows: Iterable[Dict[str, Any]] = iter_local_rows(input_path)
        resolved_revision = f"local:{input_fingerprint(input_path)}"
    elif access == "automatic":
        rows, resolved_revision = load_huggingface_rows(
            source, revision or str(source["default_revision"]), cache_dir, hf_endpoint
        )
    else:
        raise ValueError(f"{source_id} requires --input with authorized local files")

    selected, scanned = deterministic_sample(rows, source_id, selected_limit, seed)
    raw_path = source_output / "raw_selected.jsonl"
    _write_jsonl(raw_path, (row for _, row, _ in selected))
    accepted, rejected, counts = normalize_selected(selected, source_id, source)
    for split in ("train", "validation", "test"):
        _write_jsonl(
            source_output / "normalized" / split / "records.jsonl",
            (row for row in accepted if row["split"] == split),
        )
    rejected_path = source_output / "rejected.jsonl"
    _write_jsonl(rejected_path, rejected)

    split_counts = Counter(row["split"] for row in accepted)
    trajectory_counts = Counter(str(row["labels"].get("trajectory_type", "unknown")) for row in accepted)
    report = {
        "source_id": source_id,
        "passed": len(accepted) > 0,
        "raw_rows_scanned": scanned,
        "raw_rows_selected": len(selected),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "acceptance_rate": round(len(accepted) / len(selected), 6) if selected else 0.0,
        "rejection_counts": dict(sorted((key, value) for key, value in counts.items() if key != "accepted")),
        "split_counts": dict(sorted(split_counts.items())),
        "trajectory_counts": dict(sorted(trajectory_counts.items())),
        "selected_content_sha256": sha256_text("".join(content_hash for _, _, content_hash in selected)),
    }
    report_path = source_output / "report.json"
    with report_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    artifact_paths = [raw_path, rejected_path, report_path]
    artifact_paths.extend(sorted((source_output / "normalized").rglob("*.jsonl")))
    manifest = {
        "manifest_version": "1.0.0",
        "transform_version": TRANSFORM_VERSION,
        "source_id": source_id,
        "source_url": source["source_url"],
        "download_endpoint": hf_endpoint or "default",
        "license": source["license"],
        "access": access,
        "usage": source["usage"],
        "rights_acknowledged": rights_acknowledged,
        "resolved_revision": resolved_revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_limit": selected_limit,
        **report,
        "artifacts": [
            {
                "path": str(path.relative_to(source_output)).replace("\\", "/"),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path = source_output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare traceable public-data evidence for ecommerce pilot.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "ecommerce" / "public_pilot")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--revision")
    parser.add_argument("--hf-endpoint", help="Optional Hugging Face Hub transport endpoint.")
    parser.add_argument("--rights-acknowledged", action="store_true")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    manifest = prepare_pilot(
        source_id=args.source,
        output_root=args.output_root,
        input_path=args.input,
        limit=args.limit,
        seed=args.seed,
        revision=args.revision,
        hf_endpoint=args.hf_endpoint,
        rights_acknowledged=args.rights_acknowledged,
        registry_path=args.registry,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
