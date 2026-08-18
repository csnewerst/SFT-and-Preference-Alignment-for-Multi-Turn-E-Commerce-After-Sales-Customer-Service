#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOLS_CONFIG = ROOT / "configs" / "ecommerce" / "tools_v1.json"
AUDITOR_VERSION = "1.0.0"

PII_PATTERNS = {
    "email": re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "cn_mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "cn_id": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "api_token": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    "cn_address": re.compile(
        r"(?:\u5317\u4eac\u5e02|\u4e0a\u6d77\u5e02|\u5929\u6d25\u5e02|\u91cd\u5e86\u5e02|"
        r"[\u4e00-\u9fff]{2,8}\u7701)[\u4e00-\u9fff]{2,12}"
        r"(?:\u5e02|\u533a|\u53bf).{0,30}(?:\u8def|\u8857|\u9053|\u5df7|\u53f7)"
    ),
}


@dataclass
class AuditRow:
    file: str
    row_index: int
    split: str
    row: Dict[str, Any]
    metadata: Dict[str, Any]
    text: str
    normalized_text: str
    content_hash: str
    simhash: int

    @property
    def sample_id(self) -> str | None:
        value = self.metadata.get("sample_id")
        return value if isinstance(value, str) and value else None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_manifest(path: Path | None) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if path is None:
        return {}
    records: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSONL: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            file_name = record.get("file")
            row_index = record.get("row_index")
            if not isinstance(file_name, str) or not isinstance(row_index, int) or row_index < 1:
                raise ValueError(f"{path}:{line_number} requires file and positive row_index")
            key = (file_name.replace("\\", "/"), row_index)
            if key in records:
                raise ValueError(f"duplicate manifest key: {key}")
            records[key] = record
    return records


def _infer_split(relative_path: str) -> str:
    path = Path(relative_path)
    lowered_parts = [part.lower() for part in path.parts]
    for split in ("train", "validation", "test"):
        if split in lowered_parts or path.name.lower().startswith(split):
            return split
    return "unknown"


def _extract_text(row: Mapping[str, Any]) -> str:
    parts: List[str] = []
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, Mapping):
                parts.append(f"{message.get('from', '')}:{message.get('value', '')}")
    for field in ("chosen", "rejected"):
        value = row.get(field)
        if isinstance(value, str):
            parts.append(f"{field}:{value}")
    return "\n".join(parts)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def simhash64(text: str, ngram_size: int = 4) -> int:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0
    features = (
        [compact]
        if len(compact) <= ngram_size
        else [compact[index : index + ngram_size] for index in range(len(compact) - ngram_size + 1)]
    )
    weights = [0] * 64
    for feature in features:
        digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    value = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            value |= 1 << bit
    return value


def simhash_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def _issue(
    code: str,
    severity: str,
    row: AuditRow | None,
    message: str,
    **details: Any,
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if row is not None:
        value.update({"file": row.file, "row_index": row.row_index, "sample_id": row.sample_id})
    else:
        for location_field in ("file", "row_index"):
            if location_field in details:
                value[location_field] = details.pop(location_field)
    if details:
        value["details"] = details
    return value


def _validate_schema(row: AuditRow, allowed_tools: set[str]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    conversations = row.row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return [_issue("invalid_schema", "error", row, "conversations must be a non-empty list")]

    saw_user = False
    saw_assistant = False
    saw_function_call = False
    saw_observation = False
    for message_index, message in enumerate(conversations, start=1):
        if not isinstance(message, dict):
            issues.append(
                _issue(
                    "invalid_schema",
                    "error",
                    row,
                    "conversation message must be an object",
                    message_index=message_index,
                )
            )
            continue
        role = message.get("from")
        value = message.get("value")
        if role not in {"system", "human", "user", "gpt", "assistant", "function_call", "observation"}:
            issues.append(
                _issue(
                    "invalid_role",
                    "error",
                    row,
                    f"unsupported role: {role!r}",
                    message_index=message_index,
                )
            )
        saw_user = saw_user or role in {"human", "user"}
        saw_assistant = saw_assistant or role in {"gpt", "assistant"}
        saw_function_call = saw_function_call or role == "function_call"
        saw_observation = saw_observation or role == "observation"
        if not isinstance(value, str) or not value.strip():
            issues.append(
                _issue(
                    "invalid_schema",
                    "error",
                    row,
                    "message value must be a non-empty string",
                    message_index=message_index,
                )
            )
            continue
        if role in {"function_call", "observation"}:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                issues.append(
                    _issue(
                        "invalid_tool_json",
                        "error",
                        row,
                        f"{role} value must be valid JSON",
                        message_index=message_index,
                    )
                )
                continue
            if role == "function_call":
                if (
                    not isinstance(parsed, dict)
                    or not isinstance(parsed.get("name"), str)
                    or not isinstance(parsed.get("arguments"), dict)
                ):
                    issues.append(
                        _issue(
                            "invalid_tool_json",
                            "error",
                            row,
                            "function_call requires name and object arguments",
                            message_index=message_index,
                        )
                    )
                elif parsed["name"] not in allowed_tools:
                    issues.append(
                        _issue(
                            "unknown_tool",
                            "error",
                            row,
                            f"tool is not in v1 schema: {parsed.get('name')!r}",
                            message_index=message_index,
                        )
                    )
            elif (
                not isinstance(parsed, dict)
                or not isinstance(parsed.get("ok"), bool)
                or parsed.get("tool") not in allowed_tools
            ):
                issues.append(
                    _issue(
                        "invalid_tool_json",
                        "error",
                        row,
                        "observation requires boolean ok and a v1 tool name",
                        message_index=message_index,
                    )
                )

    chosen_present = "chosen" in row.row
    rejected_present = "rejected" in row.row
    if chosen_present != rejected_present:
        issues.append(_issue("invalid_schema", "error", row, "chosen and rejected must appear together"))
    if chosen_present:
        for field in ("chosen", "rejected"):
            if not isinstance(row.row.get(field), str) or not row.row[field].strip():
                issues.append(_issue("invalid_schema", "error", row, f"{field} must be non-empty"))
        preference_level = row.metadata.get("preference_level")
        if preference_level is not None:
            if preference_level not in {"decision", "parameter", "response"}:
                issues.append(
                    _issue(
                        "invalid_preference_schema",
                        "error",
                        row,
                        f"unsupported preference_level: {preference_level!r}",
                    )
                )
            target_turn_index = row.metadata.get("target_turn_index")
            if not isinstance(target_turn_index, int) or target_turn_index < 0:
                issues.append(
                    _issue(
                        "invalid_preference_schema",
                        "error",
                        row,
                        "versioned preference rows require a non-negative target_turn_index",
                    )
                )
            if not isinstance(row.metadata.get("primary_error"), str) or not row.metadata["primary_error"]:
                issues.append(
                    _issue(
                        "invalid_preference_schema",
                        "error",
                        row,
                        "versioned preference rows require primary_error",
                    )
                )
            chosen = row.row.get("chosen", "")
            rejected = row.row.get("rejected", "")
            chosen_is_action = isinstance(chosen, str) and chosen.startswith("Action:")
            rejected_is_action = isinstance(rejected, str) and rejected.startswith("Action:")
            expected_kinds = {
                "chosen_target_kind": "action" if chosen_is_action else "response",
                "rejected_target_kind": "action" if rejected_is_action else "response",
            }
            for field, expected in expected_kinds.items():
                if row.metadata.get(field) != expected:
                    issues.append(
                        _issue(
                            "invalid_preference_schema",
                            "error",
                            row,
                            f"{field} must match the serialized target kind: {expected}",
                        )
                    )
            if preference_level == "parameter" and not (chosen_is_action and rejected_is_action):
                issues.append(
                    _issue(
                        "invalid_preference_schema",
                        "error",
                        row,
                        "parameter preferences must compare two tool-call targets",
                    )
                )
            if preference_level == "decision" and not (chosen_is_action or rejected_is_action):
                issues.append(
                    _issue(
                        "invalid_preference_schema",
                        "error",
                        row,
                        "decision preferences must include at least one tool-call target",
                    )
                )
            if preference_level == "response" and (chosen_is_action or rejected_is_action):
                issues.append(
                    _issue(
                        "invalid_preference_schema",
                        "error",
                        row,
                        "response preferences must compare final natural-language replies",
                    )
                )
    if not saw_user:
        issues.append(_issue("invalid_schema", "error", row, "sample must contain a user turn"))
    if not chosen_present and not saw_assistant:
        issues.append(_issue("invalid_schema", "error", row, "SFT sample must contain an assistant turn"))
    if saw_function_call != saw_observation:
        issues.append(
            _issue("invalid_schema", "error", row, "function_call and observation turns must appear together")
        )

    tools = row.row.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or not tools:
            issues.append(_issue("invalid_schema", "error", row, "tools must be a non-empty list when present"))
        else:
            for tool_index, tool in enumerate(tools, start=1):
                if (
                    not isinstance(tool, dict)
                    or tool.get("name") not in allowed_tools
                    or not isinstance(tool.get("description"), str)
                    or not isinstance(tool.get("parameters"), dict)
                ):
                    issues.append(
                        _issue(
                            "invalid_schema",
                            "error",
                            row,
                            "each tool must match the v1 name and contain description and parameters",
                            tool_index=tool_index,
                        )
                    )
    return issues


def _find_near_duplicate_pairs(
    rows: Sequence[AuditRow],
    max_distance: int,
    max_pairs: int,
) -> Tuple[List[Tuple[int, int, int]], bool]:
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    pairs: List[Tuple[int, int, int]] = []
    seen_pairs: set[Tuple[int, int]] = set()
    truncated = False
    for index, row in enumerate(rows):
        candidates: set[int] = set()
        for band in range(4):
            band_value = (row.simhash >> (band * 16)) & 0xFFFF
            candidates.update(buckets[(band, band_value)])
        for other_index in sorted(candidates):
            pair = (other_index, index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            other = rows[other_index]
            if row.content_hash == other.content_hash:
                continue
            distance = simhash_distance(row.simhash, other.simhash)
            if distance <= max_distance:
                pairs.append((other_index, index, distance))
                if len(pairs) >= max_pairs:
                    truncated = True
                    return pairs, truncated
        for band in range(4):
            band_value = (row.simhash >> (band * 16)) & 0xFFFF
            buckets[(band, band_value)].append(index)
    return pairs, truncated


def _quantiles(values: Sequence[int]) -> Dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(values)

    def pick(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "max": ordered[-1],
    }


def audit_dataset(
    dataset_root: Path,
    metadata_manifest: Path | None = None,
    require_metadata: bool = False,
    near_duplicate_distance: int = 3,
    max_near_duplicate_pairs: int = 500,
    tools_config: Path = DEFAULT_TOOLS_CONFIG,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not 0 <= near_duplicate_distance <= 3:
        raise ValueError("near_duplicate_distance must be between 0 and 3 for the four-band index")
    if max_near_duplicate_pairs < 1:
        raise ValueError("max_near_duplicate_pairs must be positive")
    tools = _load_json(tools_config).get("tools", [])
    allowed_tools = {tool["name"] for tool in tools if isinstance(tool, dict) and "name" in tool}
    manifest = _load_manifest(metadata_manifest)
    rows: List[AuditRow] = []
    issues: List[Dict[str, Any]] = []

    manifest_resolved = metadata_manifest.resolve() if metadata_manifest is not None else None
    files = sorted(path for path in dataset_root.rglob("*.jsonl") if path.resolve() != manifest_resolved)
    if not files:
        issues.append(_issue("empty_dataset", "error", None, "dataset root contains no JSONL files"))
    for path in files:
        relative_path = path.relative_to(dataset_root).as_posix()
        split = _infer_split(relative_path)
        with path.open("r", encoding="utf-8") as input_file:
            for row_index, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(
                        _issue(
                            "invalid_json",
                            "error",
                            None,
                            str(exc),
                            file=relative_path,
                            row_index=row_index,
                        )
                    )
                    continue
                if not isinstance(value, dict):
                    issues.append(
                        _issue(
                            "invalid_schema",
                            "error",
                            None,
                            "row must be a JSON object",
                            file=relative_path,
                            row_index=row_index,
                        )
                    )
                    continue
                embedded_metadata = value.get("metadata")
                metadata = dict(embedded_metadata) if isinstance(embedded_metadata, dict) else {}
                metadata.update(manifest.get((relative_path, row_index), {}))
                text = _extract_text(value)
                normalized = normalize_text(text)
                row = AuditRow(
                    file=relative_path,
                    row_index=row_index,
                    split=split,
                    row=value,
                    metadata=metadata,
                    text=text,
                    normalized_text=normalized,
                    content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    simhash=simhash64(normalized),
                )
                rows.append(row)
                issues.extend(_validate_schema(row, allowed_tools))

                if split == "unknown":
                    issues.append(
                        _issue(
                            "unknown_split",
                            "error",
                            row,
                            "file path must identify train, validation, or test split",
                        )
                    )

                required_metadata = ("sample_id", "group_id", "source_id", "scenario", "intent")
                missing_metadata = [name for name in required_metadata if not metadata.get(name)]
                if missing_metadata:
                    issues.append(
                        _issue(
                            "missing_metadata",
                            "error" if require_metadata else "warning",
                            row,
                            f"missing metadata fields: {', '.join(missing_metadata)}",
                        )
                    )
                if not normalized:
                    issues.append(_issue("empty_text", "error", row, "sample has no auditable text"))
                for pii_type, pattern in PII_PATTERNS.items():
                    if pattern.search(text):
                        issues.append(_issue("pii_detected", "error", row, f"detected {pii_type}", pii_type=pii_type))

    if files and not rows:
        issues.append(_issue("empty_dataset", "error", None, "JSONL files contain no valid rows"))

    sample_ids: Dict[str, AuditRow] = {}
    for row in rows:
        if not row.sample_id:
            continue
        if row.sample_id in sample_ids:
            issues.append(
                _issue(
                    "duplicate_sample_id",
                    "error",
                    row,
                    f"sample_id duplicates {sample_ids[row.sample_id].file}:{sample_ids[row.sample_id].row_index}",
                )
            )
        else:
            sample_ids[row.sample_id] = row

    groups: Dict[str, List[AuditRow]] = defaultdict(list)
    hashes: Dict[str, List[AuditRow]] = defaultdict(list)
    for row in rows:
        group_id = row.metadata.get("group_id")
        if isinstance(group_id, str) and group_id:
            groups[group_id].append(row)
        hashes[row.content_hash].append(row)

    for group_id, group_rows in sorted(groups.items()):
        splits = sorted({row.split for row in group_rows})
        if len(splits) > 1:
            issues.append(
                _issue(
                    "group_split_leakage",
                    "error",
                    group_rows[0],
                    f"group_id {group_id!r} appears in splits {splits}",
                    occurrences=[f"{row.file}:{row.row_index}" for row in group_rows],
                )
            )

    for content_hash, hash_rows in sorted(hashes.items()):
        if len(hash_rows) < 2:
            continue
        splits = sorted({row.split for row in hash_rows})
        code = "content_split_leakage" if len(splits) > 1 else "exact_duplicate"
        severity = "error" if len(splits) > 1 else "warning"
        issues.append(
            _issue(
                code,
                severity,
                hash_rows[0],
                f"normalized content appears {len(hash_rows)} times across splits {splits}",
                content_hash=content_hash,
                occurrences=[f"{row.file}:{row.row_index}" for row in hash_rows],
            )
        )

    near_pairs, near_pairs_truncated = _find_near_duplicate_pairs(
        rows, near_duplicate_distance, max_near_duplicate_pairs
    )
    for left_index, right_index, distance in near_pairs:
        left, right = rows[left_index], rows[right_index]
        cross_split = left.split != right.split
        issues.append(
            _issue(
                "near_duplicate_split_leakage" if cross_split else "near_duplicate",
                "error" if cross_split else "warning",
                right,
                f"SimHash distance {distance} from {left.file}:{left.row_index}",
                other_file=left.file,
                other_row_index=left.row_index,
                distance=distance,
            )
        )

    issue_counts = Counter(issue["code"] for issue in issues)
    severity_counts = Counter(issue["severity"] for issue in issues)
    source_counts = Counter(str(row.metadata.get("source_id", "unknown")) for row in rows)
    scenario_counts = Counter(str(row.metadata.get("scenario", "unknown")) for row in rows)
    split_counts = Counter(row.split for row in rows)
    combined_hash = hashlib.sha256("".join(sorted(row.content_hash for row in rows)).encode("ascii")).hexdigest()
    report = {
        "auditor_version": AUDITOR_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "file_count": len(files),
        "row_count": len(rows),
        "passed": severity_counts["error"] == 0,
        "require_metadata": require_metadata,
        "near_duplicate_distance": near_duplicate_distance,
        "near_duplicate_pairs_truncated": near_pairs_truncated,
        "content_set_sha256": combined_hash,
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "text_length_chars": _quantiles([len(row.text) for row in rows]),
        "issue_counts": dict(sorted(issue_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
    }
    return report, issues


def write_audit(output_dir: Path, report: Dict[str, Any], issues: Iterable[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    with (output_dir / "issues.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        for issue in issues:
            output.write(json.dumps(issue, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ecommerce SFT/DPO JSONL data quality and leakage.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=500)
    parser.add_argument("--tools-config", type=Path, default=DEFAULT_TOOLS_CONFIG)
    args = parser.parse_args()

    report, issues = audit_dataset(
        args.dataset_root,
        metadata_manifest=args.metadata_manifest,
        require_metadata=args.require_metadata,
        near_duplicate_distance=args.near_duplicate_distance,
        max_near_duplicate_pairs=args.max_near_duplicate_pairs,
        tools_config=args.tools_config,
    )
    write_audit(args.output_dir, report, issues)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
