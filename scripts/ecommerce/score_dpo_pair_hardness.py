#!/usr/bin/env python3
"""Score DPO chosen/rejected completions with the frozen selected SFT policy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_1p5b_token_lengths import _messages


def load_jsonl_dir(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in sorted(path.rglob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as input_file:
            rows.extend(json.loads(line) for line in input_file if line.strip())
    if not rows:
        raise ValueError(f"no JSONL rows found under {path}")
    return rows


def completion_token_ids(tokenizer: Any, row: Mapping[str, Any], field: str) -> List[int]:
    token_ids = tokenizer.encode(str(row[field]), add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"{field} completion must contain at least one token")
    if tokenizer.eos_token_id is not None and token_ids[-1] != tokenizer.eos_token_id:
        token_ids.append(tokenizer.eos_token_id)
    return token_ids


def attach_scores(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> List[Dict[str, Any]]:
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from trl.trainer.utils import selective_log_softmax

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output: List[Dict[str, Any]] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        sequences = []
        completion_masks = []
        completion_lengths = []
        for row in batch_rows:
            prompt_ids = tokenizer.apply_chat_template(
                _messages(row, "default"), tokenize=True, add_generation_prompt=True
            )
            if isinstance(prompt_ids, Mapping):
                prompt_ids = prompt_ids["input_ids"]
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            if not 0 < len(prompt_ids) <= max_source_length:
                raise ValueError(f"prompt token length {len(prompt_ids)} is outside the configured limit")
            for field in ("chosen", "rejected"):
                target_ids = completion_token_ids(tokenizer, row, field)
                if len(target_ids) > max_target_length:
                    raise ValueError(f"{field} token length {len(target_ids)} exceeds {max_target_length}")
                sequences.append(torch.tensor(list(prompt_ids) + target_ids, dtype=torch.long))
                completion_masks.append(
                    torch.tensor([0] * len(prompt_ids) + [1] * len(target_ids), dtype=torch.bool)
                )
                completion_lengths.append(len(target_ids))

        input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        masks = pad_sequence(completion_masks, batch_first=True, padding_value=False).to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            token_logps = selective_log_softmax(logits[:, :-1, :], input_ids[:, 1:])
            shifted_masks = masks[:, 1:]
            token_logps = token_logps.masked_fill(~shifted_masks, 0.0)
            sums = token_logps.sum(dim=1).float().cpu().tolist()
            counts = shifted_masks.sum(dim=1).cpu().tolist()

        for row_index, source_row in enumerate(batch_rows):
            chosen_sum = float(sums[2 * row_index])
            rejected_sum = float(sums[2 * row_index + 1])
            chosen_count = int(counts[2 * row_index])
            rejected_count = int(counts[2 * row_index + 1])
            chosen_mean = chosen_sum / chosen_count
            rejected_mean = rejected_sum / rejected_count
            row = copy.deepcopy(dict(source_row))
            metadata = row.setdefault("metadata", {})
            metadata["sft_hardness"] = {
                "chosen_logp_sum": chosen_sum,
                "rejected_logp_sum": rejected_sum,
                "chosen_token_count": chosen_count,
                "rejected_token_count": rejected_count,
                "chosen_mean_logp": chosen_mean,
                "rejected_mean_logp": rejected_mean,
                "mean_logp_margin": chosen_mean - rejected_mean,
            }
            output.append(row)
        print(json.dumps({"scored": len(output), "total": len(rows)}), flush=True)
    return output


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in materialized:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"count": len(materialized), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-source-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=128)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite scored output: {args.output_root}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=False
    ).to(args.device)
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False).to(args.device)
    manifest = {
        "schema_version": "1.0",
        "status": "frozen_sft_hardness_scores_not_training_data",
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path),
        "splits": {},
    }
    for split in ("train", "validation"):
        scored = attach_scores(
            load_jsonl_dir(args.input_root / split),
            tokenizer,
            model,
            args.device,
            args.batch_size,
            args.max_source_length,
            args.max_target_length,
        )
        manifest["splits"][split] = write_jsonl(args.output_root / split / "records.jsonl", scored)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
