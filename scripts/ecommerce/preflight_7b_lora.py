#!/usr/bin/env python3
"""Verify 7B files, tokenizer, BF16 LoRA forward/backward, and peak GPU memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_preflight(model_path: Path, output: Path, max_length: int = 1024, batch_size: int = 2) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 7B preflight")
    required = [model_path / "config.json", model_path / "model.safetensors.index.json"]
    missing = [str(path) for path in required if not path.is_file()]
    shards = sorted(model_path.glob("model-*.safetensors"))
    if missing or not shards:
        raise FileNotFoundError(f"incomplete model directory; missing={missing}, shards={len(shards)}")

    tracked_files = required + shards + [model_path / "tokenizer.json", model_path / "tokenizer_config.json"]
    file_hashes = {path.name: _sha256(path) for path in tracked_files if path.is_file()}
    aggregate = hashlib.sha256(
        json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    target_modules = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] != "lm_head"
        }
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        ),
    )
    prompt = "用户要求查询订单状态并按售后规则调用工具。"
    text = (prompt * (max_length // 8 + 32))
    encoded = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    encoded = {key: value.cuda() for key, value in encoded.items()}
    output_value = model(**encoded, labels=encoded["input_ids"])
    output_value.loss.backward()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "status": "passed",
        "model_path": str(model_path),
        "aggregate_file_sha256": aggregate,
        "file_sha256": file_hashes,
        "shard_count": len(shards),
        "dtype": "bfloat16",
        "lora_rank": 8,
        "lora_alpha": 16,
        "target_modules": target_modules,
        "batch_size": batch_size,
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "trainable_parameters": trainable,
        "loss": float(output_value.loss.detach().cpu()),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.model_path, args.output, args.max_length, args.batch_size), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
