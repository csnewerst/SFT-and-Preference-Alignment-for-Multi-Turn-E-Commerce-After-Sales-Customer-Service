#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.tool_utils import get_tool_utils


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _generate_stage(
    stage: str,
    base_model_path: Path,
    adapter_path: Path | None,
    tokenizer: Any,
    system_prompt: str,
    prompts: Iterable[Dict[str, Any]],
    max_new_tokens: int,
) -> list[Dict[str, Any]]:
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = (
        PeftModel.from_pretrained(base_model, str(adapter_path), device_map="cuda:0")
        if adapter_path is not None
        else base_model
    )
    model.eval()
    results = []
    for item in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(item["input"])},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append(
            {
                "stage": stage,
                "prompt_id": item["id"],
                "scenario": item["scenario"],
                "input": item["input"],
                "output": output,
            }
        )
    del model
    del base_model
    torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Initial, SFT, and SFT+DPO on fixed ecommerce prompts.")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-adapter", type=Path, default=ROOT / "outputs" / "ecommerce" / "05b_smoke" / "sft")
    parser.add_argument("--dpo-adapter", type=Path, default=ROOT / "outputs" / "ecommerce" / "05b_smoke" / "dpo")
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs" / "ecommerce" / "smoke_prompts_v1.json")
    parser.add_argument("--tools", type=Path, default=ROOT / "configs" / "ecommerce" / "tools_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ecommerce" / "05b_smoke" / "comparison.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    for path in (args.base_model, args.sft_adapter, args.dpo_adapter, args.prompts, args.tools):
        if not path.exists():
            raise FileNotFoundError(path)
    prompt_rows = _load_json(args.prompts).get("prompts")
    tools = _load_json(args.tools).get("tools")
    if not isinstance(prompt_rows, list) or not prompt_rows:
        raise ValueError("prompts must be a non-empty list")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list")

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_prompt = get_tool_utils("default").tool_formatter(tools)
    stages = (
        ("initial", None),
        ("sft", args.sft_adapter),
        ("sft_dpo", args.dpo_adapter),
    )
    all_results = []
    for stage, adapter in stages:
        all_results.extend(
            _generate_stage(
                stage,
                args.base_model,
                adapter,
                tokenizer,
                system_prompt,
                prompt_rows,
                args.max_new_tokens,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in all_results:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(all_results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
