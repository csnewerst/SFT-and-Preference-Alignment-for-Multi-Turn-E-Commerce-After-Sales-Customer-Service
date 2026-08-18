#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from tool_simulator import DEFAULT_CONFIG_DIR, EcommerceToolSimulator


GenerateFn = Callable[[List[Dict[str, str]]], str]
ACTION_RE = re.compile(
    r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)",
    re.DOTALL,
)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_generation_metrics(
    records: Sequence[Mapping[str, float]],
    *,
    case_count: int,
    model_load_seconds: float,
    end_to_end_seconds: float,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
) -> Dict[str, Any]:
    latencies = [float(record["latency_seconds"]) for record in records]
    steady = latencies[1:] if len(latencies) > 1 else latencies
    output_tokens = sum(int(record["output_tokens"]) for record in records)
    input_tokens = sum(int(record["input_tokens"]) for record in records)
    generation_seconds = sum(latencies)
    return {
        "schema_version": "1.0",
        "case_count": case_count,
        "generation_call_count": len(records),
        "model_load_seconds": model_load_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "cases_per_second": case_count / end_to_end_seconds if end_to_end_seconds else 0.0,
        "first_call_latency_seconds": latencies[0] if latencies else 0.0,
        "generation_latency_seconds": {
            "mean": generation_seconds / len(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "steady_state_p50": _percentile(steady, 0.50),
            "steady_state_p95": _percentile(steady, 0.95),
        },
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "output_tokens_per_second": output_tokens / generation_seconds if generation_seconds else 0.0,
            "mean_input_per_call": input_tokens / len(records) if records else 0.0,
            "mean_output_per_call": output_tokens / len(records) if records else 0.0,
        },
        "gpu_memory_mib": {
            "peak_allocated": peak_allocated_mib,
            "peak_reserved": peak_reserved_mib,
        },
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def observation_code(observation: Mapping[str, Any]) -> str:
    if not observation.get("ok"):
        error = observation.get("error")
        return str(error.get("code", "UNKNOWN_ERROR")) if isinstance(error, Mapping) else "UNKNOWN_ERROR"
    data = observation.get("data")
    if isinstance(data, Mapping):
        return str(data.get("reason_code", "OK"))
    return "OK"


def parse_assistant_output(content: str) -> Dict[str, Any]:
    text = content.strip()
    matches = ACTION_RE.findall(text)
    if not matches:
        if re.search(r"\bAction\s*:", text, re.IGNORECASE):
            return {"kind": "parse_error", "error": "malformed_action", "raw": text}
        return {"kind": "final", "content": text}

    calls: List[Dict[str, Any]] = []
    for name, raw_arguments in matches:
        candidate = raw_arguments.strip().strip('"').strip("`").strip()
        try:
            arguments = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return {
                "kind": "parse_error",
                "error": f"invalid_action_json: {exc.msg}",
                "raw": text,
            }
        if not isinstance(arguments, dict):
            return {"kind": "parse_error", "error": "action_input_not_object", "raw": text}
        calls.append({"name": name.strip(), "arguments": arguments})
    return {"kind": "tool_calls", "tool_calls": calls}


def run_case(
    case: Mapping[str, Any],
    generate: GenerateFn,
    *,
    simulator: EcommerceToolSimulator | None = None,
    system_prompt: str = "",
    max_steps: int = 6,
) -> Dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case must contain a non-empty string case_id")
    source_messages = case.get("messages")
    if not isinstance(source_messages, list) or not source_messages:
        raise ValueError(f"case {case_id} must contain non-empty messages")

    environment = simulator or EcommerceToolSimulator.from_config_dir(DEFAULT_CONFIG_DIR)
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in source_messages:
        if not isinstance(message, Mapping) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"case {case_id} contains an invalid message")
        messages.append({"role": str(message["role"]), "content": str(message.get("content", ""))})

    trace: Dict[str, Any] = {
        "case_id": case_id,
        "category": case.get("category", "unknown"),
        "turns": [],
        "parsed_tool_calls": [],
        "tool_observations": [],
        "final_answer": "",
        "parse_errors": [],
        "environment_state_before": environment.snapshot(),
        "environment_state_after": {},
        "termination_reason": "",
        "max_steps": max_steps,
    }

    for step in range(max_steps):
        model_output = str(generate(copy.deepcopy(messages))).strip()
        parsed = parse_assistant_output(model_output)
        turn: Dict[str, Any] = {"step": step, "model_output": model_output, "parsed_kind": parsed["kind"]}

        if parsed["kind"] == "parse_error":
            error = {"step": step, "error": parsed["error"], "model_output": model_output}
            trace["parse_errors"].append(error)
            turn["parse_error"] = parsed["error"]
            trace["turns"].append(turn)
            trace["termination_reason"] = "parse_error"
            break

        if parsed["kind"] == "final":
            trace["final_answer"] = parsed["content"]
            turn["final_answer"] = parsed["content"]
            trace["turns"].append(turn)
            trace["termination_reason"] = "final_answer"
            break

        calls = parsed["tool_calls"]
        observations = []
        for call in calls:
            observation = environment.call(call["name"], call["arguments"])
            call_record = {"step": step, **copy.deepcopy(call)}
            observation_record = {
                "step": step,
                "call": copy.deepcopy(call),
                "observation": copy.deepcopy(observation),
                "observation_code": observation_code(observation),
            }
            trace["parsed_tool_calls"].append(call_record)
            trace["tool_observations"].append(observation_record)
            observations.append(observation)
        turn["tool_calls"] = copy.deepcopy(calls)
        turn["observations"] = copy.deepcopy(observations)
        trace["turns"].append(turn)

        messages.append({"role": "assistant", "content": model_output})
        observation_payload: Any = observations[0] if len(observations) == 1 else observations
        messages.append(
            {
                "role": "user",
                "content": "Observation: " + json.dumps(observation_payload, ensure_ascii=False, sort_keys=True),
            }
        )
    else:
        trace["termination_reason"] = "max_steps"

    trace["environment_state_after"] = environment.snapshot()
    return trace


def run_cases(
    cases: Iterable[Mapping[str, Any]],
    generate: GenerateFn,
    *,
    system_prompt: str = "",
    max_steps: int = 6,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> List[Dict[str, Any]]:
    return [
        run_case(
            case,
            generate,
            simulator=EcommerceToolSimulator.from_config_dir(config_dir),
            system_prompt=system_prompt,
            max_steps=max_steps,
        )
        for case in cases
    ]


class HuggingFaceGenerator:
    def __init__(
        self,
        base_model: Path,
        adapter: Path | None,
        *,
        device: str,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_started = time.perf_counter()
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.call_metrics: List[Dict[str, float]] = []
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device,
            trust_remote_code=True,
        )
        if adapter is None:
            self.model = base
        else:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, str(adapter), device_map=device)
        self.model.eval()
        self.model_load_seconds = time.perf_counter() - load_started
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.model.device)

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize(self.model.device)
        latency = time.perf_counter() - started
        output_ids = generated[0, encoded["input_ids"].shape[1] :]
        self.call_metrics.append(
            {
                "latency_seconds": latency,
                "input_tokens": float(encoded["input_ids"].shape[1]),
                "output_tokens": float(output_ids.shape[0]),
            }
        )
        return self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
        ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run executable model-tool-environment ecommerce rollouts.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tools", type=Path, default=ROOT / "configs" / "ecommerce" / "tools_v1.json")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()

    for path in (args.cases, args.base_model, args.tools, args.config_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.adapter is not None and not args.adapter.exists():
        raise FileNotFoundError(args.adapter)
    with args.tools.open("r", encoding="utf-8") as input_file:
        tools = json.load(input_file).get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools file must contain a non-empty tools list")

    from training.tool_utils import get_tool_utils

    cases = load_jsonl(args.cases)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    generator = HuggingFaceGenerator(
        args.base_model,
        args.adapter,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    system_prompt = get_tool_utils("default").tool_formatter(tools)
    rollout_started = time.perf_counter()
    traces = run_cases(
        cases,
        generator,
        system_prompt=system_prompt,
        max_steps=args.max_steps,
        config_dir=args.config_dir,
    )
    end_to_end_seconds = time.perf_counter() - rollout_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        for trace in traces:
            output_file.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
    if args.metrics_output is not None:
        peak_allocated_mib = 0.0
        peak_reserved_mib = 0.0
        if generator.torch.cuda.is_available():
            peak_allocated_mib = generator.torch.cuda.max_memory_allocated(generator.model.device) / 1024**2
            peak_reserved_mib = generator.torch.cuda.max_memory_reserved(generator.model.device) / 1024**2
        metrics = summarize_generation_metrics(
            generator.call_metrics,
            case_count=len(cases),
            model_load_seconds=generator.model_load_seconds,
            end_to_end_seconds=end_to_end_seconds,
            peak_allocated_mib=peak_allocated_mib,
            peak_reserved_mib=peak_reserved_mib,
        )
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"case_count": len(cases), "output": str(args.output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
