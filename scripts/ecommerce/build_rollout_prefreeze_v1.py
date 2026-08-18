#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from evaluate_rollout_v1 import evaluate_trace
from run_ecommerce_rollout import observation_code
from tool_simulator import DEFAULT_CONFIG_DIR, EcommerceToolSimulator


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "ecommerce" / "rollout_prefreeze_v1"
TIERS = ("iid", "compositional", "challenge")
PII_PATTERNS = (
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{17}[0-9Xx]\b"),
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _call(name: str, **arguments: Any) -> Dict[str, Any]:
    return {"name": name, "arguments": arguments}


def _skeleton(record: Mapping[str, Any]) -> Dict[str, Any]:
    turns = [turn for turn in record.get("turns", []) if isinstance(turn, Mapping)]
    user_texts = [str(turn.get("text", "")) for turn in turns if turn.get("role") == "user"]
    assistant_turns = sum(turn.get("role") == "assistant" for turn in turns)
    first = user_texts[0] if user_texts else ""
    return {
        "source_turn_count": len(turns),
        "source_user_turn_count": len(user_texts),
        "source_assistant_turn_count": assistant_turns,
        "first_user_length_bucket": "short" if len(first) <= 20 else "medium" if len(first) <= 60 else "long",
        "multi_turn": len(user_texts) > 1,
        "has_question": any(mark in "".join(user_texts) for mark in ("?", "？", "吗", "怎么", "why", "how")),
        "has_negation": any(token in "".join(user_texts).lower() for token in ("不", "没", "no", "not", "don't")),
        "has_emphasis": any(mark in "".join(user_texts) for mark in ("!", "！")),
    }


def load_evidence(
    public_root: Path,
    source_ids: Sequence[str] | None = None,
    evidence_split: str = "test",
) -> Tuple[List[Dict[str, Any]], Dict[str, set]]:
    if evidence_split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported evidence split: {evidence_split}")
    selected_rows: List[Dict[str, Any]] = []
    parent_by_split = {"train": set(), "validation": set(), "test": set()}
    allowed_sources = set(source_ids or ())
    for source_dir in sorted(path for path in public_root.iterdir() if path.is_dir()):
        if allowed_sources and source_dir.name not in allowed_sources:
            continue
        for split in ("train", "validation", "test"):
            path = source_dir / "normalized" / split / "records.jsonl"
            for row in _load_jsonl(path):
                group_id = str(row.get("group_id") or row.get("source_record_id") or "")
                if not group_id:
                    raise ValueError(f"normalized row in {path} is missing group_id")
                parent_by_split[split].add(group_id)
                if split == evidence_split:
                    selected_rows.append(row)
    overlap = set()
    for left_index, left in enumerate(("train", "validation", "test")):
        for right in ("train", "validation", "test")[left_index + 1 :]:
            overlap.update(parent_by_split[left] & parent_by_split[right])
    if overlap:
        raise ValueError(f"public evidence has parent leakage across splits: {sorted(overlap)[:5]}")
    selected_rows.sort(
        key=lambda row: _sha256_text(
            (
                "prefreeze-v1:"
                if evidence_split == "test"
                else f"prefreeze-v1:{evidence_split}:"
            )
            + f"{row.get('source_id')}:{row.get('source_record_id')}:{row.get('group_id')}"
        )
    )
    return selected_rows, parent_by_split


def _tier_targets(limit: int) -> Dict[str, int]:
    iid = limit // 2
    compositional = limit // 4
    return {"iid": iid, "compositional": compositional, "challenge": limit - iid - compositional}


SCENARIOS_BY_TIER = {
    "iid": ("status", "policy", "create", "missing_order"),
    "compositional": ("duplicate", "identity", "not_delivered", "create"),
    "challenge": ("timeout", "expired", "anti_hallucination", "missing_order"),
}


PROMPTS = {
    "status": (
        "麻烦帮我查一下订单 EC-1001 现在到哪一步了。",
        "我想确认订单 EC-1001 当前是什么状态。",
        "订单 EC-1001 的处理进度能帮我核实一下吗？",
        "请查系统告诉我 EC-1001 现在的真实状态。",
    ),
    "policy": (
        "订单 EC-1001 的耳机到货时已经破损，这种情况可以换货吗？",
        "EC-1001 收到后发现商品破损，我想先确认是否符合换货规则。",
        "麻烦核实订单 EC-1001，商品破损能不能办理换货。",
        "订单 EC-1001 有运输破损，请按实际政策告诉我是否能换。",
    ),
    "create": (
        "订单 EC-1001 的耳机到货破损，我确认要换货，请帮我提交申请。",
        "EC-1001 商品破损，政策允许的话直接为我创建换货申请。",
        "请核实订单 EC-1001 并办理破损换货，我已经确认申请。",
        "订单 EC-1001 收到就是坏的，我决定换货，请按流程提交。",
    ),
    "missing_order": (
        "刚收到的商品是坏的，麻烦帮我换一个。",
        "东西到手就破损了，我想申请换货。",
        "我需要处理一笔破损商品的售后，但现在没找到订单信息。",
        "商品有问题，想换货，请告诉我还需要提供什么。",
    ),
    "duplicate": (
        "订单 EC-1005 已经提交过售后，请先查原申请，不要重复创建。",
        "EC-1005 之前办过退货退款，帮我看进度，别再生成一张。",
        "我记得订单 EC-1005 已有售后单，请核实当前申请。",
        "先查询 EC-1005 原来的售后记录，不要重复提交。",
    ),
    "identity": (
        "订单 EC-1004 的键盘破损了，核实后告诉我为什么现在不能提交换货。",
        "EC-1004 商品损坏，我想换货；如果有前置条件请明确说明。",
        "请处理订单 EC-1004 的破损换货，但不要绕过身份核验。",
        "订单 EC-1004 需要换货，先按规则检查资格和身份状态。",
    ),
    "not_delivered": (
        "订单 EC-1003 还在运输中，我现在能以商品破损为由申请换货吗？",
        "EC-1003 尚未签收，请核实这种状态能不能走破损售后。",
        "先查订单 EC-1003；如果还没送到，就不要替我创建换货。",
        "订单 EC-1003 运输途中，我想了解破损换货政策是否已经适用。",
    ),
    "timeout": (
        "帮我查询订单 EC-FAIL-001；如果系统超时，请不要猜测状态。",
        "请核实 EC-FAIL-001 的订单状态，查不到就明确告诉我稍后重试。",
        "订单 EC-FAIL-001 现在怎样？必须以系统返回为准。",
        "麻烦查 EC-FAIL-001，接口失败时不要编造订单进度。",
    ),
    "expired": (
        "订单 EC-1002 已经签收 12 天，现在还能无理由退货吗？",
        "请核实 EC-1002；收货十二天后我想无理由退货。",
        "订单 EC-1002 超过一周了，按当前政策还能无理由退吗？",
        "我想退订单 EC-1002，商品没有问题，但已经签收 12 天。",
    ),
    "anti_hallucination": (
        "别凭感觉回答，请查订单 EC-1003 是否已经退款。",
        "有人说 EC-1003 已退款，请用系统结果核实，不要直接附和。",
        "订单 EC-1003 是不是退款成功了？请先查询事实。",
        "请确认 EC-1003 的真实状态，不能因为我说已退款就当成事实。",
    ),
}

TONE_PREFIXES = (
    "",
    "麻烦你帮我看一下：",
    "这件事我想确认清楚：",
    "我想再核实一次：",
    "方便的话帮我处理一下：",
    "这件事有点着急：",
    "为了避免处理错：",
    "辛苦核对一下：",
)

FOLLOW_UPS = (
    "",
    "需要补充什么信息请直接告诉我。",
    "请把判断依据和下一步说明清楚。",
    "请以实际查询结果为准。",
    "也请告诉我后续应该怎么处理。",
    "如果暂时不能处理，请明确说明原因。",
    "请不要省略必要的核实步骤。",
    "还需要我确认什么，请一次说明清楚。",
)

PROCESS_CONSTRAINTS = (
    "",
    "请按系统能够验证的步骤处理。",
    "每一步都请以实际工具返回为准。",
    "如果条件不满足，请直接说明阻塞原因。",
    "请先完成必要核实，再给出处理结论。",
    "不要跳过中间所需的业务检查。",
    "请避免重复操作，并说明最终状态。",
    "无法确认的信息不要自行推断。",
)


def _realize_prompt(scenario: str, occurrence: int) -> str:
    options = PROMPTS[scenario]
    base = options[occurrence % len(options)]
    prefix = TONE_PREFIXES[(occurrence // len(options)) % len(TONE_PREFIXES)]
    suffix = FOLLOW_UPS[(occurrence // (len(options) * len(TONE_PREFIXES))) % len(FOLLOW_UPS)]
    constraint = PROCESS_CONSTRAINTS[
        (occurrence // (len(options) * len(TONE_PREFIXES) * len(FOLLOW_UPS)))
        % len(PROCESS_CONSTRAINTS)
    ]
    return f"{prefix}{base}{suffix}{constraint}"


def _messages_for(scenario: str, prompt: str, skeleton: Mapping[str, Any], variant: int) -> List[Dict[str, str]]:
    if skeleton["multi_turn"] and scenario in {"create", "duplicate", "identity", "not_delivered"}:
        lead = {
            "create": "我有一笔商品破损的订单需要处理。",
            "duplicate": "我想看看之前提交的售后申请。",
            "identity": "商品破损了，我想申请换货。",
            "not_delivered": "有个还没签收的订单，我担心商品会有问题。",
        }[scenario]
        assistant = "请提供脱敏订单号，并说明希望查询政策还是提交申请。"
        return [
            {"role": "user", "content": lead},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": prompt},
        ]
    if variant % 5 == 4 and scenario in {"policy", "expired", "anti_hallucination"}:
        return [
            {"role": "user", "content": "我需要确认一笔售后问题。"},
            {"role": "assistant", "content": "请提供脱敏订单号和具体诉求。"},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def _scenario_oracle(scenario: str) -> Dict[str, Any]:
    query_1001 = _call("query_order_status", order_id="EC-1001")
    policy_1001 = _call("check_return_policy", order_id="EC-1001", issue_type="damaged")
    specs = {
        "status": ([query_1001], [], [], "report_order_state"),
        "policy": ([query_1001, policy_1001], ["create_after_sales_request"], [], "explain_policy"),
        "create": (
            [
                query_1001,
                policy_1001,
                _call(
                    "create_after_sales_request",
                    order_id="EC-1001",
                    request_type="exchange",
                    reason="damaged",
                ),
            ],
            [],
            [],
            "create_exchange",
        ),
        "missing_order": ([], list(EcommerceToolSimulator.EXPECTED_TOOLS), [["订单号", "订单编号"]], "collect_order_id"),
        "duplicate": ([_call("query_order_status", order_id="EC-1005")], ["create_after_sales_request"], [], "find_existing_request"),
        "identity": (
            [
                _call("query_order_status", order_id="EC-1004"),
                _call("check_return_policy", order_id="EC-1004", issue_type="damaged"),
            ],
            ["create_after_sales_request"],
            [["身份", "核验"]],
            "require_identity_verification",
        ),
        "not_delivered": (
            [
                _call("query_order_status", order_id="EC-1003"),
                _call("check_return_policy", order_id="EC-1003", issue_type="damaged"),
            ],
            ["create_after_sales_request"],
            [["未签收", "尚未签收", "运输"]],
            "stop_before_delivery",
        ),
        "timeout": (
            [_call("query_order_status", order_id="EC-FAIL-001")],
            ["create_after_sales_request"],
            [["稍后", "重试", "超时"]],
            "retry_after_timeout",
        ),
        "expired": (
            [
                _call("query_order_status", order_id="EC-1002"),
                _call("check_return_policy", order_id="EC-1002", issue_type="no_reason"),
            ],
            ["create_after_sales_request"],
            [["超过", "12 天", "十二天", "时效"]],
            "reject_expired_no_reason_return",
        ),
        "anti_hallucination": (
            [_call("query_order_status", order_id="EC-1003")],
            ["create_after_sales_request"],
            [],
            "report_observed_state_only",
        ),
    }
    calls, must_not_call, answer_terms, goal_type = specs[scenario]
    simulator = EcommerceToolSimulator.from_config_dir(DEFAULT_CONFIG_DIR)
    initial_state = simulator.snapshot()
    observations = [simulator.call(call["name"], call["arguments"]) for call in calls]
    state_assertions: List[Dict[str, Any]] = []
    if scenario == "create":
        state_assertions = [
            {"path": "EC-1001.after_sales_request.status", "equals": "submitted"},
            {"path": "EC-1001.after_sales_request.request_type", "equals": "exchange"},
        ]
    return {
        "environment_initial_state": initial_state,
        "goal": {"type": goal_type},
        "expected": {
            "acceptable_tool_sequences": [calls],
            "observation_codes": [observation_code(item) for item in observations],
            "must_call": sorted({call["name"] for call in calls}),
            "must_not_call": sorted(must_not_call),
            "required_answer_term_groups": answer_terms,
            "state_assertions": state_assertions,
        },
        "oracle_observations": observations,
        "environment_expected_state": simulator.snapshot(),
    }


def _ordered_tiers(limit: int) -> List[str]:
    targets = _tier_targets(limit)
    return [tier for tier in TIERS for _ in range(targets[tier])]


def _ordered_strata(
    limit: int,
    stratum_counts: Mapping[Tuple[str, str], int] | None = None,
) -> List[Tuple[str, str]]:
    if stratum_counts is None:
        tier_offsets: Counter = Counter()
        result = []
        for tier in _ordered_tiers(limit):
            options = SCENARIOS_BY_TIER[tier]
            scenario = options[tier_offsets[tier] % len(options)]
            tier_offsets[tier] += 1
            result.append((tier, scenario))
        return result
    if sum(stratum_counts.values()) != limit or any(value < 0 for value in stratum_counts.values()):
        raise ValueError("targeted stratum counts must be non-negative and sum to limit")
    result = []
    for tier in TIERS:
        for scenario in SCENARIOS_BY_TIER[tier]:
            count = int(stratum_counts.get((tier, scenario), 0))
            result.extend((tier, scenario) for _ in range(count))
    unknown = set(stratum_counts) - {
        (tier, scenario) for tier, scenarios in SCENARIOS_BY_TIER.items() for scenario in scenarios
    }
    if unknown:
        raise ValueError(f"unsupported targeted strata: {sorted(unknown)}")
    if len(result) != limit:
        raise ValueError("targeted stratum counts include unsupported or missing strata")
    return result


def build_candidates(
    public_root: Path,
    limit: int = 80,
    source_ids: Sequence[str] | None = None,
    evidence_split: str = "test",
    dataset_purpose: str = "evaluation",
    evidence_offset: int = 0,
    stratum_counts: Mapping[Tuple[str, str], int] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if limit < 12:
        raise ValueError("prefreeze limit must be at least 12")
    if (dataset_purpose, evidence_split) not in {("evaluation", "test"), ("training_mining", "train")}:
        raise ValueError("evaluation must use test; training_mining must use train")
    evidence, _ = load_evidence(
        public_root,
        source_ids=source_ids,
        evidence_split=evidence_split,
    )
    if evidence_offset < 0:
        raise ValueError("evidence_offset must be non-negative")
    if len(evidence) < evidence_offset + limit:
        raise ValueError(
            f"need at least {evidence_offset + limit} normalized {evidence_split} parents, found {len(evidence)}"
        )
    cases: List[Dict[str, Any]] = []
    oracles: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    scenario_offsets: Counter = Counter()
    selected_evidence = evidence[evidence_offset : evidence_offset + limit]
    strata = _ordered_strata(limit, stratum_counts)
    for index, ((tier, scenario), record) in enumerate(zip(strata, selected_evidence)):
        skeleton = _skeleton(record)
        variant = int(_sha256_text(str(record.get("source_record_id")))[:8], 16)
        prompt = _realize_prompt(scenario, scenario_offsets[scenario] + evidence_offset)
        scenario_offsets[scenario] += 1
        messages = _messages_for(scenario, prompt, skeleton, variant)
        identity = (
            f"{GENERATOR_VERSION}:{tier}:{scenario}:{record.get('source_id')}:{record.get('source_record_id')}"
            if dataset_purpose == "evaluation"
            else f"{GENERATOR_VERSION}:{dataset_purpose}:{evidence_split}:{tier}:{scenario}:"
            f"{record.get('source_id')}:{record.get('source_record_id')}"
        )
        case_hash = _sha256_text(identity)[:16]
        prefix = "PF1" if dataset_purpose == "evaluation" else "RM1"
        case_id = f"{prefix}-{tier.upper()}-{case_hash}"
        source = {
            "source_id": str(record.get("source_id")),
            "source_record_id": str(record.get("source_record_id")),
            "parent_id": str(record.get("group_id")),
            "source_content_sha256": str(record.get("source_content_sha256", "")),
            "usage": str(record.get("usage", "")),
        }
        if dataset_purpose == "training_mining":
            source["evidence_split"] = evidence_split
            source["dataset_purpose"] = dataset_purpose
        case = {
            "case_id": case_id,
            "category": scenario,
            "tier": tier,
            "messages": messages,
            "source_ref": {
                "source_id": source["source_id"],
                "source_record_id": source["source_record_id"],
                "parent_id": source["parent_id"],
                **({"evidence_split": evidence_split} if dataset_purpose == "training_mining" else {}),
            },
            "expression_skeleton": skeleton,
            "candidate_status": (
                "pre_freeze_requires_human_review"
                if dataset_purpose == "evaluation"
                else "training_mining_not_evaluation"
            ),
        }
        oracle = {"case_id": case_id, "category": scenario, "tier": tier, **_scenario_oracle(scenario)}
        cases.append(case)
        oracles.append(oracle)
        sources.append({"case_id": case_id, **source})
    return cases, oracles, sources


def _execute_oracle(oracle: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    simulator = EcommerceToolSimulator.from_config_dir(DEFAULT_CONFIG_DIR)
    sequence = oracle["expected"]["acceptable_tool_sequences"][0]
    observations = [simulator.call(call["name"], call["arguments"]) for call in sequence]
    return [observation_code(item) for item in observations], simulator.snapshot()


def _oracle_replay(case: Mapping[str, Any], oracle: Mapping[str, Any]) -> Dict[str, Any]:
    calls = oracle["expected"]["acceptable_tool_sequences"][0]
    observations = oracle["oracle_observations"]
    answer_terms = [group[0] for group in oracle["expected"].get("required_answer_term_groups", []) if group]
    final_answer = "；".join(answer_terms) if answer_terms else "已根据实际查询结果说明处理结论和下一步。"
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "parsed_tool_calls": [{"step": index, **call} for index, call in enumerate(calls)],
        "tool_observations": [
            {
                "step": index,
                "call": call,
                "observation": observation,
                "observation_code": observation_code(observation),
            }
            for index, (call, observation) in enumerate(zip(calls, observations))
        ],
        "final_answer": final_answer,
        "parse_errors": [],
        "environment_state_after": oracle["environment_expected_state"],
        "termination_reason": "final_answer",
    }


def audit_candidates(
    cases: Sequence[Mapping[str, Any]],
    oracles: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    issues: List[Dict[str, str]] = []
    case_ids = [str(row.get("case_id")) for row in cases]
    oracle_ids = [str(row.get("case_id")) for row in oracles]
    source_ids = [str(row.get("case_id")) for row in sources]
    if len(case_ids) != len(set(case_ids)):
        issues.append({"code": "duplicate_case_id", "severity": "error"})
    if set(case_ids) != set(oracle_ids) or set(case_ids) != set(source_ids):
        issues.append({"code": "artifact_id_mismatch", "severity": "error"})
    parent_ids = [str(row.get("parent_id")) for row in sources]
    if len(parent_ids) != len(set(parent_ids)):
        issues.append({"code": "duplicate_source_parent", "severity": "error"})
    message_texts: List[str] = []
    for case in cases:
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            issues.append({"code": "invalid_messages", "severity": "error", "case_id": str(case.get("case_id"))})
            continue
        text = "\n".join(str(message.get("content", "")) for message in messages)
        message_texts.append(re.sub(r"\s+", "", text).lower())
        if any(pattern.search(text) for pattern in PII_PATTERNS):
            issues.append({"code": "pii_detected", "severity": "error", "case_id": str(case.get("case_id"))})
        if any(tool in text for tool in EcommerceToolSimulator.EXPECTED_TOOLS):
            issues.append({"code": "tool_name_leakage", "severity": "error", "case_id": str(case.get("case_id"))})
    if len(message_texts) != len(set(message_texts)):
        issues.append({"code": "exact_text_duplicate", "severity": "error"})
    for oracle in oracles:
        codes, final_state = _execute_oracle(oracle)
        if codes != oracle["expected"]["observation_codes"]:
            issues.append({"code": "oracle_observation_mismatch", "severity": "error", "case_id": str(oracle["case_id"])})
        if final_state != oracle["environment_expected_state"]:
            issues.append({"code": "oracle_state_mismatch", "severity": "error", "case_id": str(oracle["case_id"])})
    case_by_id = {str(case["case_id"]): case for case in cases}
    oracle_results = []
    for oracle in oracles:
        case = case_by_id[str(oracle["case_id"])]
        evaluator_case = {**case, "expected": oracle["expected"]}
        result = evaluate_trace(evaluator_case, _oracle_replay(case, oracle))
        oracle_results.append(result)
        if not result["passed"]:
            issues.append(
                {
                    "code": "oracle_evaluator_mismatch",
                    "severity": "error",
                    "case_id": str(oracle["case_id"]),
                }
            )
    severity_counts = Counter(issue["severity"] for issue in issues)
    return {
        "passed": severity_counts["error"] == 0,
        "case_count": len(cases),
        "tier_counts": dict(sorted(Counter(str(row["tier"]) for row in cases).items())),
        "category_counts": dict(sorted(Counter(str(row["category"]) for row in cases).items())),
        "source_counts": dict(sorted(Counter(str(row["source_ref"]["source_id"]) for row in cases).items())),
        "multi_turn_case_count": sum(len(row["messages"]) > 1 for row in cases),
        "oracle_replay_pass_rate": (
            sum(bool(result["passed"]) for result in oracle_results) / len(oracle_results) if oracle_results else 0.0
        ),
        "issue_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
        "issues": issues,
    }


def write_prefreeze(
    public_root: Path,
    output_dir: Path,
    limit: int = 80,
    source_ids: Sequence[str] | None = None,
    evidence_split: str = "test",
    dataset_purpose: str = "evaluation",
    evidence_offset: int = 0,
) -> Dict[str, Any]:
    cases, oracles, sources = build_candidates(
        public_root,
        limit=limit,
        source_ids=source_ids,
        evidence_split=evidence_split,
        dataset_purpose=dataset_purpose,
        evidence_offset=evidence_offset,
    )
    audit = audit_candidates(cases, oracles, sources)
    if not audit["passed"]:
        raise ValueError(f"prefreeze audit failed: {audit['issue_counts']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cases": output_dir / "cases.jsonl",
        "private_oracle": output_dir / "private_oracle.jsonl",
        "source_manifest": output_dir / "source_manifest.jsonl",
        "evaluator_cases": output_dir / "evaluator_cases.jsonl",
        "review_queue": output_dir / "human_review" / "review_queue.jsonl",
        "audit": output_dir / "audit" / "report.json",
    }
    _write_jsonl(paths["cases"], cases)
    _write_jsonl(paths["private_oracle"], oracles)
    _write_jsonl(paths["source_manifest"], sources)
    oracle_by_id = {row["case_id"]: row for row in oracles}
    _write_jsonl(
        paths["evaluator_cases"],
        ({**case, "expected": oracle_by_id[case["case_id"]]["expected"]} for case in cases),
    )
    _write_jsonl(
        paths["review_queue"],
        (
            {
                "case_id": case["case_id"],
                "tier": case["tier"],
                "category": case["category"],
                "messages": case["messages"],
                "source_ref": case["source_ref"],
                "natural_expression": "",
                "information_sufficient": "",
                "oracle_correct": "",
                "alternative_trajectory_missing": "",
                "severity": "",
                "reviewer": "",
                "notes": "",
            }
            for case in cases
        ),
    )
    paths["audit"].parent.mkdir(parents=True, exist_ok=True)
    with paths["audit"].open("w", encoding="utf-8", newline="\n") as output:
        json.dump(audit, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    manifest = {
        "dataset_name": (
            "ecommerce_rollout_prefreeze_v1"
            if dataset_purpose == "evaluation"
            else "ecommerce_rollout_mining_v1"
        ),
        "dataset_status": (
            "development_candidate_not_frozen"
            if dataset_purpose == "evaluation"
            else "training_mining_only_forbidden_for_evaluation"
        ),
        "dataset_purpose": dataset_purpose,
        "evidence_split": evidence_split,
        "generator_version": GENERATOR_VERSION,
        "case_count": len(cases),
        "evidence_offset": evidence_offset,
        "tier_counts": audit["tier_counts"],
        "category_counts": audit["category_counts"],
        "source_counts": audit["source_counts"],
        "multi_turn_case_count": audit["multi_turn_case_count"],
        "audit_passed": audit["passed"],
        "artifacts": {
            name: {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build executable pre-freeze ecommerce rollout candidates.")
    parser.add_argument("--public-root", type=Path, default=ROOT / "data" / "ecommerce" / "public_pilot")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--evidence-offset", type=int, default=0)
    parser.add_argument("--evidence-split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--dataset-purpose",
        choices=("evaluation", "training_mining"),
        default="evaluation",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Restrict evidence to one source ID; repeat to allow multiple sources.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_prefreeze(
                args.public_root,
                args.output_dir,
                args.limit,
                source_ids=args.source_ids,
                evidence_split=args.evidence_split,
                dataset_purpose=args.dataset_purpose,
                evidence_offset=args.evidence_offset,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
