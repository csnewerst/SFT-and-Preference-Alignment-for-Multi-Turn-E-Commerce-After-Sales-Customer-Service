#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from audit_ecommerce_data import audit_dataset, normalize_text, simhash64, simhash_distance, write_audit
from tool_simulator import EcommerceToolSimulator


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "ecommerce"
GENERATOR_VERSION = "1.3.2"

CASE_NAMES = (
    "damaged_exchange",
    "wrong_item_return",
    "missing_item_refund",
    "no_reason_expired",
    "not_delivered",
    "identity_required",
    "duplicate_request",
    "tool_timeout",
    "missing_order_id",
    "order_not_found",
)

OPENERS = {
    "train": ("麻烦帮我处理一下", "想请你帮忙看看", "这个订单需要售后", "我遇到个售后问题"),
    "validation": ("请协助核实一下", "我想咨询这笔订单", "能否帮我处理", "这边需要申请售后"),
    "test": ("帮我确认一下怎么处理", "我需要解决这个问题", "请问这种情况怎么办", "想核对一下售后方案"),
}

ISSUE_TEXT = {
    "damaged": ("商品到手已经破损", "拆包后发现外壳裂了", "收到时商品有明显损坏", "商品运输后出现破损"),
    "wrong_item": ("收到的型号和下单的不一致", "商家发错了商品", "包裹里的商品不是我买的", "实际到货款式发错了"),
    "missing_item": ("包裹里少了一件商品", "订单中的配件没有收到", "拆箱后发现有商品漏发", "到货数量比订单少"),
    "no_reason": ("商品完好但我不想要了", "暂时不需要这件商品了", "没有质量问题但想退货", "商品未损坏，我想申请退货"),
    "quality_issue": ("商品使用后出现质量问题", "商品功能异常", "收到的商品无法正常使用", "商品存在明显质量故障"),
}

REQUEST_TEXT = {
    "exchange": ("希望换货", "想换一件新的", "请帮我申请换货", "我选择更换商品"),
    "return_refund": ("希望退货退款", "想把商品退回并退款", "请帮我走退货退款", "我选择退货并退款"),
    "refund_only": ("希望直接退款", "想申请仅退款", "请帮我办理退款", "我选择退款处理"),
}

FOLLOW_UPS = (
    "需要我补充什么信息吗？",
    "那接下来应该怎么处理？",
    "可以帮我确认下一步吗？",
    "这种情况现在还能办理吗？",
    "麻烦告诉我后面要怎么做。",
)

REPLY_PREFIXES = (
    "我核对了当前返回结果。",
    "已经按你提供的信息查过了。",
    "我看到了系统这次的查询结果。",
    "结合当前订单状态和售后规则，",
    "这笔订单我已经帮你核实过了。",
    "先说结论：",
)

MISSING_ORDER_OPENERS = (
    "这笔订单目前还无法定位。",
    "现在缺少用于查询的订单标识。",
    "我还不能确认你说的是哪一笔订单。",
    "目前只有商品信息，无法对应到具体订单。",
    "要继续核实，还需要先找到对应订单。",
    "当前信息不足以查询履约和售后状态。",
)

MISSING_ORDER_REQUESTS = (
    "请提供脱敏后的订单号。",
    "麻烦从订单列表复制脱敏订单号发给我。",
    "请补充这笔交易的脱敏订单号。",
    "可以把隐藏敏感位后的订单号发来吗？",
    "请先确认并提供正确的脱敏订单号。",
    "麻烦补充订单号，敏感部分可以隐藏。",
)

MISSING_ORDER_NEXT_STEPS = (
    "拿到后我会先查询状态，再确认适用的售后方案。",
    "查询到系统结果后，我再告诉你可以采取的下一步。",
    "在状态返回前，我不会替你猜测是否满足办理条件。",
    "确认订单和政策结果后，我们再继续处理。",
    "我会根据实际返回结果判断是否需要调用售后工具。",
    "核实签收、身份和政策状态后，我再给出准确结论。",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _stable_index(key: str, size: int, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8], 16) % size


def _pick(key: str, salt: str, choices: Sequence[str]) -> str:
    return choices[_stable_index(key, len(choices), salt)]


def _reply(parent_id: str, case: str, choices: Sequence[str], *, add_prefix: bool = True) -> str:
    text = _pick(parent_id, f"reply:{case}", choices)
    if not add_prefix:
        return text
    prefix = _pick(parent_id, f"prefix:{case}", REPLY_PREFIXES)
    return f"{prefix}{text}"


def _missing_order_reply(parent_id: str) -> str:
    return "".join(
        (
            _pick(parent_id, "missing-order:opener", MISSING_ORDER_OPENERS),
            _pick(parent_id, "missing-order:request", MISSING_ORDER_REQUESTS),
            _pick(parent_id, "missing-order:next", MISSING_ORDER_NEXT_STEPS),
        )
    )


def _compact_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_evidence(
    public_pilot_root: Path,
    source_ids: Sequence[str] | None = None,
    source_splits: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    allowed_sources = set(source_ids or ())
    allowed_splits = set(source_splits or ())
    for path in sorted(public_pilot_root.glob("*/normalized/*/records.jsonl")):
        source_id = path.parents[2].name
        split = path.parent.name
        if allowed_sources and source_id not in allowed_sources:
            continue
        if allowed_splits and split not in allowed_splits:
            continue
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                if row.get("split") != split:
                    raise ValueError(f"{path}:{line_number} split mismatch")
                rows.append(row)
    if not rows:
        raise ValueError(f"no normalized public evidence found under {public_pilot_root}")
    return rows


def select_evidence(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"domain-v1:{row['source_record_id']}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:limit]


def _scenario_for_evidence(row: Mapping[str, Any], variant: int = 0) -> str:
    labels = row.get("labels") if isinstance(row.get("labels"), Mapping) else {}
    intent = str(labels.get("intent", "")).lower()
    trajectory = str(labels.get("trajectory_type", ""))
    if "damag" in intent:
        return "damaged_exchange"
    if "wrong" in intent:
        return "wrong_item_return"
    if "missing" in intent:
        return "missing_item_refund"
    if "refund" in intent or "return" in intent:
        return "no_reason_expired"
    trajectory_cases = {
        "multi_call": ("damaged_exchange", "wrong_item_return", "identity_required"),
        "multi_turn_call": ("missing_item_refund", "not_delivered", "duplicate_request"),
        "single_call": ("order_not_found", "tool_timeout", "not_delivered"),
        "no_call": ("missing_order_id", "no_reason_expired", "duplicate_request"),
    }
    candidates = trajectory_cases.get(trajectory, CASE_NAMES)
    key = f"{row['source_record_id']}:variant-{variant}"
    return candidates[_stable_index(key, len(candidates), "scenario")]


def _order_id(parent_id: str) -> str:
    return f"EC-GEN-{hashlib.sha256(parent_id.encode('utf-8')).hexdigest()[:10].upper()}"


def _make_simulator(base_order_id: str | None, generated_order_id: str) -> EcommerceToolSimulator:
    tools = _load_json(CONFIG_DIR / "tools_v1.json")
    policies = _load_json(CONFIG_DIR / "policies_v1.json")
    scenarios = _load_json(CONFIG_DIR / "scenarios_v1.json")
    orders = []
    if base_order_id is not None:
        source_order = next(order for order in scenarios["orders"] if order["order_id"] == base_order_id)
        order = copy.deepcopy(source_order)
        old_order_id = order["order_id"]
        order["order_id"] = generated_order_id
        request = order.get("after_sales_request")
        if isinstance(request, dict):
            request["order_id"] = generated_order_id
            request["request_id"] = str(request["request_id"]).replace(old_order_id, generated_order_id)
        orders.append(order)
    return EcommerceToolSimulator(tools, policies, {"orders": orders})


def _add_call(conversations: List[Dict[str, str]], simulator: EcommerceToolSimulator, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    call = {"name": name, "arguments": args}
    observation = simulator.call(name, args)
    conversations.append({"from": "function_call", "value": _compact_json(call)})
    conversations.append({"from": "observation", "value": _compact_json(observation)})
    return observation


def _user_text(split: str, parent_id: str, order_id: str | None, issue: str, request_type: str | None) -> str:
    opener = OPENERS[split][_stable_index(parent_id, len(OPENERS[split]), "opener")]
    issue_texts = ISSUE_TEXT[issue]
    problem = issue_texts[_stable_index(parent_id, len(issue_texts), "issue")]
    parts = [opener]
    if order_id:
        parts.append(f"订单号是 {order_id}")
    else:
        parts.append(f"商品编号 SKU-{hashlib.sha256(parent_id.encode()).hexdigest()[:8].upper()}")
    parts.append(problem)
    if request_type:
        requests = REQUEST_TEXT[request_type]
        parts.append(requests[_stable_index(parent_id, len(requests), "request")])
    return "，".join(parts) + "。"


def build_sft_row(
    evidence: Mapping[str, Any], tools: List[Dict[str, Any]], variant: int = 0
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    source_parent_id = str(evidence["source_record_id"])
    parent_id = f"{source_parent_id}:variant-{variant}"
    split = str(evidence["split"])
    case = _scenario_for_evidence(evidence, variant=variant)
    order_id = _order_id(parent_id)
    base_order = "EC-1001"
    issue = "damaged"
    request_type: str | None = "exchange"
    if case == "wrong_item_return":
        issue, request_type = "wrong_item", "return_refund"
    elif case == "missing_item_refund":
        issue, request_type = "missing_item", "refund_only"
    elif case == "no_reason_expired":
        base_order, issue, request_type = "EC-1002", "no_reason", "return_refund"
    elif case == "not_delivered":
        base_order, issue, request_type = "EC-1003", "damaged", "exchange"
    elif case == "identity_required":
        base_order = "EC-1004"
    elif case == "duplicate_request":
        base_order, issue, request_type = "EC-1005", "quality_issue", "return_refund"
    elif case == "tool_timeout":
        base_order, issue, request_type = "EC-FAIL-001", "quality_issue", None
    elif case == "missing_order_id":
        request_type = None
    elif case == "order_not_found":
        base_order, issue, request_type = None, "quality_issue", None

    simulator = _make_simulator(base_order, order_id)
    visible_order_id = None if case == "missing_order_id" else order_id
    conversations = [{"from": "human", "value": _user_text(split, parent_id, visible_order_id, issue, request_type)}]
    primary_error = "hallucinated_state"
    if case == "missing_order_id":
        final = _missing_order_reply(parent_id)
        rejected = _reply(
            parent_id,
            f"{case}:rejected",
            (
                "虽然还没有订单号，但这类情况一般都能办，我先按符合售后条件给你说明流程。",
                "暂时查不到具体订单，不过大概率可以处理；你可以先准备退货，订单号之后再补。",
                "商品编号已经足够判断，我先按订单状态正常处理，稍后再补订单号即可。",
                "我可以先默认订单已经签收并符合政策，直接进入售后步骤。",
            ),
            add_prefix=False,
        )
        primary_error = "missing_argument"
    else:
        query = _add_call(conversations, simulator, "query_order_status", {"order_id": order_id})
        if case == "order_not_found":
            final = _reply(parent_id, case, (
                "系统没有找到这笔订单。请检查订单号是否完整、是否来自当前账号，确认后再发给我查询。",
                "这次查询没有匹配到订单，我不能继续判断履约或售后状态。麻烦核对脱敏订单号。",
                "当前返回的是订单不存在。你可以先到订单列表复制正确的脱敏订单号，我再重新核实。",
                "暂时查不到对应订单，因此不能直接办理。请确认账号和订单号后，我们再继续。",
            ))
            rejected = _reply(parent_id, f"{case}:rejected", (
                "系统可能只是同步慢，我先按订单已签收为你说明退款步骤，之后再核对订单号。",
                "虽然没有查到记录，但这类订单通常已经进入履约，我可以先帮你准备售后申请。",
                "查询为空不影响处理，我先默认订单有效并按退货退款流程继续。",
                "可能是订单号显示问题，我先按正常订单受理，稍后再补查状态。",
            ))
        elif case == "tool_timeout":
            final = _reply(parent_id, case, (
                "订单服务这次超时了，我还没有拿到可靠状态。建议稍后重试；在此之前先不创建申请。",
                "查询暂时没有返回结果，因此现在不能判断是否满足售后条件。过一会儿我可以再帮你查。",
                "系统响应超时，我无法确认签收和售后状态。请稍后重试，避免按猜测继续办理。",
                "这次查询失败了。等订单服务恢复后再核对状态，当前不会替你提交未经确认的申请。",
            ))
            rejected = _reply(parent_id, f"{case}:rejected", (
                "虽然查询超时，但按常见进度应该已经签收，可以先发起售后，之后再补状态。",
                "暂时没有返回结果，不过大概率符合条件；我先按可售后方案继续。",
                "系统不稳定时可以先采用最近状态，我先默认订单已签收并准备申请。",
                "查询失败不必等待，我可以先提交申请，若状态不符再撤回。",
            ))
        elif case == "duplicate_request":
            final = _reply(parent_id, case, (
                "这笔订单已经有一张售后申请，当前不能重复创建。请先在原申请中查看进度或补充材料。",
                "系统检测到进行中的售后单，所以这次不再新建。你可以继续跟进现有申请。",
                "已有申请正在处理，重复提交可能造成冲突。建议回到原售后单查看最新状态。",
                "同一问题已经受理，我不会再创建第二张申请；后续操作请在现有申请中完成。",
            ))
            rejected = _reply(parent_id, f"{case}:rejected", (
                "原申请如果处理较慢，可以再提交一张相同类型的售后单，提高处理优先级。",
                "已经有申请也没关系，我可以重新创建一张，让系统自动保留进度更快的那张。",
                "为了避免等待，我先再发起一次，之后你再选择保留其中一张申请。",
                "现有申请可能卡住了，我直接补建一张新的售后单会更快。",
            ))
            primary_error = "policy_violation"
        else:
            policy = _add_call(
                conversations,
                simulator,
                "check_return_policy",
                {"order_id": order_id, "issue_type": issue},
            )
            eligible = bool(policy.get("data", {}).get("eligible"))
            if case == "no_reason_expired":
                final = _reply(parent_id, case, (
                    "订单已经超过七天无理由退货期限，因此不能按无理由退货创建申请。如果商品另有质量问题，可以提供情况再核实。",
                    "当前已过无理由退货时效，这个理由下无法提交。若存在质量故障，请说明实际问题，我可以重新核对对应政策。",
                    "系统判定无理由退货已超期，所以这次不能直接办理。你仍可补充是否有质量或错漏发问题。",
                    "按当前规则，这笔订单不再适用七天无理由退货，我不会承诺创建成功；其他售后原因可以另行核实。",
                ))
                rejected = _reply(parent_id, f"{case}:rejected", (
                    "虽然刚超过期限，通常仍有协商空间，我先按无理由退货为你提交一次试试。",
                    "规则显示已超期，但系统有时会放行，我可以先创建申请，失败后再考虑其他方案。",
                    "时间只超过了一点，我先按仍在有效期内处理，后续审核一般会通过。",
                    "无理由退货已经过期，不过我可以先替你受理，不需要改成其他售后原因。",
                ))
                primary_error = "policy_violation"
            elif case == "not_delivered":
                final = _reply(parent_id, case, (
                    "订单目前还未签收，现阶段不能直接创建换货申请。请先关注配送进度；若物流异常，再按履约问题处理。",
                    "查询显示商品尚在履约中，换货流程还不能启动。建议先确认配送状态，确有异常再继续。",
                    "这笔订单未完成签收，因此当前不满足该售后申请条件。我先不提交，避免产生错误工单。",
                    "目前应先处理未送达问题，而不是创建换货。等订单状态明确后，再选择对应售后方案。",
                ))
                rejected = _reply(parent_id, f"{case}:rejected", (
                    "虽然还没签收，但可以先把换货申请建好，收到商品后系统会自动继续处理。",
                    "配送中的订单一般也能预先申请，我先提交换货，后续再补签收状态。",
                    "未送达不影响受理，我先按商品问题创建申请，这样可以节省等待时间。",
                    "我先替你发起换货；如果最后仍未送达，再把申请改成物流异常即可。",
                ))
                primary_error = "policy_violation"
            elif case == "identity_required":
                final = _reply(parent_id, case, (
                    "订单当前还没有完成身份核验，因此这次不能创建售后申请。请先完成验证，我再继续提交。",
                    "查询结果显示身份尚未核验。完成本人验证后，可以回来继续办理换货。",
                    "这一步需要先确认订单归属，目前不能调用创建工具。请完成身份核验后再发起申请。",
                    "当前身份状态不满足创建条件。为了保护订单信息，需要先完成验证。",
                ))
                rejected = _reply(parent_id, f"{case}:rejected", (
                    "身份验证可以之后补，我先代为创建申请，避免错过处理时间。",
                    "系统要求核验，但这类情况通常能先受理，我先提交，后续再验证本人信息。",
                    "为了提高效率，我可以先跳过这一步创建售后单，审核时再补身份材料。",
                    "订单信息基本一致，我先按本人操作继续，不必等待身份核验。",
                ))
                primary_error = "policy_violation"
            elif eligible and request_type:
                created = _add_call(
                    conversations,
                    simulator,
                    "create_after_sales_request",
                    {"order_id": order_id, "request_type": request_type, "reason": issue},
                )
                if created.get("ok"):
                    request_id = created["data"]["request_id"]
                    final = _reply(parent_id, case, (
                        f"申请已经提交，售后单号是 {request_id}。目前状态仅为已提交，请按页面要求补充凭证并等待审核。",
                        f"已为你创建售后单 {request_id}，系统返回的状态是已提交。后续审核和处理结果请以申请页面为准。",
                        f"这次创建成功，申请编号为 {request_id}。现在还没有审核结论，你可以在售后页面继续上传材料、查看进度。",
                        f"售后申请 {request_id} 已进入提交状态。请保留商品和相关凭证，最终结果需要等待平台审核。",
                        f"我已经完成申请创建，编号 {request_id}。当前只能确认已提交，不能提前承诺退款或换货结果。",
                    ))
                    rejected = _reply(parent_id, f"{case}:rejected", (
                        f"售后单 {request_id} 已提交，通常很快就会通过，我先按审核成功告诉你准备退款。",
                        f"申请编号是 {request_id}，提交成功基本等于审核通过，款项预计很快到账。",
                        f"已经生成售后单 {request_id}，我可以确认后续会按你的要求完成退款或换货。",
                        f"申请 {request_id} 已进入系统，审核只是流程步骤，你可以视为已经处理成功。",
                    ))
                    primary_error = "hallucinated_state"
                else:
                    raise AssertionError(f"eligible deterministic create unexpectedly failed for {case}: {created}")
            else:
                final = _reply(parent_id, case, (
                    "当前返回条件不符合对应售后政策，所以这次不能创建申请。你可以补充其他真实问题，我再核对是否适用不同规则。",
                    "系统判断该请求不满足办理条件，我不会直接提交。若商品情况有变化，请提供准确信息后重新核实。",
                    "按现有状态和规则，这个申请无法创建。我可以继续说明规则，但不能越过系统条件办理。",
                    "这次政策校验没有通过，因此先不创建售后单。请确认问题类型是否选择准确。",
                ))
                rejected = _reply(parent_id, f"{case}:rejected", (
                    "政策结果只是参考，我可以先提交申请，让后续审核人员再判断是否符合条件。",
                    "虽然当前显示不满足条件，但先创建通常不会有影响，我替你试着提交。",
                    "系统规则可能没有覆盖特殊情况，我先按符合政策创建，之后再补充说明。",
                    "不用受这次校验结果限制，我可以直接进入申请步骤。",
                ))
                primary_error = "policy_violation"
    if case not in {"missing_order_id", "order_not_found", "tool_timeout"} and _stable_index(parent_id, 3, "follow-up") == 0:
        conversations.append({"from": "human", "value": _pick(parent_id, "follow-up-text", FOLLOW_UPS)})
    conversations.append({"from": "gpt", "value": final})
    sample_id = f"sft-v1-{hashlib.sha256(parent_id.encode()).hexdigest()[:16]}"
    metadata = {
        "sample_id": sample_id,
        "parent_id": source_parent_id,
        "group_id": str(evidence["group_id"]),
        "source_id": str(evidence["source_id"]),
        "scenario": case,
        "intent": issue,
        "difficulty": "hard" if case in {"identity_required", "tool_timeout", "duplicate_request"} else "medium",
        "policy_version": "2026-08-v1",
        "tool_schema_version": "1.0.0",
        "generator_version": GENERATOR_VERSION,
        "style_variant": _stable_index(parent_id, 24, "style-variant"),
        "source_variant": variant,
        "review_status": "auto_generated",
    }
    return {"conversations": conversations, "tools": tools, "metadata": metadata}, {
        "chosen": final,
        "rejected": rejected,
        "primary_error": primary_error,
    }


def _parse_call_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    value = json.loads(str(message["value"]))
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not isinstance(value.get("arguments"), dict):
        raise ValueError(f"invalid function_call message: {message}")
    return value


def _format_call_target(call: Mapping[str, Any]) -> str:
    return f"Action: {call['name']}\nAction Input: {json.dumps(call['arguments'], ensure_ascii=False, sort_keys=True)}"


def _request_type_for_intent(intent: str) -> str:
    return {
        "damaged": "exchange",
        "wrong_item": "return_refund",
        "missing_item": "refund_only",
        "no_reason": "return_refund",
        "quality_issue": "return_refund",
    }[intent]


def _base_dpo_metadata(sft_row: Mapping[str, Any], level: str, primary_error: str) -> Dict[str, Any]:
    metadata = copy.deepcopy(sft_row["metadata"])
    base_id = metadata["sample_id"].replace("sft-v1-", "dpo-v1-")
    metadata["sample_id"] = f"{base_id}-{level}"
    metadata.update(
        {
            "preference_level": level,
            "primary_error": primary_error,
            "secondary_errors": [],
            "pair_source": "deterministic_counterfactual",
            "review_reason": f"single {level}-level business behavior contrast",
            "counterfactual_strength": "near_miss",
        }
    )
    return metadata


def _target_kind(target: str) -> str:
    return "action" if target.startswith("Action:") else "response"


def build_dpo_candidates(
    sft_row: Mapping[str, Any],
    reply_preference: Mapping[str, str],
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    conversations = list(sft_row["conversations"])
    call_indices = [index for index, message in enumerate(conversations) if message.get("from") == "function_call"]
    scenario = str(sft_row["metadata"]["scenario"])
    intent = str(sft_row["metadata"]["intent"])
    parent_id = str(sft_row["metadata"]["parent_id"])
    order_id = _order_id(parent_id)
    rows: List[Dict[str, Any]] = []

    reply_metadata = _base_dpo_metadata(sft_row, "response", reply_preference["primary_error"])
    reply_metadata["target_turn_index"] = len(conversations) - 1
    reply_metadata["chosen_target_kind"] = _target_kind(reply_preference["chosen"])
    reply_metadata["rejected_target_kind"] = _target_kind(reply_preference["rejected"])
    rows.append(
        {
            "conversations": copy.deepcopy(conversations[:-1]),
            "chosen": reply_preference["chosen"],
            "rejected": reply_preference["rejected"],
            "tools": tools,
            "metadata": reply_metadata,
        }
    )

    terminal_after_observation = {
        "order_not_found",
        "tool_timeout",
        "duplicate_request",
        "no_reason_expired",
        "not_delivered",
        "identity_required",
    }
    if not call_indices:
        fabricated = {"name": "query_order_status", "arguments": {"order_id": order_id}}
        decision_prompt = copy.deepcopy(conversations[:-1])
        decision_chosen = reply_preference["chosen"]
        decision_rejected = _format_call_target(fabricated)
        decision_error = "missing_argument"
        target_turn_index = len(conversations) - 1
    elif scenario in terminal_after_observation:
        decision_prompt = copy.deepcopy(conversations[:-1])
        rejected_call = {
            "name": "create_after_sales_request",
            "arguments": {
                "order_id": order_id,
                "request_type": _request_type_for_intent(intent),
                "reason": intent,
            },
        }
        decision_chosen = reply_preference["chosen"]
        decision_rejected = _format_call_target(rejected_call)
        decision_error = "unnecessary_tool"
        target_turn_index = len(conversations) - 1
    else:
        target_turn_index = call_indices[-1]
        correct_call = _parse_call_message(conversations[target_turn_index])
        decision_prompt = copy.deepcopy(conversations[:target_turn_index])
        repeated_policy = {
            "name": "check_return_policy",
            "arguments": {"order_id": order_id, "issue_type": intent},
        }
        decision_chosen = _format_call_target(correct_call)
        decision_rejected = _format_call_target(repeated_policy)
        decision_error = "unnecessary_tool"

    decision_metadata = _base_dpo_metadata(sft_row, "decision", decision_error)
    decision_metadata["target_turn_index"] = target_turn_index
    decision_metadata["chosen_target_kind"] = _target_kind(decision_chosen)
    decision_metadata["rejected_target_kind"] = _target_kind(decision_rejected)
    rows.append(
        {
            "conversations": decision_prompt,
            "chosen": decision_chosen,
            "rejected": decision_rejected,
            "tools": tools,
            "metadata": decision_metadata,
        }
    )

    for target_turn_index in call_indices[:-1]:
        correct_call = _parse_call_message(conversations[target_turn_index])
        continue_metadata = _base_dpo_metadata(sft_row, "decision", "premature_stop")
        continue_metadata["sample_id"] = f"{continue_metadata['sample_id']}-step-{target_turn_index}"
        continue_metadata["target_turn_index"] = target_turn_index
        continue_metadata["chosen_target_kind"] = "action"
        continue_metadata["rejected_target_kind"] = "response"
        rows.append(
            {
                "conversations": copy.deepcopy(conversations[:target_turn_index]),
                "chosen": _format_call_target(correct_call),
                "rejected": reply_preference["chosen"],
                "tools": tools,
                "metadata": continue_metadata,
            }
        )

    for target_turn_index in call_indices:
        correct_call = _parse_call_message(conversations[target_turn_index])
        wrong_call = copy.deepcopy(correct_call)
        if wrong_call["name"] == "create_after_sales_request":
            wrong_call["arguments"]["reason"] = "商品描述不是原因代码"
        elif wrong_call["name"] == "check_return_policy":
            wrong_call["arguments"]["issue_type"] = "RETURN_WINDOW_EXPIRED"
        else:
            wrong_call["arguments"] = {}
        parameter_metadata = _base_dpo_metadata(sft_row, "parameter", "invalid_argument")
        parameter_metadata["sample_id"] = f"{parameter_metadata['sample_id']}-step-{target_turn_index}"
        parameter_metadata["target_turn_index"] = target_turn_index
        parameter_metadata["chosen_target_kind"] = "action"
        parameter_metadata["rejected_target_kind"] = "action"
        rows.append(
            {
                "conversations": copy.deepcopy(conversations[:target_turn_index]),
                "chosen": _format_call_target(correct_call),
                "rejected": _format_call_target(wrong_call),
                "tools": tools,
                "metadata": parameter_metadata,
            }
        )
    return rows


def select_balanced_dpo(
    rows: Sequence[Tuple[str, Dict[str, Any]]], limit: int
) -> Tuple[List[Tuple[str, Dict[str, Any]]], int]:
    if limit < 1:
        raise ValueError("limit must be positive")
    targets = {
        "decision": int(limit * 0.40),
        "parameter": int(limit * 0.25),
    }
    targets["response"] = limit - targets["decision"] - targets["parameter"]
    accepted_all, removed = near_deduplicate(rows, max(len(rows), 1))
    accepted_by_level: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for level in ("decision", "parameter", "response"):
        accepted_by_level[level] = [
            item for item in accepted_all if item[1]["metadata"]["preference_level"] == level
        ]

    selected: List[Tuple[str, Dict[str, Any]]] = []
    leftovers: List[Tuple[str, Dict[str, Any]]] = []

    decision_rows = accepted_by_level["decision"]
    decision_action = [item for item in decision_rows if item[1]["metadata"]["chosen_target_kind"] == "action"]
    decision_response = [
        item for item in decision_rows if item[1]["metadata"]["chosen_target_kind"] == "response"
    ]
    decision_action_target = targets["decision"] // 2
    decision_response_target = targets["decision"] - decision_action_target
    selected.extend(decision_action[:decision_action_target])
    selected.extend(decision_response[:decision_response_target])
    leftovers.extend(decision_action[decision_action_target:])
    leftovers.extend(decision_response[decision_response_target:])

    for level in ("parameter", "response"):
        accepted = accepted_by_level[level]
        selected.extend(accepted[: targets[level]])
        leftovers.extend(accepted[targets[level] :])
    leftovers.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    selected.extend(leftovers[: max(0, limit - len(selected))])
    selected.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    return selected[:limit], removed


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _row_text(row: Mapping[str, Any]) -> str:
    parts = []
    for message in row.get("conversations", []):
        if isinstance(message, Mapping):
            parts.append(f"{message.get('from', '')}:{message.get('value', '')}")
    for field in ("chosen", "rejected"):
        if isinstance(row.get(field), str):
            parts.append(f"{field}:{row[field]}")
    return normalize_text("\n".join(parts))


def near_deduplicate(
    rows: Sequence[Tuple[str, Dict[str, Any]]], limit: int
) -> Tuple[List[Tuple[str, Dict[str, Any]]], int]:
    split_priority = {"test": 0, "validation": 1, "train": 2}
    ordered = sorted(
        rows,
        key=lambda item: (
            split_priority[item[0]],
            hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest(),
        ),
    )
    accepted: List[Tuple[str, Dict[str, Any], int, str]] = []
    buckets: Dict[Tuple[int, int], List[int]] = {}
    rejected = 0
    for split, row in ordered:
        text = _row_text(row)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        fingerprint = simhash64(text)
        candidates = set()
        for band in range(4):
            band_value = (fingerprint >> (band * 16)) & 0xFFFF
            candidates.update(buckets.get((band, band_value), []))
        duplicate = any(
            content_hash == accepted[index][3]
            or simhash_distance(fingerprint, accepted[index][2]) <= 3
            for index in candidates
        )
        if duplicate:
            rejected += 1
            continue
        accepted_index = len(accepted)
        accepted.append((split, row, fingerprint, content_hash))
        for band in range(4):
            band_value = (fingerprint >> (band * 16)) & 0xFFFF
            buckets.setdefault((band, band_value), []).append(accepted_index)
    selected = [(split, row) for split, row, _, _ in accepted]
    selected.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    return selected[:limit], rejected


def _round_robin_strata(
    rows: Sequence[Tuple[str, Dict[str, Any]]], limit: int
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Tuple[str, Dict[str, Any]]]]:
    buckets: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any]]]] = {}
    for item in rows:
        metadata = item[1]["metadata"]
        key = (str(metadata["scenario"]), str(metadata["source_id"]))
        buckets.setdefault(key, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    selected: List[Tuple[str, Dict[str, Any]]] = []
    keys = sorted(buckets)
    while len(selected) < limit and keys:
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    leftovers = [item for key in sorted(buckets) for item in buckets[key]]
    leftovers.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    return selected, leftovers


def select_stratified_sft(
    rows: Sequence[Tuple[str, Dict[str, Any]]], limit: int
) -> List[Tuple[str, Dict[str, Any]]]:
    available_splits = {split for split, _ in rows}
    if available_splits == {"train", "validation"}:
        targets = {"train": limit - int(limit * 0.10), "validation": int(limit * 0.10)}
    elif available_splits == {"train", "validation", "test"}:
        targets = {"train": int(limit * 0.80), "validation": int(limit * 0.10)}
        targets["test"] = limit - targets["train"] - targets["validation"]
    else:
        ordered_splits = sorted(available_splits)
        base = limit // max(len(ordered_splits), 1)
        targets = {split: base for split in ordered_splits}
        for split in ordered_splits[: limit - base * len(ordered_splits)]:
            targets[split] += 1

    selected: List[Tuple[str, Dict[str, Any]]] = []
    leftovers: List[Tuple[str, Dict[str, Any]]] = []
    for split in sorted(available_splits):
        split_rows = [item for item in rows if item[0] == split]
        split_selected, split_leftovers = _round_robin_strata(split_rows, targets.get(split, 0))
        selected.extend(split_selected)
        leftovers.extend(split_leftovers)
    leftovers.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    selected.extend(leftovers[: max(0, limit - len(selected))])
    selected.sort(key=lambda item: hashlib.sha256(item[1]["metadata"]["sample_id"].encode()).hexdigest())
    return selected[:limit]


def build_domain_pilot(
    public_pilot_root: Path,
    output_root: Path,
    sft_limit: int = 2000,
    dpo_limit: int = 800,
    source_ids: Sequence[str] | None = None,
    source_splits: Sequence[str] | None = None,
    variants_per_parent: int = 1,
) -> Dict[str, Any]:
    if variants_per_parent < 1:
        raise ValueError("variants_per_parent must be positive")
    all_evidence = load_evidence(
        public_pilot_root,
        source_ids=source_ids,
        source_splits=source_splits,
    )
    evidence = select_evidence(all_evidence, len(all_evidence))
    tools = _load_json(CONFIG_DIR / "tools_v1.json")["tools"]
    raw_sft_candidates: List[Tuple[str, Dict[str, Any]]] = []
    preference_by_sample: Dict[str, Dict[str, str]] = {}
    for source_row in evidence:
        for variant in range(variants_per_parent):
            sft_row, preference = build_sft_row(source_row, tools, variant=variant)
            split = str(source_row["split"])
            raw_sft_candidates.append((split, sft_row))
            preference_by_sample[sft_row["metadata"]["sample_id"]] = preference

    deduplicated_sft, sft_near_duplicates_removed = near_deduplicate(
        raw_sft_candidates, max(len(raw_sft_candidates), 1)
    )
    selected_sft = select_stratified_sft(deduplicated_sft, sft_limit)
    sft_by_split: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    raw_dpo_candidates: List[Tuple[str, Dict[str, Any]]] = []
    for split, sft_row in selected_sft:
        sft_by_split[split].append(sft_row)
        preference = preference_by_sample[sft_row["metadata"]["sample_id"]]
        raw_dpo_candidates.extend(
            (split, row) for row in build_dpo_candidates(sft_row, preference, tools)
        )

    ranked_dpo, dpo_near_duplicates_removed = select_balanced_dpo(raw_dpo_candidates, dpo_limit)
    dpo_by_split: Dict[str, List[Dict[str, Any]]] = {name: [] for name in sft_by_split}
    for split, row in ranked_dpo:
        dpo_by_split[split].append(row)

    for split, rows in sft_by_split.items():
        _write_jsonl(output_root / "sft" / split / "data.jsonl", rows)
    for split, rows in dpo_by_split.items():
        _write_jsonl(output_root / "dpo" / split / "data.jsonl", rows)

    audits = {}
    for task in ("sft", "dpo"):
        report, issues = audit_dataset(output_root / task, require_metadata=True)
        write_audit(output_root / "audits" / task, report, issues)
        audits[task] = report
        if not report["passed"]:
            raise ValueError(f"{task} audit failed: {report['issue_counts']}")
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "requested_source_ids": sorted(source_ids or []),
        "requested_source_splits": sorted(source_splits or []),
        "variants_per_parent": variants_per_parent,
        "raw_sft_candidates": len(raw_sft_candidates),
        "sft_rows": sum(len(rows) for rows in sft_by_split.values()),
        "dpo_rows": sum(len(rows) for rows in dpo_by_split.values()),
        "evidence_rows_considered": len(evidence),
        "sft_near_duplicates_removed": sft_near_duplicates_removed,
        "dpo_near_duplicates_removed": dpo_near_duplicates_removed,
        "sft_split_counts": {split: len(rows) for split, rows in sft_by_split.items()},
        "dpo_split_counts": {split: len(rows) for split, rows in dpo_by_split.items()},
        "dpo_preference_level_counts": dict(
            sorted(
                Counter(
                    row["metadata"]["preference_level"]
                    for rows in dpo_by_split.values()
                    for row in rows
                ).items()
            )
        ),
        "dpo_primary_error_counts": dict(
            sorted(
                Counter(
                    row["metadata"]["primary_error"]
                    for rows in dpo_by_split.values()
                    for row in rows
                ).items()
            )
        ),
        "dpo_decision_chosen_target_kind_counts": dict(
            sorted(
                Counter(
                    row["metadata"]["chosen_target_kind"]
                    for rows in dpo_by_split.values()
                    for row in rows
                    if row["metadata"]["preference_level"] == "decision"
                ).items()
            )
        ),
        "source_counts": dict(
            sorted(
                Counter(
                    row["metadata"]["source_id"]
                    for rows in sft_by_split.values()
                    for row in rows
                ).items()
            )
        ),
        "scenario_counts": dict(
            sorted(Counter(row["metadata"]["scenario"] for rows in sft_by_split.values() for row in rows).items())
        ),
        "audits": {task: {"content_set_sha256": report["content_set_sha256"], "passed": report["passed"]} for task, report in audits.items()},
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ecommerce SFT/DPO pilot from public evidence.")
    parser.add_argument("--public-pilot-root", type=Path, default=ROOT / "data" / "ecommerce" / "public_pilot")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "ecommerce" / "domain_pilot_v1")
    parser.add_argument("--sft-limit", type=int, default=2000)
    parser.add_argument("--dpo-limit", type=int, default=800)
    parser.add_argument("--source-id", action="append", dest="source_ids")
    parser.add_argument("--variants-per-parent", type=int, default=1)
    parser.add_argument(
        "--source-split",
        action="append",
        dest="source_splits",
        choices=("train", "validation", "test"),
    )
    args = parser.parse_args()
    manifest = build_domain_pilot(
        args.public_pilot_root,
        args.output_root,
        args.sft_limit,
        args.dpo_limit,
        source_ids=args.source_ids,
        source_splits=args.source_splits,
        variants_per_parent=args.variants_per_parent,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
