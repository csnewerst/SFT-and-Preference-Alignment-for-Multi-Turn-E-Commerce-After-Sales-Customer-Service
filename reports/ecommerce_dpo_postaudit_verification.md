# 1.5B DPO 审计修复验证记录

## 审计结论

- TRL 0.19.2 在 `PeftModel + ref_model=None` 且未使用 `target_parameters` 时，会把已加载的 SFT adapter 复制为冻结的 `ref` adapter；本项目不是以原始 Base 作为 DPO reference。
- 历史 DPO run 的 `training_args.bin` 显示实际 `weight_decay=0.0`，确认旧训练器漏传该参数。旧结果只保留为负面诊断，不作为修复后配方结果。
- 600 条集合已参与 SFT parent 选择，定性为 development gate。最终 7B 四模型对比必须使用在配置冻结后一次性打开的新 formal test v2，并在同一批样本上做 paired bootstrap。
- 1.5B 仅用于链路校验和淘汰失败方案；7B 需对胜出的 1 至 2 套配置短程校准。默认不增加 3B，除非 1.5B 与 7B 的方向明显矛盾。

## 修复后实测

验证 run：`dpo-preflight1-response-beta0p1-seed42-from-r4-all-postaudit-v1`

- Git commit：`036208af84a35d586bec73a428c01532e653953c`
- 数据方案：`response_only_matched`
- SFT parent：`sft-r4-all-full-seed42-v1`
- seed：42
- beta：0.1
- 训练预算：1 optimizer step（只验证首步前 reference 与落盘参数，不用于效果结论）
- 实际落盘参数：`weight_decay=0.01`、`warmup_steps=1`、`warmup_ratio=None`、`max_steps=1`
- reference adapter：`ref`；可训练参数：0
- 首步前 policy/reference chosen log-prob 最大绝对差：0.0
- 首步前 policy/reference rejected log-prob 最大绝对差：0.0
- 容差：1e-4；前检状态：passed
- train loss：0.693147
- eval loss：0.693147；eval samples：179

该 run 证明 reference 语义、weight decay 传参和训练/评测链路已经按审计要求工作；它不证明 DPO 带来业务指标提升。下一阶段仍需用新的 run ID 执行有界 1.5B DPO 筛选，并用相同 rollout evaluator 比较 SFT 与 DPO。
