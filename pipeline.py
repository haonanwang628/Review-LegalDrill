# -*- coding: utf-8 -*-
"""
pipeline.py —— 端到端主流程，串联 Phase 1-5，对应 Fig.1 的完整循环

【论文对应】
    Fig.1: Phase1 Exploration -> Phase2 Diagnosis -> Phase3 Targeted Generation
           -> (Sec.3.2 Self-Reflective Verification，插在 Phase3 与 Phase4 之间)
           -> Phase4 Model Optimization: t=0 时先 SFT 再 DPO；t>=1 时只做 DPO
    Sec.2.2 Iterative Preference Optimization：训练跑 T 轮，每轮用当前策略
           pi_theta_t 重新生成偏好数据 D_t，更新得到 pi_theta_{t+1}
"""

import json

import config
from diagnosis import ErrorInstruction, run_diagnosis
from evaluate import evaluate
from exploration import run_exploration
from llm_client import _extract_final_answer, get_audit_client, get_judge_client, get_teacher_client
from optimizer import dpo_step, sft_step
from student_model import MockStudentClient
from targeted_generation import run_targeted_generation
from verification import run_verification


def load_dataset(path: str = config.DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    train = [s for s in samples if s["split"] == "train"]
    eval_set = [s for s in samples if s["split"] == "eval"]
    return train, eval_set


def _is_wrong(exploration_result) -> bool:
    predicted = _extract_final_answer(exploration_result.response)
    return predicted != exploration_result.sample["answer"].strip().lower()


def print_confusion(title: str, confusion: dict) -> None:
    print("  " + title + ": " + ", ".join(f"{k}={v:.3f}" for k, v in confusion.items()))


def run_pipeline(verbose_demo: bool = True):
    train_samples, eval_samples = load_dataset()

    student = MockStudentClient()
    teacher_client = get_teacher_client()
    audit_client = get_audit_client()
    judge_client = get_judge_client()

    print("=" * 70)
    print("LegalDrill Mock 复现 —— 端到端流程")
    print("=" * 70)

    print("\n[初始状态] 训练前先在 eval 集上做一次基线评估：")
    baseline = evaluate(student, judge_client, eval_samples)
    print(f"  Accuracy={baseline.accuracy:.3f}  F1={baseline.f1:.3f}  JudgeAcc={baseline.judge_accuracy:.3f}")
    print_confusion("初始 confusion_weights", student.snapshot())

    # Error Instruction Bank Phi_err：跨迭代累积（见 diagnosis.py 顶部关于这一点的说明与不确定性标注）
    error_bank: "list[ErrorInstruction]" = []

    for t in range(config.T_ITERATIONS):
        print(f"\n{'-' * 70}\n迭代 t={t}\n{'-' * 70}")

        # ---- Phase 1: Exploration ----
        exploration_results = run_exploration(student, train_samples)
        n_wrong = sum(1 for r in exploration_results if _is_wrong(r))
        print(f"[Phase1 Exploration] 学生在 {len(train_samples)} 条训练样本上，答错 {n_wrong} 条")

        # ---- Phase 2: Diagnosis ----
        new_instructions = run_diagnosis(audit_client, exploration_results)
        error_bank.extend(new_instructions)
        print(f"[Phase2 Diagnosis] 本轮新增 {len(new_instructions)} 条错误指令，Bank 累计 {len(error_bank)} 条")

        if verbose_demo and t == 0 and new_instructions:
            demo = new_instructions[0]
            print("  —— 示例诊断 ——")
            print(f"  来源样本: {demo.source_sample_id}  错误类型: {demo.error_types}")
            print(f"  Generic Summary: {demo.generic_summary}")
            print(f"  Reproduction Instruction: {demo.reproduction_instruction}")

        if not error_bank:
            print("  Bank 为空（学生全部答对），跳过本轮训练")
            continue

        # ---- Phase 3: Targeted Generation ----
        pairs = run_targeted_generation(
            teacher_client, train_samples, error_bank, k=config.K_ERROR_INSTRUCTIONS
        )
        print(
            f"[Phase3 Targeted Generation] 合成 {len(pairs)} 个偏好对 "
            f"(|train|={len(train_samples)} x K={config.K_ERROR_INSTRUCTIONS})"
        )

        if verbose_demo and t == 0 and pairs:
            demo_pair = pairs[0]
            print("  —— 示例偏好对 ——")
            print(f"  应用到样本: {demo_pair.sample['id']}  指令来自错误类型: {demo_pair.trap_type}")
            print(f"  [Rejected]\n    {demo_pair.rejected}")
            print(f"  [Chosen]\n    {demo_pair.chosen}")

        # ---- Phase 4: Self-Reflective Verification ----
        verified = run_verification(student, pairs, tau=config.DS_THRESHOLD)
        print(f"[Phase4 Verification] DS > {config.DS_THRESHOLD} 的高价值样本：{len(verified)}/{len(pairs)}")
        if verified:
            avg_ds = sum(v.ds for v in verified) / len(verified)
            print(f"  被保留样本的平均 Difficulty Score = {avg_ds:.3f}")

        if not verified:
            print("  没有样本通过筛选（学生已经很难被这批错误迷惑），跳过本轮训练")
            continue

        # ---- Phase 5: Optimization ----
        if t == 0:
            sft_metrics = sft_step(student, verified)
            print(f"[Phase5 SFT ] avg_proxy_loss={sft_metrics.avg_proxy_loss:.3f}  n={sft_metrics.n_samples}")
        dpo_metrics = dpo_step(student, verified)
        print(f"[Phase5 DPO ] avg_proxy_loss={dpo_metrics.avg_proxy_loss:.3f}  n={dpo_metrics.n_samples}")
        print_confusion("更新后 confusion_weights", student.snapshot())

        report = evaluate(student, judge_client, eval_samples)
        print(f"[Eval] Accuracy={report.accuracy:.3f}  F1={report.f1:.3f}  JudgeAcc={report.judge_accuracy:.3f}")

    print("\n" + "=" * 70)
    print("训练结束，最终评估：")
    final_report = evaluate(student, judge_client, eval_samples)
    print(f"  Accuracy={final_report.accuracy:.3f}  F1={final_report.f1:.3f}  JudgeAcc={final_report.judge_accuracy:.3f}")
    print_confusion("最终 confusion_weights", student.snapshot())
    return student, final_report


if __name__ == "__main__":
    run_pipeline()
