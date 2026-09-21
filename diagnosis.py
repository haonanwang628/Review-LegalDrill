# -*- coding: utf-8 -*-
"""
diagnosis.py —— Phase 2: Diagnosis / Audit Agent / Error Instruction Bank

【论文对应】Sec.3.1 Diagnosis 段落：
    - Audit Agent: I^(i) ~ pi_audit(· | AgentPrompt(x_i, y_hat_i))
    - "we constrain the Agent to generate context-agnostic instructions ...
       This decoupling allows us to compile a reusable Error Instruction Bank,
       denoted as Phi_err = {I^(1), ..., I^(N)}"

【一处论文含糊点，以及我们采用的理解】
论文正文没有明确说明 Error Instruction Bank Phi_err 是"每轮迭代重新构建"，
还是"跨迭代持续累积"。Sec.3.1 把它描述成"at each iteration t"流程的一环，
读起来更像是每轮基于当轮 Exploration 结果重新构建；但同一节 Targeted Generation
部分又说"decoupling error types from contexts enables combinatorial expansion:
we can generate arbitrarily many training pairs by recombining contexts with
diverse error instructions"，这更像是在暗示 Bank 应该越攒越大，才谈得上"组合爆炸"。
本实现采用更贴近"combinatorial expansion"表述的理解：**Bank 跨迭代持续累积、
不清空**（见 pipeline.py 中 Phi_err 的生命周期）。这是我们的推断，不是论文明文规定，
如果你重新读论文后有不同理解，只需修改 pipeline.py 里 Phi_err 的重置逻辑即可。
"""

import json
from dataclasses import dataclass
from typing import List

from exploration import ExplorationResult
from llm_client import LLMClient, _extract_final_answer


@dataclass
class ErrorInstruction:
    trap_type: str
    error_types: list
    generic_summary: str
    reproduction_instruction: str
    source_sample_id: str  # 便于追溯这条指令最初是从哪个样本的错误中诊断出来的


def run_diagnosis(
    audit_client: LLMClient, exploration_results: List[ExplorationResult]
) -> List[ErrorInstruction]:
    """对每条学生回答跑 Audit Agent，收集非平凡（即学生确实答错）的错误指令。"""
    bank: List[ErrorInstruction] = []
    for res in exploration_results:
        sample = res.sample
        student_answer = _extract_final_answer(res.response) or ""
        meta = {
            "role": "audit",
            "student_answer": student_answer,
            "ground_truth": sample["answer"],
            "trap_type": sample["trap_type"],
        }
        raw = audit_client.generate(system_prompt="", user_prompt="", meta=meta)
        diagnosis = json.loads(raw)

        if diagnosis["status"] == "CORRECT":
            continue  # 学生已经答对，不产生错误指令（无需诊断）

        bank.append(
            ErrorInstruction(
                trap_type=sample["trap_type"],
                error_types=diagnosis["error_types"],
                generic_summary=diagnosis["generic_summary"],
                reproduction_instruction=diagnosis["reproduction_instruction"],
                source_sample_id=sample["id"],
            )
        )
    return bank
