# -*- coding: utf-8 -*-
"""
exploration.py —— Phase 1: Exploration

【论文对应】Sec.3.1 Exploration 段落：
    "the Exploration stage prompts the current student model pi_theta_t to generate
     a preliminary response y_hat_i ~ pi_theta_t(·|x_i) for each query x_i.
     To further encourage detailed reasoning traces, we utilize the CoT system prompt,
     forcing the model to explain its internal logic and thereby making latent
     reasoning errors observable."

【设计意图】
这一步的目的不是"评估模型好不好"，而是主动"引诱"模型暴露自己的错误模式，
为下一步 Diagnosis 提供原始素材。因此这里只拿 (query, 学生的完整回答)，
不在这一步判断对错——判断对错是 Audit Agent 的职责（关注点分离，见 diagnosis.py）。
"""

from dataclasses import dataclass
from typing import List

from student_model import MockStudentClient, STUDENT_SYSTEM_PROMPT, _build_student_user_prompt


@dataclass
class ExplorationResult:
    sample: dict  # 原始样本 (id, contract, question, answer, trap_type, split)
    response: str  # 学生的完整回答（含推理与 "Final Answer: Yes/No"）


def run_exploration(student: MockStudentClient, samples: List[dict]) -> List[ExplorationResult]:
    """对一批 legal query 跑一遍学生模型，收集初步回答。"""
    results = []
    for sample in samples:
        user_prompt = _build_student_user_prompt(sample["contract"], sample["question"])
        meta = {
            "trap_type": sample["trap_type"],
            "ground_truth": sample["answer"],
            "contract": sample["contract"],
            "question": sample["question"],
        }
        response = student.generate(STUDENT_SYSTEM_PROMPT, user_prompt, meta=meta)
        results.append(ExplorationResult(sample=sample, response=response))
    return results
