# -*- coding: utf-8 -*-
"""
targeted_generation.py —— Phase 3: Targeted Generation（偏好对合成）

【论文对应】Sec.3.1 Targeted Generation 段落，核心公式：
    对每个训练样本 x，从 Error Instruction Bank 中采样 K 条指令 {I_k}_{k=1}^K；
    对每条指令 I_k，教师模型两步生成：
        y_-^(k) ~ pi_teach(· | x, I_k)                （先生成 rejected）
        y_+^(k) ~ pi_teach(· | x, I_k, y_-^(k))        （再基于 rejected 生成 chosen）
    最终得到合成数据集：
        D^t_syn = {(x, y_+^(k), y_-^(k)) | x ∈ D, k = 1,...,K}

【设计意图：为什么要"先生成错的，再基于错的生成对的"】
论文强调这个两步顺序（而不是同时/独立生成两者）是为了让 chosen 更有针对性——
教师模型能明确看到"学生大概会怎么错"，然后专门纠正这个具体的错误点，
而不是泛泛地给出一个标准答案。这也是"diagnosis-driven"名字的由来。

【设计意图：为什么 instruction 可以来自 Bank 里的任意样本，而不必是 x 自己的错误】
因为 reproduction_instruction 被约束为 context-agnostic（见 diagnosis.py），
同一条错误指令可以套用到任意上下文 x 上。这正是论文所说的"combinatorial expansion"：
不再受限于原始数据集大小，而是可以用 |D| x K 的规模合成偏好对。
"""

from dataclasses import dataclass
from typing import List

import config
from diagnosis import ErrorInstruction
from llm_client import (
    LLMClient,
    TEACHER_CHOSEN_SYSTEM_PROMPT,
    TEACHER_REJECTED_SYSTEM_PROMPT,
    _build_teacher_chosen_user_prompt,
    _build_teacher_rejected_user_prompt,
)


@dataclass
class PreferencePair:
    sample: dict                 # 应用指令的目标上下文 x（可以和指令的原始来源样本不同）
    instruction: ErrorInstruction
    chosen: str                  # y_+^(k)
    rejected: str                # y_-^(k)

    @property
    def trap_type(self) -> str:
        return self.instruction.trap_type


def run_targeted_generation(
    teacher_client: LLMClient,
    train_samples: List[dict],
    error_bank: List[ErrorInstruction],
    k: int = config.K_ERROR_INSTRUCTIONS,
) -> List[PreferencePair]:
    """对每个训练样本 x，从 error_bank 采样 K 条指令，合成 K 个偏好对。"""
    if not error_bank:
        return []

    pairs: List[PreferencePair] = []
    for sample in train_samples:
        # 有放回采样：Bank 大小可能小于 K，且论文允许同一指令被复用到不同上下文
        idxs = config.RNG.randint(0, len(error_bank), size=k)
        sampled_instructions = [error_bank[i] for i in idxs]

        for instruction in sampled_instructions:
            rejected_meta = {
                "role": "teacher_rejected",
                "trap_type": instruction.trap_type,
                "contract": sample["contract"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
            }
            rejected_user_prompt = _build_teacher_rejected_user_prompt(
                contract=sample["contract"],
                question=sample["question"],
                ground_truth=sample["answer"],
                error_types=instruction.error_types,
                generic_summary=instruction.generic_summary,
                reproduction_instruction=instruction.reproduction_instruction,
            )
            y_minus = teacher_client.generate(
                TEACHER_REJECTED_SYSTEM_PROMPT, rejected_user_prompt, meta=rejected_meta
            )

            chosen_meta = {
                "role": "teacher_chosen",
                "trap_type": instruction.trap_type,
                "contract": sample["contract"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
            }
            chosen_user_prompt = _build_teacher_chosen_user_prompt(
                contract=sample["contract"],
                question=sample["question"],
                ground_truth=sample["answer"],
                rejected_response=y_minus,
            )
            y_plus = teacher_client.generate(
                TEACHER_CHOSEN_SYSTEM_PROMPT, chosen_user_prompt, meta=chosen_meta
            )

            pairs.append(
                PreferencePair(sample=sample, instruction=instruction, chosen=y_plus, rejected=y_minus)
            )
    return pairs
