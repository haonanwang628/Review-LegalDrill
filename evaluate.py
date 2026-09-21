# -*- coding: utf-8 -*-
"""
evaluate.py —— 评估指标：Accuracy / F1 / Judge Accuracy

【论文对应】
- Sec.4.1 Metrics: "We report accuracy and F1 to evaluate each model's judgment
  performance on the binary QA tasks. In addition, we introduce a 'judge accuracy'
  metric ... using an LLM as a Judge."
- Appendix A.1 Judge Accuracy: "the judge checks whether the reasoning contains
  any potential error; if so, the entire reasoning is marked as incorrect."
  （附录里用 Qwen3-8B 做裁判；这里用 MockLLMClient 的 judge 角色代替）

【本项目里 Judge Accuracy 会退化得很像 Accuracy，这是一处已知的模拟损失】
论文引入 Judge Accuracy 是为了抓住"最终答案蒙对了，但推理过程其实有错"这种情况——
真实 LLM 能读懂推理文本本身是否站得住脚，而不只是看最后的 Yes/No。
但我们的 Mock 学生模型只有两种"人格模板"：完全体现某个陷阱的错误推理，
或完全正确、简洁的推理，二者与 Final Answer 是否等于 ground_truth 强绑定，
不存在"结论蒙对但推理过程有独立错误"的中间态。因此在本项目里，
Judge Accuracy 基本等价于 Accuracy，这一现象本身就说明了 Judge Accuracy
这个指标的价值——它是用来检测"表面正确、内在有问题"的场景，
而这恰恰是我们的 Mock 无法忠实建模的部分。
"""

import json
from dataclasses import dataclass
from typing import List

from llm_client import LLMClient, JUDGE_SYSTEM_PROMPT, _build_judge_user_prompt, _extract_final_answer
from student_model import MockStudentClient, STUDENT_SYSTEM_PROMPT, _build_student_user_prompt


@dataclass
class EvalRecord:
    sample_id: str
    ground_truth: str
    predicted: str
    correct: bool
    judge_verdict: str
    response: str


@dataclass
class EvalReport:
    accuracy: float
    f1: float
    judge_accuracy: float
    n: int
    records: List[EvalRecord]


def evaluate(
    student: MockStudentClient, judge_client: LLMClient, eval_samples: List[dict]
) -> EvalReport:
    records: List[EvalRecord] = []

    for sample in eval_samples:
        user_prompt = _build_student_user_prompt(sample["contract"], sample["question"])
        meta = {
            "trap_type": sample["trap_type"],
            "ground_truth": sample["answer"],
            "contract": sample["contract"],
            "question": sample["question"],
        }
        response = student.generate(STUDENT_SYSTEM_PROMPT, user_prompt, meta=meta)
        predicted = _extract_final_answer(response) or "no"
        ground_truth = sample["answer"].strip().lower()
        is_correct = predicted == ground_truth

        judge_user_prompt = _build_judge_user_prompt(
            sample["contract"], sample["question"], sample["answer"], response
        )
        judge_verdict = judge_client.generate(
            JUDGE_SYSTEM_PROMPT,
            judge_user_prompt,
            meta={
                "role": "judge",
                "ground_truth": sample["answer"],
                "model_generation": response,
            },
        )

        records.append(
            EvalRecord(
                sample_id=sample["id"],
                ground_truth=ground_truth,
                predicted=predicted,
                correct=is_correct,
                judge_verdict=judge_verdict,
                response=response,
            )
        )

    n = len(records)
    accuracy = sum(r.correct for r in records) / n if n else 0.0
    judge_accuracy = sum(r.judge_verdict == "correct" for r in records) / n if n else 0.0
    f1 = _binary_f1(records, positive_label="yes")

    return EvalReport(accuracy=accuracy, f1=f1, judge_accuracy=judge_accuracy, n=n, records=records)


def _binary_f1(records: List[EvalRecord], positive_label: str = "yes") -> float:
    """二分类 F1，positive class 固定取 'yes'。
    样本量极小（个位数）时，F1 只有示意意义，不具备统计显著性，
    这一点在最终"批判性分析"步骤中还会再强调一次。
    """
    tp = sum(1 for r in records if r.predicted == positive_label and r.ground_truth == positive_label)
    fp = sum(1 for r in records if r.predicted == positive_label and r.ground_truth != positive_label)
    fn = sum(1 for r in records if r.predicted != positive_label and r.ground_truth == positive_label)

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
