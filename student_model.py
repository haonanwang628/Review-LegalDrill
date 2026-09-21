# -*- coding: utf-8 -*-
"""
student_model.py —— 学生模型 (SLM) 的 Mock 实现

【论文对应】
- Sec.2.1 Legal Task Formulation：学生模型 pi_theta_t 接收 x=(c,q)，输出 y（含推理与结论）
- Sec.3.1 Exploration：y_hat_i ~ pi_theta_t(·|x_i)，配合 CoT system prompt 暴露潜在推理错误
- Sec.3.2 Self-Reflective Verification：
      s_theta_t(y|x) = pi(correct|Pver) / (pi(correct|Pver) + pi(incorrect|Pver))
  这是"强制二选一打分"，不是对整段话做似然打分（论文特别强调这一点，
  因为整段似然对长度、表面形式敏感，见 Sec.3.2 第二段）。
- Sec.3.3：SFT / DPO 更新的对象就是这个 pi_theta_t

【核心简化：用"每类错误的困惑度权重"代替真实的神经网络参数】
真实论文里，学生是一个 0.6B/1.7B 参数的 Transformer，SFT/DPO 都是对着几十万个参数
做梯度下降。我们完全无法在离线、无 GPU 的环境里复现这一点。
退而求其次，我们把学生的"能力"压缩成一个极简的状态：
    confusion_weights: Dict[trap_type -> float]   # 取值范围 [0, 1]
含义：confusion_weights[T] 越大，代表学生在遇到"错误类型 T"时，
      越容易被误导、越容易给出错误答案，也越容易在强制二选一时误判"错误推理"为 correct。
SFT / DPO 训练的效果，就体现为这些标量被有方向地调低（见 optimizer.py）。

这样做的好处：能完整跑通论文的算法流程（Exploration -> Diagnosis -> Targeted
Generation -> Verification -> SFT/DPO -> 下一轮迭代），并能定量看到"困惑度随迭代下降"
这一符合论文定性结论的现象。
代价：这个"学生"不具备任何真实的语言理解能力，它的输出完全是模板拼接，
无法处理训练数据之外的、真正需要新颖理解的合同文本；confusion_weight 的数值
更新公式也是我们类比出来的，不等价于真实的梯度下降，因此不能用来说明
论文报告的具体准确率数字（如 Table 1 里的 0.84/0.91 等）。
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

import config
from llm_client import LLMClient
from mock_texts import TRAP_BANK, opposite_answer

# 学生模型的初始"困惑度"：数值越高，代表基座 SLM 在这类错误上越薄弱。
# 这几个初始值是我们主观设定的、用于制造"有提升空间"的教学效果，并非来自论文的真实测量。
INITIAL_CONFUSION: Dict[str, float] = {
    "missing_condition": 0.80,
    "logical_leap": 0.70,
    "term_misreading": 0.75,
    "scope_overgeneralization": 0.65,
}

STUDENT_SYSTEM_PROMPT = """\
系统角色：你是一名专注于法律合同分析的 AI 助手。请仔细分析合同，引用具体条款，
并给出分步推理。
任务要求：
1. 引用：明确指出并引用与问题相关的条款/术语。
2. 分步推理（Chain-of-Thought）：对每一步，先复述/概括相关条款内容，再解释其对结论的逻辑影响。
3. 最终答案：另起一行，以 'Final Answer: Yes' 或 'Final Answer: No' 结束。
"""


def _build_student_user_prompt(contract: str, question: str) -> str:
    return f"合同：{contract}\n问题：{question}"


@dataclass
class MockStudentClient(LLMClient):
    """Mock 版本的可训练学生模型。"""

    confusion_weights: Dict[str, float] = field(
        default_factory=lambda: dict(INITIAL_CONFUSION)
    )
    rng: np.random.RandomState = field(default_factory=lambda: config.RNG)

    # ------------------------------------------------------------------
    # Exploration 阶段：对应 Sec.3.1 "y_hat_i ~ pi_theta_t(·|x_i)"
    # ------------------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str, meta: Optional[dict] = None) -> str:
        meta = meta or {}
        trap_type = meta["trap_type"]
        ground_truth = meta["ground_truth"]
        contract = meta["contract"]
        question = meta["question"]

        p_wrong = self.confusion_weights[trap_type]
        is_wrong = self.rng.rand() < p_wrong

        bank = TRAP_BANK[trap_type]
        if is_wrong:
            wrong_answer = opposite_answer(ground_truth)
            text = bank["flawed_reasoning"].format(
                contract=contract, question=question, wrong_answer=wrong_answer
            )
        else:
            text = bank["chosen_reasoning"].format(
                contract=contract, question=question, correct_answer=ground_truth
            )
        return text

    # ------------------------------------------------------------------
    # Self-Reflective Verification 阶段：对应 Sec.3.2 s_theta_t(y|x)
    # ------------------------------------------------------------------
    def score_choices(
        self,
        system_prompt: str,
        user_prompt: str,
        choices: Tuple[str, str] = ("correct", "incorrect"),
        meta: Optional[dict] = None,
    ) -> Dict[str, float]:
        """强制二选一打分：给定候选回复 y，返回 {"correct": s, "incorrect": 1-s}。

        meta 需要包含：
            trap_type : 该候选回复所对应的错误类型
            is_flawed : True 表示 y 是体现该错误的 y-（rejected），
                        False 表示 y 是纠正该错误的 y+（chosen）
        """
        meta = meta or {}
        trap_type = meta["trap_type"]
        is_flawed = meta["is_flawed"]
        c = self.confusion_weights[trap_type]

        if is_flawed:
            # 困惑度越高，越容易把"体现错误的推理"误判为 correct
            raw_correct = 0.5 + 0.4 * c
        else:
            # 困惑度越高，学生对"真正正确的纠正推理"也越缺乏信心
            raw_correct = 0.9 - 0.4 * c

        raw_correct = float(np.clip(raw_correct, 0.05, 0.95))
        raw_incorrect = 1.0 - raw_correct  # Mock 里两者按构造互补

        # 显式做一次归一化，呼应论文公式 s = pi(correct)/(pi(correct)+pi(incorrect))，
        # 即便在这里分母恒为 1，也保留这一步以体现该公式的含义。
        denom = raw_correct + raw_incorrect
        return {"correct": raw_correct / denom, "incorrect": raw_incorrect / denom}

    # ------------------------------------------------------------------
    # 供 optimizer.py 调用的底层状态更新入口
    # ------------------------------------------------------------------
    def nudge_confusion(self, trap_type: str, delta: float) -> None:
        """将 confusion_weights[trap_type] 调整 delta（可正可负），并裁剪到 [0,1]。"""
        new_val = self.confusion_weights[trap_type] + delta
        self.confusion_weights[trap_type] = float(np.clip(new_val, 0.0, 1.0))

    def snapshot(self) -> Dict[str, float]:
        return dict(self.confusion_weights)
