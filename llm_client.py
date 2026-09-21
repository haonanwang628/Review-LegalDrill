# -*- coding: utf-8 -*-
"""
llm_client.py —— 统一 LLM 调用接口 + Mock 实现（Teacher / Audit Agent / Judge）

【论文对应】
- Sec.3.1 Diagnosis: Audit Agent  I^(i) ~ pi_audit(· | AgentPrompt(x_i, y_hat_i))
- Sec.3.1 Targeted Generation: Teacher 两步生成
      y_-^(k) ~ pi_teach(· | x, I_k)                (先生成 rejected)
      y_+^(k) ~ pi_teach(· | x, I_k, y_-^(k))        (再基于 rejected 生成 chosen)
- Appendix A.1 / D.4 Judge：LLM-as-a-Judge，只输出 correct / incorrect
- Appendix D.1 / D.2 / D.4：Audit Agent / Teacher / Judge 的 System Prompt 与 User Prompt 模板

【接口设计意图】
LLMClient 是一个"与后端无关"的调用契约：只要实现了 generate()（和可选的 score_choices()），
上层的 diagnosis.py / targeted_generation.py / verification.py 完全不需要关心
背后到底是 MockLLMClient、OpenAI API、还是本地 vLLM 部署的 Qwen3。
真实迁移时，只需要在 config.py 里把 USE_MOCK_BACKEND 改成 False，
并把 get_teacher_client() / get_audit_client() / get_judge_client() 换成
一个真正发 HTTP 请求或调用本地推理引擎的 XXXClient 实现即可，
diagnosis.py / targeted_generation.py 等业务逻辑代码一行都不用改。

【为什么 system_prompt / user_prompt 与 meta 分开传】
system_prompt / user_prompt 是"如果这是一次真实 API 调用，会发送出去的完整文本"，
完全照抄论文附录 D 的模板结构；meta 是"Mock 后端为了确定性模拟而需要的结构化信息"
（比如 ground_truth、trap_type 等，真实 LLM 当然不需要这些，它是靠读懂 prompt 文本
自己推理出来的）。这样设计保证了：
    1) 换成真实 API 时，只需要发送 system_prompt+user_prompt，meta 直接丢弃即可；
    2) Mock 后端可以在不具备真实语言理解能力的前提下，依然产出结构化、可控的输出。
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from mock_texts import TRAP_BANK, opposite_answer


class LLMClient(ABC):
    """统一 LLM 调用接口。"""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        meta: Optional[dict] = None,
    ) -> str:
        """给定 system/user prompt，返回模型的文本输出。

        meta: Mock 后端用来"作弊"实现确定性输出的结构化信息；
              真实后端应当忽略此参数（或仅用于日志/调试）。
        """
        raise NotImplementedError

    def score_choices(
        self,
        system_prompt: str,
        user_prompt: str,
        choices: Tuple[str, str] = ("correct", "incorrect"),
        meta: Optional[dict] = None,
    ) -> Dict[str, float]:
        """强制二选一打分接口，对应 Sec.3.2 的 s_theta(y|x) 计算。

        真实后端应当通过获取 choices 两个 token 的 logprob 并做归一化来实现
        （对应论文公式：对 {correct, incorrect} 两个词的概率做归一化）。
        默认不是所有角色都需要这个能力（教师/审计/裁判不需要），
        因此基类直接抛错，只有 student_model.py 里的 MockStudentClient 会重写它。
        """
        raise NotImplementedError("该角色不支持 score_choices（仅学生模型需要）")


# ---------------------------------------------------------------------------
# Prompt 模板（照抄/翻译自论文 Appendix D，用于展示"如果接真实API会发送什么"）
# ---------------------------------------------------------------------------

AUDIT_AGENT_SYSTEM_PROMPT = """\
角色：你是一名严格的法律合同分析助教（Audit Agent）。
核心目标：
1. 诊断：将学生答案与标准答案对照，评估其正确性与推理过程。
2. 指令生成：若存在错误，给出一条抽象的、可在完全不同法律语境下"复现"该错误的指令。
内部评估流程：
1. 核对学生的最终答案是否与标准答案一致。
2. 检查推理过程的逻辑严密性（如遗漏条件、逻辑跳跃、误读定义、范围泛化等）。
3. 依据给定的错误分类表（Error Taxonomy）对错误进行分类。
4. 起草一条面向 Teacher 模型的 reproduction_instruction。
复现指令要求：必须是"与上下文无关"且"可执行"的，不能出现具体条款/实体名。
输出格式：严格 JSON，包含 status / error_types / generic_summary / reproduction_instruction。
"""

TEACHER_REJECTED_SYSTEM_PROMPT = """\
角色与目标：你是一名扮演"学生"角色的 AI 助教，需要生成一个有缺陷的教学示例——
一个"错误"的学生答案。
核心机制：
1. 输入处理：你会收到正确答案（Ground Truth）与具体的错误摘要（Error Summary）。
2. 缺陷体现：生成一段自然的、分步骤的推理过程，隐式地体现指定错误
   （例如忽略某个条件），但不能明说"我在犯错"。
3. 目标结果：推理过程必须自然地导向与标准答案相反的结论。
约束：不得在输出中提及错误摘要或标准答案本身；完全以"困惑学生"的口吻作答。
"""

TEACHER_CHOSEN_SYSTEM_PROMPT = """\
角色与目标：你是一名法律推理专家 AI 助教，需要基于一个已知的错误示例（rejected response），
生成一个简洁、逻辑严密、能够纠正该错误的正确推理（chosen response）。
核心机制：
1. 输入处理：你会收到合同文本、问题、标准答案，以及刚生成的 rejected response。
2. 纠错重点：明确点出 rejected response 忽略/误读了什么，并给出针对性的正确推理。
3. 目标结果：推理过程必须自然地导向与标准答案一致的结论，且尽量简洁（避免冗长的自我探索）。
"""

JUDGE_SYSTEM_PROMPT = """\
你是一名严格的法律推理裁判。字段定义：question=待判断的问题；
contract=作为依据的合同/条款文本；ground_truth=该问题的标准答案。
给定 question、contract、ground_truth 和一段模型回复，判断该回复是否包含
任何法律错误（事实、逻辑、解释或结论不一致）。
硬性规则：若模型回复的最终答案与 ground_truth 不一致，则必须判为 incorrect。
只输出一个词：correct 或 incorrect。
"""


def _build_audit_user_prompt(contract: str, question: str, ground_truth: str, student_response: str) -> str:
    return (
        f"合同文本：{contract}\n"
        f"问题：{question}\n"
        f"标准答案：{ground_truth}\n"
        f"学生回答：{student_response}\n"
        f"请按照 System Prompt 中定义的 JSON 格式输出诊断结果。"
    )


def _build_teacher_rejected_user_prompt(
    contract: str, question: str, ground_truth: str, error_types, generic_summary: str, reproduction_instruction: str
) -> str:
    return (
        f"合同文本：{contract}\n"
        f"问题：{question}\n"
        f"标准答案：{ground_truth}\n"
        f"目标错误类型：{error_types}\n"
        f"错误概述：{generic_summary}\n"
        f"复现指令：{reproduction_instruction}\n"
        f"请直接开始分步推理，最后一行以 'Final Answer: Yes' 或 'Final Answer: No' 结束。"
    )


def _build_teacher_chosen_user_prompt(contract: str, question: str, ground_truth: str, rejected_response: str) -> str:
    return (
        f"合同文本：{contract}\n"
        f"问题：{question}\n"
        f"标准答案：{ground_truth}\n"
        f"以下是需要纠正的错误示例（rejected response）：\n{rejected_response}\n"
        f"请生成简洁、正确的分步推理，最后一行以 'Final Answer: Yes' 或 'Final Answer: No' 结束。"
    )


def _build_judge_user_prompt(contract: str, question: str, ground_truth: str, model_generation: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Legal Context:\n{contract}\n\n"
        f"Ground Truth Answer:\n{ground_truth}\n\n"
        f"Model Response To Judge:\n{model_generation}\n\n"
        f"请只输出一个词：correct 或 incorrect。"
    )


class MockLLMClient(LLMClient):
    """模拟"强模型"角色：Audit Agent / Teacher / Judge 共用同一个 Mock 后端。

    真实论文中这三个角色背后是同一个强模型（Qwen3-30B-A3B-Instruct 或 GPT-4o），
    这里同样用一个类来承担三种角色，通过 meta["role"] 区分行为，
    与论文"教师模型与审计代理共用同一个强模型"的设定保持一致（见 Sec.4.1）。
    """

    def generate(self, system_prompt: str, user_prompt: str, meta: Optional[dict] = None) -> str:
        meta = meta or {}
        role = meta.get("role")
        if role == "audit":
            return self._audit(meta)
        if role == "teacher_rejected":
            return self._teacher_rejected(meta)
        if role == "teacher_chosen":
            return self._teacher_chosen(meta)
        if role == "judge":
            return self._judge(meta)
        raise ValueError(f"MockLLMClient 不认识的 role: {role}")

    # -- Audit Agent（对应 Sec.3.1 Diagnosis，附录 D.1）--
    # 附录 D.1 要求 Agent "Respond with a strict JSON object"，因此这里和真实 LLM
    # 一样返回 JSON 格式的字符串，而不是直接返回 dict，调用方需要自己 json.loads()。
    # 这样即使换成真实 API，diagnosis.py 的解析逻辑也不需要修改。
    def _audit(self, meta: dict) -> str:
        student_answer = meta["student_answer"]
        ground_truth = meta["ground_truth"]
        trap_type = meta["trap_type"]

        if student_answer.strip().lower() == ground_truth.strip().lower():
            # 学生本来就答对了：诊断为 CORRECT，不产生 reproduction_instruction
            result = {
                "status": "CORRECT",
                "error_types": [],
                "generic_summary": "学生的回答与标准答案一致，未发现需要修正的逻辑错误。",
                "reproduction_instruction": None,
            }
        else:
            bank = TRAP_BANK[trap_type]
            result = {
                "status": "INCORRECT_ANSWER",
                "error_types": bank["error_types"],
                "generic_summary": bank["generic_summary"],
                "reproduction_instruction": bank["reproduction_instruction"],
            }
        return json.dumps(result, ensure_ascii=False)

    # -- Teacher: rejected（对应 y_-^(k) ~ pi_teach(·|x, I_k)）--
    def _teacher_rejected(self, meta: dict) -> str:
        bank = TRAP_BANK[meta["trap_type"]]
        wrong_answer = opposite_answer(meta["ground_truth"])
        return bank["flawed_reasoning"].format(
            contract=meta["contract"], question=meta["question"], wrong_answer=wrong_answer
        )

    # -- Teacher: chosen（对应 y_+^(k) ~ pi_teach(·|x, I_k, y_-^(k))）--
    def _teacher_chosen(self, meta: dict) -> str:
        bank = TRAP_BANK[meta["trap_type"]]
        return bank["chosen_reasoning"].format(
            contract=meta["contract"], question=meta["question"], correct_answer=meta["ground_truth"]
        )

    # -- Judge（对应 Appendix A.1 / D.4）--
    def _judge(self, meta: dict) -> str:
        ground_truth = meta["ground_truth"].strip().lower()
        extracted = _extract_final_answer(meta["model_generation"])
        # 论文 Judge Prompt 的硬性规则：最终答案不匹配 => 一律 incorrect
        if extracted is None or extracted != ground_truth:
            return "incorrect"
        return "correct"


def _extract_final_answer(text: str) -> Optional[str]:
    """从形如 'Final Answer: Yes' 的文本中抽取 yes/no，找不到则返回 None。"""
    marker = "Final Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    tail = text[idx + len(marker):].strip().lower()
    if tail.startswith("yes"):
        return "yes"
    if tail.startswith("no"):
        return "no"
    return None


# ---------------------------------------------------------------------------
# 工厂函数：只改这里（或 config.USE_MOCK_BACKEND）即可切换到真实后端
# ---------------------------------------------------------------------------

def get_teacher_client() -> LLMClient:
    import config

    if config.USE_MOCK_BACKEND:
        return MockLLMClient()
    # 真实迁移示例（伪代码，未实现，避免引入网络依赖）：
    #   from real_clients import OpenAIChatClient
    #   return OpenAIChatClient(model="gpt-4o")
    raise NotImplementedError("请在此接入真实 Teacher API 客户端")


def get_audit_client() -> LLMClient:
    import config

    if config.USE_MOCK_BACKEND:
        return MockLLMClient()
    raise NotImplementedError("请在此接入真实 Audit Agent API 客户端")


def get_judge_client() -> LLMClient:
    import config

    if config.USE_MOCK_BACKEND:
        return MockLLMClient()
    raise NotImplementedError("请在此接入真实 Judge API 客户端")


__all__ = [
    "LLMClient",
    "MockLLMClient",
    "get_teacher_client",
    "get_audit_client",
    "get_judge_client",
    "AUDIT_AGENT_SYSTEM_PROMPT",
    "TEACHER_REJECTED_SYSTEM_PROMPT",
    "TEACHER_CHOSEN_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "_build_audit_user_prompt",
    "_build_teacher_rejected_user_prompt",
    "_build_teacher_chosen_user_prompt",
    "_build_judge_user_prompt",
]
