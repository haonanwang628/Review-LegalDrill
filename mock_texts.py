# -*- coding: utf-8 -*-
"""
mock_texts.py —— 错误类型知识库 TRAP_BANK

【论文对应】Sec.3.1 Diagnosis 段落：
    "To ensure that the diagnosis aligns with industry standards, we inject
     a taxonomy of common legal mistakes into the Agent's prompt."
论文里 Audit Agent 的 Prompt 中会内置一份"常见法律错误分类表"，Agent 依据这个分类表
去诊断学生的错误、并写出 context-agnostic 的 reproduction_instruction
（见附录D.1 Agent System Prompt 的 Error Taxonomy / Reproduction Instruction Guidelines）。

【模拟简化声明】
真实论文中，这份 taxonomy 只是给 Agent 的"参考标准"，具体每条错误的归因、摘要、
复现指令都是强模型（GPT-4o / Qwen3-30B）当场读懂上下文后现场生成的，具有很强的
上下文理解能力。我们离线跑不了真实LLM，无法做到"读懂任意合同文本"，
因此这里退而求其次：直接把"taxonomy 的四个条目"做成模板化的知识库 TRAP_BANK，
每个条目预置好 generic_summary / reproduction_instruction 的固定写法，
以及"体现该错误"和"纠正该错误"两种推理话术的模板。
这样得到的效果是：错误类型、诊断话术、纠正话术是确定且可控的，
但换成任何真实合同文本时，模板套话不会像真实LLM那样精确引用合同原文细节，
这是我们为了在无网络环境下"端到端跑通"所付出的保真度代价。
"""

from typing import Dict

TRAP_BANK: Dict[str, dict] = {
    "missing_condition": {
        # 对应论文 Fig.2 示例中出现的 "Missing condition" 错误类型标签
        "error_types": ["Missing condition"],
        "generic_summary": (
            "学生忽略了条款中附加的限制/例外条件，把一条『有条件才成立』的规则，"
            "当成『无条件、普遍成立』的规则来适用。"
        ),
        "reproduction_instruction": (
            "识别文本中限制某项权利或规则适用范围的条件、例外或但书条款；"
            "在推理时完全不提及该条件，把规则当作无条件、普遍适用的规则来处理，"
            "并据此得出与正确结论相反的最终判断。"
        ),
        "flawed_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 该条款描述了一项规则，规则本身已经足以支撑对问题的判断。\n"
            "Step 3: 条款中提到的限定或例外情形只是补充说明，不影响规则的核心适用范围，"
            "因此可以忽略这些限定，直接按规则的字面主干进行判断。\n"
            "针对问题“{question}”，按照规则主干可以直接得出结论。\n"
            "Final Answer: {wrong_answer}"
        ),
        "chosen_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 该条款不仅规定了一般规则，还明确附加了限制/例外条件，"
            "这些条件是规则能否适用的关键前提，不能被忽略。\n"
            "Step 3: 结合问题“{question}”，需要逐一核对该限制/例外条件是否被满足，"
            "只有在条件被满足（或不构成例外）时，规则才成立；反之则不成立。\n"
            "综合以上限定条件的核查结果，可以得出准确结论。\n"
            "Final Answer: {correct_answer}"
        ),
    },
    "logical_leap": {
        "error_types": ["Logical leap"],
        "generic_summary": (
            "学生在条款文本没有明确支持的情况下，做了一次跨越式的推断"
            "（例如把『提交申请』等同于『立即生效』，或把『部分满足』等同于『完全满足』触发条件）。"
        ),
        "reproduction_instruction": (
            "在条款描述的触发条件与其法律效果之间，插入一个文本并未明确支持的额外假设"
            "（例如把一个渐进的、需要满足特定前提的过程等同于立即发生的结果），"
            "并据此得出与正确结论相反的最终判断。"
        ),
        "flawed_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 条款描述了某个触发条件会引出某种法律效果。\n"
            "Step 3: 可以合理地认为，只要出现了与该触发条件相关的迹象，效果就应当立即、"
            "完全地成立，不需要再等待条款中提到的中间过程或额外前提。\n"
            "针对问题“{question}”，据此直接得出结论。\n"
            "Final Answer: {wrong_answer}"
        ),
        "chosen_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 条款对触发条件与法律效果之间的关系有明确、具体的界定"
            "（例如特定的时间点、特定的前置程序或特定的次数/幅度）。\n"
            "Step 3: 不能脱离这些明确界定去外推结论；需要严格核对问题“{question}”"
            "中描述的情形，是否精确满足条款界定的触发条件。\n"
            "经核对，只有精确满足条款界定的情形才成立。\n"
            "Final Answer: {correct_answer}"
        ),
    },
    "term_misreading": {
        "error_types": ["Misreading defined term"],
        "generic_summary": (
            "学生混淆了条款中明确定义的关键术语的外延，"
            "把定义之外的主体/情形也当作该术语的适用对象（或反之，错误地把定义内的对象排除在外）。"
        ),
        "reproduction_instruction": (
            "找到文本中一个被明确定义（限定外延）的关键术语；"
            "在推理时不遵守该定义给出的边界，转而使用该术语的日常/宽泛含义来判断问题，"
            "并据此得出与正确结论相反的最终判断。"
        ),
        "flawed_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 条款中出现了一个关键术语，日常语境下这个词的含义比较宽泛。\n"
            "Step 3: 按照日常、宽泛的理解来适用这个术语，而不去核对条款是否对它给出了"
            "更严格的专门定义。\n"
            "针对问题“{question}”，按宽泛理解直接得出结论。\n"
            "Final Answer: {wrong_answer}"
        ),
        "chosen_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 条款对关键术语给出了明确、限定性的定义，该定义的外延"
            "小于（或不同于）这个词在日常语境下的含义。\n"
            "Step 3: 必须严格按照条款给出的定义边界来判断问题“{question}”"
            "中的主体/情形是否落在定义范围内，而不能套用日常含义。\n"
            "经核对定义边界，得出准确结论。\n"
            "Final Answer: {correct_answer}"
        ),
    },
    "scope_overgeneralization": {
        "error_types": ["Scope overgeneralization"],
        "generic_summary": (
            "学生把条款中限定给特定主体、特定情形或特定阶段的规则，"
            "错误地扩大（或缩小）适用到条款文本未覆盖的其他主体/情形/阶段。"
        ),
        "reproduction_instruction": (
            "找到文本中一条只适用于特定主体、特定情形或特定阶段的规则；"
            "在推理时忽略这一适用范围的限定，将规则一般化地套用到问题所问的、"
            "条款并未明确覆盖的其他主体/情形/阶段上，并据此得出与正确结论相反的最终判断。"
        ),
        "flawed_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 条款规定了一项规则，这类规则背后体现的原则具有普遍性和合理性。\n"
            "Step 3: 既然原则是合理的，就应当类推适用到问题“{question}”所描述的"
            "其他主体/情形/阶段，而不必拘泥于条款字面限定的适用范围。\n"
            "据此得出结论。\n"
            "Final Answer: {wrong_answer}"
        ),
        "chosen_reasoning": (
            "Step 1: 定位条款原文——“{contract}”。\n"
            "Step 2: 条款明确将规则的适用范围限定在特定主体/特定情形/特定阶段，"
            "这是条款文本本身划定的边界，不能仅凭『原则合理』就类推扩大。\n"
            "Step 3: 核对问题“{question}”所描述的主体/情形/阶段，"
            "是否落在条款明确限定的适用范围之内。\n"
            "经核对适用范围，得出准确结论。\n"
            "Final Answer: {correct_answer}"
        ),
    },
}


def opposite_answer(answer: str) -> str:
    """给定 Yes/No，返回相反答案。用于构造 rejected 样本的错误结论。"""
    return "No" if answer.strip().lower() == "yes" else "Yes"
