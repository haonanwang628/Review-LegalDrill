# -*- coding: utf-8 -*-
"""
verification.py —— Phase 4: Self-Reflective Quality Verification

【论文对应】Sec.3.2，核心公式：
    构造强制二选一验证提示 Pver(c,q,y)，词表限制为 V={correct, incorrect}
    s_theta_t(y|x) = pi_theta_t(correct|Pver) / (pi_theta_t(correct|Pver) + pi_theta_t(incorrect|Pver))

    Difficulty Score:
        DS(x, y_+^(k), y_-^(k)) = s_theta_t(y_-^(k)|x) - s_theta_t(y_+^(k)|x)

    过滤规则：只保留 DS > tau 的样本进入最终训练集 D^t_train。
    - DS < 0：学生本来就能分辨出 y_+ 更对，样本"太简单"，不值得训练
    - DS > 0：学生反而更相信错误的 y_-，这才是真正的"盲点"，值得训练

【设计意图】
论文特别强调不用整段序列的似然 pi_theta_t(y|x) 来衡量"学生信不信这段话"，
因为长度、措辞的表面差异会干扰似然（更啰嗦或更"正式"的文本likelihood 天然更低/更高，
不代表内容对错）。改用强制二选一（correct/incorrect）的归一化打分，
是为了把"学生对内容对错的信念"和"文本的表面统计特性"解耦。
"""

from dataclasses import dataclass
from typing import List

import config
from student_model import MockStudentClient
from targeted_generation import PreferencePair


@dataclass
class VerifiedPair:
    pair: PreferencePair
    ds: float  # Difficulty Score

    @property
    def sample(self) -> dict:
        return self.pair.sample

    @property
    def chosen(self) -> str:
        return self.pair.chosen

    @property
    def rejected(self) -> str:
        return self.pair.rejected

    @property
    def trap_type(self) -> str:
        return self.pair.trap_type


def compute_difficulty_score(student: MockStudentClient, pair: PreferencePair) -> float:
    """计算单个偏好对的 Difficulty Score：DS = s(y-|x) - s(y+|x)。"""
    trap_type = pair.trap_type

    s_rejected = student.score_choices(
        system_prompt="",
        user_prompt="",
        meta={"trap_type": trap_type, "is_flawed": True},
    )["correct"]

    s_chosen = student.score_choices(
        system_prompt="",
        user_prompt="",
        meta={"trap_type": trap_type, "is_flawed": False},
    )["correct"]

    return s_rejected - s_chosen


def run_verification(
    student: MockStudentClient,
    pairs: List[PreferencePair],
    tau: float = config.DS_THRESHOLD,
) -> List[VerifiedPair]:
    """计算每个偏好对的 DS，只保留 DS > tau 的"高价值"样本。"""
    verified = []
    for pair in pairs:
        ds = compute_difficulty_score(student, pair)
        if ds > tau:
            verified.append(VerifiedPair(pair=pair, ds=ds))
    return verified
