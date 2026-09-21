# -*- coding: utf-8 -*-
"""
optimizer.py —— Phase 5: SFT + DPO 优化目标

【论文对应】Sec.3.3 Optimization Objectives，核心公式：

    SFT (仅 t=0 冷启动):
        L_SFT(theta_0) = -E_{(x,y+)~D0_train} [ log pi_theta_0(y+|x) ]

    DPO (t=0 紧接 SFT 之后，以及后续每一轮 t>=1):
        L_DPO(theta) = -E[ log sigmoid( beta * (
            log(pi_theta(y+|x)/pi_ref(y+|x)) - log(pi_theta(y-|x)/pi_ref(y-|x))
        ) ) ]
    迭代设置中，参考模型 pi_ref 每轮都重置为当前策略 pi_theta_t（Sec.3.3 末尾）。

【核心简化：用"标量困惑度"代替真实梯度下降】
真实 SFT/DPO 是对 Transformer 全部参数求梯度、反传更新；这里的学生模型
没有神经网络参数，只有 confusion_weights[trap_type] 这一个标量状态
（见 student_model.py）。我们退而求其次，设计了两条"类比更新规则"：

  - SFT：让 confusion_weights[T] 按比例向 0 收缩
        new = old * (1 - SFT_UPDATE_STEP)
    直觉对应"模仿正确样本 y+，越多地看到某类错误的正确纠正，就越不容易再犯"。

  - DPO：更新幅度正比于 sigmoid(beta * DS)（DS 越大，说明当前越"自信地站在
    错误一边"，对应真实 DPO 里该样本的梯度权重也越大——因为 DPO 的梯度权重
    恰好就是 sigmoid(-margin) 这种形式，margin 越负、权重越大）：
        new = old - DPO_UPDATE_STEP * sigmoid(beta * DS)
    其中 DS 用 verification.py 算出的、更新前的 Difficulty Score。

这两条规则不是从真实反向传播推导出来的，只是数值方向和量级上"讲得通"的类比，
目的是让你观察到"confusion_weight 随着 SFT/DPO 逐步下降、DS 逐步收敛到 <=0"
这一符合论文定性结论（DPO 能强化对比信号、提升鲁棒性，见 Sec.4.4 Ablation）的现象。
不要用这里打印出来的 loss 数值去类比论文 Table 1 / Fig.3 里的真实准确率或 loss。
"""

import math
from dataclasses import dataclass
from typing import List

import config
from student_model import MockStudentClient
from verification import VerifiedPair


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class StepMetrics:
    stage: str          # "sft" or "dpo"
    avg_proxy_loss: float
    n_samples: int
    confusion_before: dict
    confusion_after: dict


def sft_step(
    student: MockStudentClient,
    verified_pairs: List[VerifiedPair],
    lr: float = config.SFT_UPDATE_STEP,
) -> StepMetrics:
    """冷启动 SFT：只用 chosen 一侧，对应 L_SFT(theta_0) = -E[log pi(y+|x)]。"""
    confusion_before = student.snapshot()

    # 代理损失：confusion_weight 本身就类比"模型犯这类错误的倾向"，
    # 数值越高，等价于真实模型对正确回答 y+ 赋予的似然越低（loss 越大）。
    losses = [confusion_before[vp.trap_type] for vp in verified_pairs]
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    for vp in verified_pairs:
        trap = vp.trap_type
        current = student.confusion_weights[trap]
        student.nudge_confusion(trap, -lr * current)

    return StepMetrics(
        stage="sft",
        avg_proxy_loss=avg_loss,
        n_samples=len(verified_pairs),
        confusion_before=confusion_before,
        confusion_after=student.snapshot(),
    )


def dpo_step(
    student: MockStudentClient,
    verified_pairs: List[VerifiedPair],
    lr: float = config.DPO_UPDATE_STEP,
    beta: float = config.DPO_BETA,
) -> StepMetrics:
    """DPO：用 (chosen, rejected) 偏好对，对应 L_DPO 公式；更新幅度按 sigmoid(beta*DS) 加权。"""
    confusion_before = student.snapshot()

    losses = [-math.log(_sigmoid(-beta * vp.ds) + 1e-8) for vp in verified_pairs]
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    for vp in verified_pairs:
        trap = vp.trap_type
        weight = _sigmoid(beta * vp.ds)  # DS 越大（越confused），更新权重越大
        student.nudge_confusion(trap, -lr * weight)

    return StepMetrics(
        stage="dpo",
        avg_proxy_loss=avg_loss,
        n_samples=len(verified_pairs),
        confusion_before=confusion_before,
        confusion_after=student.snapshot(),
    )
