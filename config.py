# -*- coding: utf-8 -*-
"""
config.py —— 全局超参数配置

【论文对应】
- K            : Sec.3.1 Targeted Generation，"we sample K error instructions {Ik} from Φerr"
- TAU (τ)      : Sec.3.2，DS(x,y+,y-) > τ 才保留该样本
- T_ITERATIONS : Sec.2.2 / Fig.1，迭代式偏好优化的总轮数 T
- DPO_BETA (β) : Sec.2.2 公式 L_DPO 中控制偏离参考模型程度的温度参数
- SFT_LR / DPO_LR: 论文 Limitations 部分提到的学习率范围
    "we find that typically, 1-3 epochs are enough with a learning rate of 1x10^-4"
    真实论文中 DPO 学习率范围是 [1e-6, 1e-4]，weight decay [1e-5, 1e-3]。
    本项目中学生模型是"简化的可训练打分器"（见 student_model.py），
    这里的学习率是我们类比出的更新步长，数值上不等价于真实 Transformer 的学习率。

【重要声明：模拟 vs 忠实复现】
本项目完全离线运行，不能调用任何真实 LLM API，也无法在本地训练真正的
Qwen3-0.6B/1.7B。因此：
  - teacher / audit agent / judge  —— 用 MockLLMClient 基于规则+模板确定性模拟（见 llm_client.py）
  - student (SLM)                  —— 用一个"极简可训练打分器" MockStudentClient 模拟（见 student_model.py），
                                        它没有神经网络参数，只有若干个标量"困惑度"权重，
                                        但依然可以真实地跑 SFT / DPO 的更新规则、观察 loss/DS 的变化趋势。
这样可以让你完整跑通论文的算法流程和数据流，但不能得到论文中的真实数值结果。
"""

import numpy as np

# 全局随机种子，保证整个流程确定性、可复现
SEED = 42
RNG = np.random.RandomState(SEED)

# ---- Sec.3.1 Targeted Generation ----
# 每个训练样本从 Error Instruction Bank 中采样的错误指令数量 K
# 论文里做了 K in {2,4,6,8,10,12,16} 的消融；这里取一个小值方便演示
K_ERROR_INSTRUCTIONS = 3

# ---- Sec.3.2 Self-Reflective Verification ----
# Difficulty Score 阈值 τ：只有 DS > TAU 的样本才会被保留进最终训练集
DS_THRESHOLD = 0.05

# ---- Sec.2.2 / Fig.1 迭代轮数 ----
T_ITERATIONS = 3

# ---- Sec.3.3 SFT / DPO ----
DPO_BETA = 0.1          # DPO 目标函数里的 beta
SFT_UPDATE_STEP = 0.35  # 简化学生模型：每次 SFT 更新降低 confusion_weight 的步长
DPO_UPDATE_STEP = 0.5   # 简化学生模型：每次 DPO 更新对 confusion_weight 的学习率系数

# 是否使用 Mock 后端（False 时应切换到真实 API/本地模型客户端，见 llm_client.py 的工厂函数）
USE_MOCK_BACKEND = True

DATA_PATH = "data/legal_qa_samples.json"
