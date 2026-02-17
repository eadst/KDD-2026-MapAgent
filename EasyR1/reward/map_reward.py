import re
from typing import Any, Dict


# =========================
# Metadata
# =========================

REWARD_NAME = "lane_error_grpo"
REWARD_TYPE = "sequential"


# =========================
# 常量定义
# =========================

VALID_ERRORS = {
    "no_error",
    "extra_lane_line",
    "category_error",
    "geometry_error",
    "structure_error",
}


# =========================
# 工具函数
# =========================

def extract_error_type(text: str) -> str:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    return lines[-1]


def extract_think_text(text: str) -> str:
    lines = [l.rstrip() for l in text.strip().splitlines()]
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[:-1])


# =========================
# Layer 1：错误类型准确性
# =========================

def accuracy_reward(response: str, ground_truth: str) -> float:
    pred = extract_error_type(response)
    gt = extract_error_type(ground_truth)

    # 完全正确
    if pred == gt and pred in VALID_ERRORS:
        return 1.0

    # 输出的是合法 error，但不匹配 GT
    if pred in VALID_ERRORS:
        return -0.3

    # 完全非法输出
    return -1.0


# =========================
# Layer 2：格式合法性
# =========================

def format_reward(response: str) -> float:
    reward = 0.0
    pred = extract_error_type(response)

    # 没有任何 error_type
    if not pred:
        reward -= 0.5

    # error_type 非法
    elif pred not in VALID_ERRORS:
        reward -= 0.5

    return reward


# =========================
# Layer 3：冗余惩罚
# =========================

def verbosity_penalty(response: str) -> float:
    think_text = extract_think_text(response)

    if not think_text:
        return 0.0

    token_estimate = len(think_text.split())

    # 超过一定长度才轻微惩罚
    if token_estimate > 200:
        return -0.2

    return 0.0


# =========================
# 总 Reward
# =========================

def compute_score(reward_input: Dict[str, Any]) -> Dict[str, float]:
    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]

    acc = accuracy_reward(response, ground_truth)
    fmt = format_reward(response)
    verb = verbosity_penalty(response)

    overall = acc + fmt + verb

    return {
        "overall": overall,
        "accuracy": acc,
        "format": fmt,
        "verbosity": verb,
    }
