# ramp.py
import math


def build_cosine_ramp(
    num_epochs: int,
    start: float = 0.25,
    end: float = 0.70,
    c_min: float = 0.10,
    c_max: float = 0.40,
):
    """
    余弦上升（cosine ramp-up）：在 [start*E, end*E] 内从 c_min 平滑升至 c_max。
    注意：这是“反向余弦退火”，专门用于副任务权重/正则的逐步引入。
    """
    E = int(num_epochs)
    s_ep = int(E * start)
    e_ep = int(E * end)
    s_ep = max(0, min(s_ep, E))
    e_ep = max(s_ep + 1, min(e_ep, E))

    def ramp_coef(epoch: int) -> float:
        if epoch <= s_ep:
            return c_min
        if epoch >= e_ep:
            return c_max
        t = (epoch - s_ep) / max(1, (e_ep - s_ep))
        # 0→1 的余弦上升：0.5*(1 - cos(pi*t))
        return c_min + (c_max - c_min) * 0.5 * (1 - math.cos(math.pi * t))

    return ramp_coef
