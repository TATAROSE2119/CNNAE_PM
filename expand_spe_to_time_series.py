import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def expand_spe_to_time_series(spe_win, t_points, T, mode='step', cl=None):
    """
    spe_win:  窗口级 SPE，形状 [Nw]
    t_points: 每个窗口代表时间点（通常 = starts + L - 1），形状 [Nw]
    T:        原序列总长度（这里是 600）
    mode:     'step'（阶梯保持）或 'linear'（线性插值）
    cl:       可选，控制限（标量），若提供将一并展开

    返回:
      spe_full: 长度 T 的逐时刻 SPE（前 L-1 处设为 np.nan）
      cl_full:  同上长度的控制限（若 cl 提供；否则为 None）
    """
    spe_full = np.full(T, np.nan, dtype=float)

    # 有效区间：从第一个窗口代表点开始（通常是 L-1），到结尾
    t0 = int(t_points[0])
    t_last = int(t_points[-1])

    if mode == 'step':
        # 阶梯保持：每两个代表点之间用前一个窗口的 SPE 填充
        for i in range(len(t_points) - 1):
            spe_full[t_points[i]: t_points[i+1]] = spe_win[i]
        # 最后一段直到结尾
        spe_full[t_points[-1]:] = spe_win[-1]

    elif mode == 'linear':
        # 线性插值：只在已定义区间内插
        # 边界：左侧保持第一个值，右侧保持最后一个值
        x = t_points.astype(float)
        y = spe_win.astype(float)
        spe_full[t0:] = np.interp(np.arange(t0, T), x, y, left=y[0], right=y[-1])
    else:
        raise ValueError("mode 必须是 'step' 或 'linear'")

    # 控制限展开（可选）
    cl_full = None
    if cl is not None:
        cl_full = np.full(T, np.nan, dtype=float)
        if mode == 'step':
            # 控制限不随时间变化，直接填常数
            cl_full[t0:] = float(cl)
        else:
            cl_full[t0:] = float(cl)

    return spe_full, cl_full