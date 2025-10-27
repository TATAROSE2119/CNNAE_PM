import numpy as np
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
def compute_metrics_window(t_points, spe_win, CL, t_fault_start):
    """
    基于窗口级统计量（spe_win）与窗口时间点（t_points）的检测指标。
    返回: 字典 metrics
    """
    alarm = spe_win > CL
    alarm_idx = np.where(alarm)[0]
    t_alarm   = t_points[alarm]

    # 首次报警时间（发生在故障之后的首次）
    first_alarm_time = None
    for ta in t_alarm:
        if ta >= t_fault_start:
            first_alarm_time = int(ta)
            break

    delay = None
    if first_alarm_time is not None:
        delay = max(0, first_alarm_time - int(t_fault_start))

    # 预故障误报率 FAR（故障前窗口段上的报警比例）
    pre_mask = t_points < t_fault_start
    FAR = float((alarm & pre_mask).sum()) / max(1, pre_mask.sum())

    # 故障段检出率 TPR（故障后窗口段上的报警比例）
    post_mask = t_points >= t_fault_start
    TPR = float((alarm & post_mask).sum()) / max(1, post_mask.sum())

    return {
        "num_alarms": int(alarm.sum()),
        "first_alarm_time": first_alarm_time,  # None 表示未检出
        "delay": None if delay is None else int(delay),
        "FAR": FAR,
        "TPR": TPR,
        "alarm_mask": alarm,
        "t_alarm": t_alarm,
    }