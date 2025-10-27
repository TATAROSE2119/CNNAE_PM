# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import DataLoader

from compute_window_time_errors import compute_window_time_errors


def compute_cl_and_spe(model,train_loader,alpha:float=0.99):
    """
    计算控制限 CL
    :param model:
    :param train_loader:  训练数据加载器，提供正常数据，形状 [N_samples, C, L]
    :param alpha:  显著性水平，默认 0.99
    :param device:  计算设备（CPU 或 GPU），默认 None 表示自动选择
    :return:  控制限 CL 值
    """
    train_eval_loader = DataLoader(train_loader.dataset, batch_size=256,shuffle=False,drop_last=False)
    E_win, _, _ = compute_window_time_errors(model, train_eval_loader)
    spe_train = E_win.reshape(-1)
    CL=float(np.quantile(spe_train,alpha))
    return CL,spe_train
