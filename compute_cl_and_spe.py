# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@torch.no_grad()
def compute_spe(model,loader,device=None):
    """
    计算 SPE 指标
    :param model:
    :param loader:  数据加载器，提供待计算数据，形状 [N_samples, C, L]
    :param device:  计算设备（CPU 或 GPU），默认 None 表示自动选择
    :return:  SPE 数组，形状 [N_samples, ]
    """
    if torch.cuda.is_available():
        device = torch.device("cuda") if device is None else device
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval() #设置模型为评估模式
    spe_list=[]
    for batch in loader:
        # 兼容多种 batch 格式： (xb,yb) | (xb,) | xb | dict{ 'x','y' }
        if isinstance(batch, dict):
            xb = batch.get('x') or batch.get('input') or batch.get('data')
            if xb is None:
                xb = batch
        elif isinstance(batch, (list, tuple)):
            xb = batch[0]
        else:
            xb = batch

        xb=xb.to(device) #将数据移动到指定设备,如GPU
        xhat,_=model(xb) #前向传播，获取重构结果
        s=((xb-xhat)**2).mean(dim=(1,2)).detach().cpu().numpy() #计算每个样本的 SPE,并转回 CPU numpy 数组
        spe_list.append(s) #收集 SPE 结果
    return np.concatenate(spe_list,axis=0) #拼接所有批次的 SPE 结果，返回形状 [N_samples, ]

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
    spe_train = compute_spe(model, train_eval_loader)
    CL=float(np.quantile(spe_train,alpha))
    return CL,spe_train