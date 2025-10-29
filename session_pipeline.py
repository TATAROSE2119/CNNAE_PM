import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from tep_data_load import tep_data_load
from train_session import train_one_session
from models_combined import CombinedAE
from eval_utils import make_eval_loader_from_dataset, plot_history, prep_windows_compat
from compute_cl_and_spe import compute_cl_and_spe


def train_mode_session(
    mode_id: str,
    normal: Tuple[str, str],
    backbone: torch.nn.Module,
    decoder: torch.nn.Module,
    teacher: Optional[torch.nn.Module],
    mem,
    mode_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    L: int,
    HOP: int,
    BATCH: int,
    VAL_RATIO: float,
    EPOCHS_SESSION: int,
    LR: float,
    DEVICE: str,
    add_to_mem_k: int = 50,
    loss_png: Optional[str] = None,
):
    """
    封装单个工况的训练流程（持续学习的一步）。
    步骤：加载数据 -> 窗口化 -> 训练 -> 画loss -> 估计窗口口径CL -> 加入记忆库

    返回：
      teacher: 新的 teacher（供后续会话蒸馏使用）
      CL_mode: 当前工况的窗口口径 CL (float)
      hist: 训练/验证损失历史
    """
    path, varname = normal
    X = tep_data_load(path, varname)
    print(f"{mode_id} 原始形状:", X.shape)

    train_loader, val_loader, (mean, std) = prep_windows_compat(
        X, L=L, hop=HOP, batch_size=BATCH, val_ratio=VAL_RATIO
    )
    mode_stats[mode_id] = (mean, std)

    teacher, hist = train_one_session(
        mode_id=mode_id,
        model_backbone=backbone,
        decoder=decoder,
        clf_head=None,
        teacher=teacher,
        mem=mem,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS_SESSION,
        lr=LR,
        device=DEVICE,
    )

    if loss_png is not None:
        os.makedirs(os.path.dirname(loss_png) or '.', exist_ok=True)
        plot_history(hist, f'Session ({mode_id}) Loss', loss_png)

    # 用当前模型组合计算窗口口径的 CL
    combo_model = CombinedAE(backbone, decoder).to(DEVICE).eval()
    eval_loader = make_eval_loader_from_dataset(train_loader.dataset)
    CL_mode, _ = compute_cl_and_spe(combo_model, eval_loader)
    print(f"{mode_id} CL =", float(CL_mode))

    # 加入记忆库（仅输入张量即可）
    xb, _ = next(iter(train_loader))
    mem.add(mode_id, xs=xb.cpu(), ys=None, k=add_to_mem_k)

    return teacher, float(CL_mode), hist

