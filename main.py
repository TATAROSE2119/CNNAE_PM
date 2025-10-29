# -*- coding: utf-8 -*-
"""
main.py — 持续学习最小可用主脚本（M1 -> M2）
依赖新增文件：backbone.py, memory.py, kd_losses.py, train_session.py
复用原文件：CNNAE.py, prepare_windows_for_cnn.py, tep_data_load.py, compute_cl_and_spe.py
"""

import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ===== 你现有模块 =====
from tep_data_load import tep_data_load
from prepare_windows_for_cnn import prepare_windows_for_cnn
from compute_cl_and_spe import compute_cl_and_spe   # 口径：窗口级 SPE 分位数阈值
from CNNAE import CNNAE

# ===== 新增模块 =====
from backbone import CNNBackbone
from memory import MemoryBank
from train_session import train_one_session


# -------------------------
# 基础设置
# -------------------------
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 滑窗参数（与原训练保持一致）
L, HOP = 100, 10
BATCH = 32
VAL_RATIO = 0.2

# 训练参数
EPOCHS_SESSION = 80
LR = 1e-3

# 记忆库
MEM_TOTAL_CAP = 300
MEM_PER_ADD = 50  # 每个新工况加入记忆库的窗口数（示例）


# -------------------------
# 小工具：把“编码器+解码器”组合成一个类，便于 compute_cl_and_spe 调用
# 需要 forward 返回 (xhat, z)
# -------------------------
class CombinedAE(torch.nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
    def forward(self, x):
        # x: [N,P,L]
        z = self.backbone(x)
        xhat = self.decoder(z)
        # 安全对齐长度
        if xhat.size(-1) > x.size(-1):
            xhat = xhat[..., :x.size(-1)]
        elif xhat.size(-1) < x.size(-1):
            pad = x.size(-1) - xhat.size(-1)
            xhat = torch.nn.functional.pad(xhat, (0, pad))
        return xhat, z


# -------------------------
# 构造一个“不打乱、不丢尾”的评估 loader（用于稳定计算 SPE/CL）
# -------------------------
def make_eval_loader_from_dataset(tds: TensorDataset, batch_size=256):
    return DataLoader(tds, batch_size=batch_size, shuffle=False, drop_last=False)


# -------------------------
# 绘图辅助
# -------------------------
def plot_history(hist, title, out_png):
    import numpy as np
    tr = np.array([h[0] for h in hist], dtype=float)
    va = np.array([h[1] for h in hist], dtype=float)
    ep = np.arange(1, len(hist) + 1)
    plt.figure(figsize=(8,4))
    plt.plot(ep, tr, label='train_loss', linewidth=1.8)
    plt.plot(ep, va, label='val_loss', linewidth=1.8)
    plt.xlabel('Epoch'); plt.ylabel('Loss (MSE/CE)')
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def prep_windows_compat(X, L, hop, batch_size, val_ratio):
    out = prepare_windows_for_cnn(X, L=L, hop=hop, batch_size=batch_size, val_ratio=val_ratio)
    # 兼容两种返回： (train,val,(mean,std)) 或 (train,val,mean,std)
    if isinstance(out, tuple) and len(out) == 4:
        train_loader, val_loader, mean, std = out
    elif isinstance(out, tuple) and len(out) == 3:
        train_loader, val_loader, ms = out
        mean, std = ms
    else:
        raise RuntimeError(f"prepare_windows_for_cnn 返回未知签名：len={len(out)}")
    return train_loader, val_loader, (mean, std)

def main():
    os.makedirs('artifacts', exist_ok=True)

    # =======================
    # 会话 1：M1
    # =======================
    # 1) 载入 M1（示例：无故障 m1d00；你可在此处合并 M1 的故障集）
    X_M1 = tep_data_load('TE_data/M1/m1d00.mat', 'm1d00')   # [T, P]
    print("M1 原始形状：", X_M1.shape)

    # 2) 窗口化（按时间切分 val）
    train_loader_M1, val_loader_M1, (mean, std) = prep_windows_compat(
        X_M1, L=L, hop=HOP, batch_size=BATCH, val_ratio=VAL_RATIO
    )

    # 3) 定义共享编码器与解码器
    #    复用你原来的 CNNAE 解码器；backbone 作为共享特征提取器
    P = train_loader_M1.dataset.tensors[0].shape[1]
    backbone = CNNBackbone(in_ch=P, feat_ch=64, latent_ch=32)
    ae_full = CNNAE(in_ch=P, latent_ch=32)
    decoder = ae_full.dec  # 取出解码器(与你原模型结构对齐)

    # 4) 记忆库与教师
    mem = MemoryBank(total_cap=MEM_TOTAL_CAP)
    teacher = None

    # 5) 训练 M1
    teacher, hist_M1 = train_one_session(
        mode_id='M1',
        model_backbone=backbone,
        decoder=decoder,
        clf_head=None,                 # 先不挂分类头
        teacher=teacher,               # 第一会话无教师
        mem=mem,
        train_loader=train_loader_M1,
        val_loader=val_loader_M1,
        epochs=EPOCHS_SESSION,
        lr=LR,
        device=DEVICE
    )
    plot_history(hist_M1, 'Session-1 (M1) Loss', 'artifacts/loss_M1.png')

    # 6) 用 M1 正常窗估计 CL（窗口口径）
    #    用组合模型(当前 backbone+decoder)来计算 SPE/CL，口径与原实现一致
    combo_model = CombinedAE(backbone, decoder).to(DEVICE).eval()
    # 这里用“训练集的 DataLoader.dataset”构造评估 loader，确保不打乱不丢尾
    eval_loader_M1 = make_eval_loader_from_dataset(train_loader_M1.dataset)
    CL_M1, _ = compute_cl_and_spe(combo_model, eval_loader_M1)
    print("M1 CL =", float(CL_M1))

    # 7) 记忆库加入 M1 少量窗口样本（仅输入张量即可）
    xb_M1, _ = next(iter(train_loader_M1))
    mem.add('M1', xs=xb_M1.cpu(), ys=None, k=MEM_PER_ADD)

    # =======================
    # 会话 2：M2
    # =======================
    # 1) 载入 M2（示例：无故障 m2d00；可自行加入 M2 故障数据）
    #    若没有 M2 数据，请注释本段。这里示意流程。
    try:
        X_M2 = tep_data_load('TE_data/M2/m2d00.mat', 'm2d00')
        print("M2 原始形状：", X_M2.shape)

        train_loader_M2, val_loader_M2, _ = prep_windows_compat(
            X_M2, L=L, hop=HOP, batch_size=BATCH, val_ratio=VAL_RATIO
        )

        # 2) 持续学习：用上会话 teacher + 记忆库，更新同一个 backbone/decoder
        teacher, hist_M2 = train_one_session(
            mode_id='M2',
            model_backbone=backbone,
            decoder=decoder,
            clf_head=None,
            teacher=teacher,
            mem=mem,
            train_loader=train_loader_M2,
            val_loader=val_loader_M2,
            epochs=EPOCHS_SESSION,
            lr=LR,
            device=DEVICE
        )
        plot_history(hist_M2, 'Session-2 (M2) Loss', 'artifacts/loss_M2.png')

        # 3) 估计 M2 的 CL
        combo_model = CombinedAE(backbone, decoder).to(DEVICE).eval()
        eval_loader_M2 = make_eval_loader_from_dataset(train_loader_M2.dataset)
        CL_M2, _ = compute_cl_and_spe(combo_model, eval_loader_M2)
        print("M2 CL =", float(CL_M2))

        # 4) 记忆库加入 M2 窗口样本
        xb_M2, _ = next(iter(train_loader_M2))
        mem.add('M2', xs=xb_M2.cpu(), ys=None, k=MEM_PER_ADD)
    except Exception as e:
        print("跳过 M2 会话（未找到数据或出错）：", e)
        CL_M2 = None

    # =======================
    # 保存工件
    # =======================
    # 模型
    torch.save(backbone.state_dict(), 'artifacts/backbone.pt')
    torch.save(decoder.state_dict(),  'artifacts/decoder.pt')

    # 阈值与标准化参数
    cl_dict = {'M1': float(CL_M1)}
    if CL_M2 is not None:
        cl_dict['M2'] = float(CL_M2)
    with open('artifacts/cls.json', 'w', encoding='utf-8') as f:
        json.dump(cl_dict, f, indent=2, ensure_ascii=False)

    np.savez('artifacts/standardize_params.npz', mean=mean, std=std)

    print("Artifacts saved in ./artifacts")
    print("Done.")


if __name__ == "__main__":
    main()
