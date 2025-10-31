# -*- coding: utf-8 -*-
"""
main.py 持续学习主脚本（M1 -> M2）
依赖新增文件：backbone.py, memory.py, kd_losses.py, train_session.py
复用原文件：CNNAE.py, prepare_windows_for_cnn.py, tep_data_load.py, compute_cl_and_spe.py

本文件已精简：公共/工具函数已迁移至
- models_combined.py: CombinedAE
- eval_utils.py: make_eval_loader_from_dataset, plot_history, prep_windows_compat
- monitoring.py: eval_after_session
"""

import os
import random
import json
import numpy as np
import torch

# 现有模块
from tep_data_load import tep_data_load
from CNNAE import CNNAE

# 新增模块
from backbone import CNNBackbone
from memory import MemoryBank
from models_combined import CombinedAE
from monitoring import eval_after_session
from session_pipeline import train_mode_session


# -------------------------
# 基础设置
# -------------------------
MODE_STATS = {}  # e.g., {'M1': (mean1, std1), 'M2': (mean2, std2), ...}
NORMALS = [
    ('TE_data/M1/m1d00.mat', 'm1d00'),
    ('TE_data/M2/m2d00.mat', 'm2d00'),
    ('TE_data/M3/m3d00.mat', 'm3d00'),
    ('TE_data/M4/m4d00.mat', 'm4d00'),
]
FAULTS = [
    ('TE_data/M1/m1d14.mat', 'm1d14'),
    ('TE_data/M2/m2d14.mat', 'm2d14'),
    ('TE_data/M3/m3d14.mat', 'm3d14'),
    ('TE_data/M4/m4d14.mat', 'm4d14'),
]

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
EPOCHS_SESSION = 200
LR = 1e-3

# 记忆库
MEM_TOTAL_CAP = 300
MEM_PER_ADD = 50  # 每个新工况加入记忆库的窗口数（示例）


def main():
    os.makedirs('artifacts', exist_ok=True)

    # =======================
    # 会话 1：M1
    # =======================
    # 1) 载入 M1（示例：无故障 m1d00；可在此处合并 M1 的故障数据）
    X_M1 = tep_data_load('TE_data/M1/m1d00.mat', 'm1d00')   # [T, P]
    print("M1 原始形状:", X_M1.shape)

    # 2) 定义共享编码器与解码器
    #    复用你原来的 CNNAE 解码器；backbone 作为共享特征提取器
    P = X_M1.shape[1]
    backbone = CNNBackbone(in_ch=P, feat_ch=64, latent_ch=32)
    ae_full = CNNAE(in_ch=P, latent_ch=32)
    decoder = ae_full.dec  # 取出解码器，与你原模型结构对齐

    # 4) 记忆库与教师
    mem = MemoryBank(total_cap=MEM_TOTAL_CAP)
    teacher = None

    # 3) 训练 M1（封装函数）
    teacher, CL_M1, hist_M1 = train_mode_session(
        mode_id='M1',
        normal=('TE_data/M1/m1d00.mat', 'm1d00'),
        backbone=backbone,
        decoder=decoder,
        teacher=teacher,
        mem=mem,
        mode_stats=MODE_STATS,
        L=L, HOP=HOP, BATCH=BATCH, VAL_RATIO=VAL_RATIO,
        EPOCHS_SESSION=EPOCHS_SESSION, LR=LR, DEVICE=DEVICE,
        add_to_mem_k= MEM_PER_ADD,
        loss_png='artifacts/loss_M1.png'
    )

    # === 会话1结束后（只评 M1）===
    combo_model = CombinedAE(backbone, decoder)
    CL1, m1, _, _ = eval_after_session(
        session_id=1, combo_model=combo_model,
        learned_modes=['M1'],
        normals=NORMALS[:1], faults=FAULTS[:1],
        L=L, hop=HOP, alpha=0.995, device=DEVICE, mode_stats=MODE_STATS
    )
    print('S1:', m1, 'CL=', CL1)

    # =======================
    # 会话 2：M2
    # =======================
    # 1) 载入 M2（示例：无故障 m2d00；可自行加入 M2 故障数据）
    #    若没有 M2 数据，请注释本段。这里示意流程
    try:
        X_M2 = tep_data_load('TE_data/M2/m2d00.mat', 'm2d00')
        print("M2 原始形状:", X_M2.shape)

        # 2) 训练 M2（封装函数）
        teacher, CL_M2, hist_M2 = train_mode_session(
            mode_id='M2',
            normal=('TE_data/M2/m2d00.mat', 'm2d00'),
            backbone=backbone,
            decoder=decoder,
            teacher=teacher,
            mem=mem,
            mode_stats=MODE_STATS,
            L=L, HOP=HOP, BATCH=BATCH, VAL_RATIO=VAL_RATIO,
            EPOCHS_SESSION=EPOCHS_SESSION, LR=LR, DEVICE=DEVICE,
            add_to_mem_k= MEM_PER_ADD,
            loss_png='artifacts/loss_M2.png'
        )

        # === 会话2结束后（评 M1+M2）===
        CL2, m2, _, _ = eval_after_session(
            session_id=2, combo_model=combo_model,
            learned_modes=['M1', 'M2'],
            normals=NORMALS[:2], faults=FAULTS[:2],
            L=L, hop=HOP, alpha=0.995, device=DEVICE, mode_stats=MODE_STATS
        )
        print('S2:', m2, 'CL=', CL2)

    except Exception as e:
        print("跳过 M2 会话（未找到数据或出错）:", e)
        CL_M2 = None
    
    try:
        X_M3 = tep_data_load('TE_data/M3/m3d00.mat', 'm3d00')
        print("M3 原始形状:", X_M3.shape)

        # 2) 训练 M3（封装函数）
        teacher, CL_M3, hist_M3 = train_mode_session(
            mode_id='M3',
            normal=('TE_data/M3/m3d00.mat', 'm3d00'),
            backbone=backbone,
            decoder=decoder,
            teacher=teacher,
            mem=mem,
            mode_stats=MODE_STATS,
            L=L, HOP=HOP, BATCH=BATCH, VAL_RATIO=VAL_RATIO,
            EPOCHS_SESSION=EPOCHS_SESSION, LR=LR, DEVICE=DEVICE,
            add_to_mem_k= MEM_PER_ADD,
            loss_png='artifacts/loss_M3.png'
        )

        # === 会话3结束后（评 M1+M2+M3）===
        CL3, m3, _, _ = eval_after_session(
            session_id=3, combo_model=combo_model,
            learned_modes=['M1', 'M2', 'M3'],
            normals=NORMALS[:3], faults=FAULTS[:3],
            L=L, hop=HOP, alpha=0.995, device=DEVICE, mode_stats=MODE_STATS
        )
        print('S3:', m3, 'CL=', CL3)
        
    except Exception as e:
        print("跳过 M3 会话（未找到数据或出错）:", e)
        CL_M3 = None
    # =======================
    # 保存工件（Artifacts）
    # =======================
    os.makedirs('artifacts', exist_ok=True)

    # 1️⃣ 模型参数（共享 CNN 编码器 + 解码器）
    torch.save(backbone.state_dict(), 'artifacts/backbone.pt')
    torch.save(decoder.state_dict(), 'artifacts/decoder.pt')

    # 2️⃣ 每个工况的 CL（控制限）
    CL_dict = {}
    if 'CL_M1' in locals(): CL_dict['M1'] = float(CL_M1)
    if 'CL_M2' in locals(): CL_dict['M2'] = float(CL_M2)
    if 'CL_M3' in locals(): CL_dict['M3'] = float(CL_M3)
    if 'CL_M4' in locals(): CL_dict['M4'] = float(CL_M4)

    with open('artifacts/cls.json', 'w', encoding='utf-8') as f:
        json.dump(CL_dict, f, indent=2, ensure_ascii=False)

    # 3️⃣ 每个工况的标准化参数
    # MODE_STATS 是全局字典：{'M1': (mean1,std1), 'M2': (mean2,std2), ...}
    npz_dict = {}
    for mode, (mean_val, std_val) in MODE_STATS.items():
        npz_dict[f'{mode}_mean'] = mean_val
        npz_dict[f'{mode}_std'] = std_val
    np.savez('artifacts/standardize_params.npz', **npz_dict)

    # 4️⃣ 记录会话索引（当前已经学到第几工况）
    meta = {'learned_modes': list(MODE_STATS.keys())}
    with open('artifacts/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n✔ Artifacts saved in ./artifacts")
    print("   包含: backbone.pt, decoder.pt, cls.json, standardize_params.npz, meta.json")


if __name__ == "__main__":
    main()
