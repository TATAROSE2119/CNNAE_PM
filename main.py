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
MODE_STATS = {}  # e.g., {'M1': (mean1, std1), 'M2': (mean2, std2), ...}
    # 已学到哪些工况就评哪些工况（逐会话递增）
NORMALS = [
    ('TE_data/M1/m1d00.mat', 'm1d00'),
    ('TE_data/M2/m2d00.mat', 'm2d00'),
    ('TE_data/M3/m3d00.mat', 'm3d00'),
    ('TE_data/M4/m4d00.mat', 'm4d00'),
]
FAULTS = [
    ('TE_data/M1/m1d04.mat', 'm1d04'),
    ('TE_data/M2/m2d04.mat', 'm2d04'),
    ('TE_data/M3/m3d04.mat', 'm3d04'),
    ('TE_data/M4/m4d04.mat', 'm4d04'),
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

# ====== 逐时间步 SPE（方案B）与多工况评估 ======
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1) 滑窗 -> E_win -> 重叠平均为逐时刻
def make_windows(X, mean, std, L=100, hop=10):
    Xn = (X - mean) / (std + 1e-8)
    starts = np.arange(0, Xn.shape[0]-L+1, hop, dtype=int)
    X_win = np.empty((len(starts), L, Xn.shape[1]), dtype=np.float32)
    for i,s in enumerate(starts): X_win[i] = Xn[s:s+L,:]
    X_cnn = torch.tensor(X_win).permute(0,2,1).contiguous()  # [N,P,L]
    return X_cnn, starts, Xn.shape[0]

@torch.no_grad()
def window_time_errors(model, X_cnn, batch=256, device=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    loader = DataLoader(TensorDataset(X_cnn, X_cnn), batch_size=batch, shuffle=False, drop_last=False)
    chunks = []
    for xb,_ in loader:
        xb = xb.to(device)
        xhat,_ = model(xb)
        # 对通道均方，保留时间维 -> [B,L]
        e = ((xb - xhat)**2).mean(dim=1).detach().cpu().numpy()
        chunks.append(e)
    E_win = np.concatenate(chunks, axis=0)  # [Nw,L]
    return E_win

def overlap_avg(E_win, starts, T):
    Nw,L = E_win.shape
    ssum = np.zeros(T, dtype=np.float64)
    cnt  = np.zeros(T, dtype=np.int32)
    for i,s in enumerate(starts):
        ssum[s:s+L] += E_win[i]
        cnt[s:s+L]  += 1
    spe_ts = np.full(T, np.nan, dtype=np.float64)
    m = cnt>0; spe_ts[m] = ssum[m]/cnt[m]
    return spe_ts, cnt

# 2) 构造“多工况拼接测试序列”
#   形如：mode1norm(200) : mode1fault(800) : mode2norm(200) : mode2fault(800) : ...
def build_multi_mode_sequence(norm_paths, fault_paths, mean, std, L=100, hop=10):
    """
    norm_paths: list of (path, varname) for normals, e.g. [('TE_data/M1/m1d00.mat','m1d00'), ...]
    fault_paths: list of (path, varname) for faults (每个mode各选一个含故障数据), e.g. [('TE_data/M1/m1d04.mat','m1d04'), ...]
    返回：X_all, seg_info（列表：[(t_norm_start,t_norm_end, t_fault_start,t_fault_end), ...]）
    """
    from tep_data_load import tep_data_load
    assert len(norm_paths)==len(fault_paths)
    X_list, segs = [], []
    t = 0
    for (pn, vn), (pf, vf) in zip(norm_paths, fault_paths):
        Xn = tep_data_load(pn, vn)[:200, :]
        Xf = tep_data_load(pf, vf)[:800, :]
        X_list.extend([Xn, Xf])
        segs.append((t, t+len(Xn)-1, t+len(Xn), t+len(Xn)+len(Xf)-1))
        t += len(Xn)+len(Xf)
    X_all = np.vstack(X_list)
    return X_all, segs

# 3) 逐时间步“全局 CL”（逐时间步口径）：用“已学习工况的正常数据”联合估计
def compute_global_CL_per_time(combo_model, learned_normal_sets, mean, std, L=100, hop=10, alpha=0.995, device=None):
    """
    learned_normal_sets: list of X_normal(ndarray)，每个为“已学习工况”的正常序列（可取整段正常）
    """
    all_spe = []
    for Xn in learned_normal_sets:
        X_cnn, starts, T = make_windows(Xn, mean, std, L, hop)
        E_win = window_time_errors(combo_model, X_cnn, device=device)
        spe_ts, _ = overlap_avg(E_win, starts, T)
        all_spe.append(spe_ts[L-1:])   # 避开左边界
    cat = np.concatenate(all_spe, axis=0)
    CL = float(np.nanquantile(cat, alpha))
    return CL

# 4) 指标（逐时间步口径，按全局CL）
def metrics_per_time(spe_ts, CL, segs):
    """
    segs: [(t_norm_s,t_norm_e,t_fault_s,t_fault_e), ...]
    """
    T = len(spe_ts)
    alarm = spe_ts > CL
    # FAR/TPR（分段累计 & 全局）
    pre_mask = np.zeros(T, dtype=bool)
    post_mask= np.zeros(T, dtype=bool)
    first_alarm = None
    for (ns,ne,fs,fe) in segs:
        pre_mask[ns:ne+1]  = True
        post_mask[fs:fe+1] = True
        after_idx = np.where(alarm[fs:fe+1])[0]
        if first_alarm is None and after_idx.size>0:
            first_alarm = fs + int(after_idx[0])
    FAR = float((alarm & pre_mask).sum()) / max(1, pre_mask.sum())
    TPR = float((alarm & post_mask).sum()) / max(1, post_mask.sum())
    delay = None if first_alarm is None else int(
        min([fs for (_,_,fs,_) in segs] + [first_alarm]) * 0 + first_alarm - segs[0][2]  # 以第一个故障起点参照
    )
    return dict(FAR=FAR, TPR=TPR, first_alarm=first_alarm, delay=delay, alarm=alarm)

# 5) 绘图
def plot_monitor(spe_ts, CL, segs, title, out_png):
    T = len(spe_ts)
    xs = np.arange(T)
    plt.figure(figsize=(11,4))
    plt.plot(xs, spe_ts, label='SPE (per-time)', linewidth=1.3)
    plt.axhline(CL, linestyle='--', label='Global CL (per-time)')
    for k,(ns,ne,fs,fe) in enumerate(segs, start=1):
        plt.axvline(ns, color='k', linestyle=':', alpha=0.5)
        plt.axvline(fs, color='r', linestyle=':', alpha=0.6)
        plt.text((ns+ne)//2, np.nanmax(spe_ts)*0.05, f'M{k}-Norm', ha='center', va='bottom', fontsize=8)
        plt.text((fs+fe)//2, np.nanmax(spe_ts)*0.10, f'M{k}-Fault', ha='center', va='bottom', fontsize=8)
    plt.title(title); plt.xlabel('Time'); plt.ylabel('SPE (per-time)')
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# 6) 一键评估：每个会话结束后调用
def eval_after_session(session_id, combo_model, learned_modes,   # e.g. ['M1'] 或 ['M1','M2',...]
                       normals, faults,                          # [(path,varname), ...] 与 learned_modes 对齐
                       L=100, hop=10, alpha=0.995, device=None):
    """
    learned_modes: 到当前会话为止已学习的工况列表，例如 ['M1'] 或 ['M1','M2']
    normals/faults: 与 learned_modes 一一对应的 (path,var) 列表
    """
    import numpy as np
    from tep_data_load import tep_data_load
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    combo_model = combo_model.to(device).eval()

    # ---- 工具：分段滑窗（按该工况自己的 mean/std），不跨段
    def _make_seg_windows(X, mean, std, L, hop):
        Xn = (X - mean) / (std + 1e-8)
        starts = np.arange(0, Xn.shape[0]-L+1, hop, dtype=int)
        Xw = np.empty((len(starts), L, Xn.shape[1]), dtype=np.float32)
        for i,s in enumerate(starts): Xw[i] = Xn[s:s+L,:]
        X_cnn = torch.tensor(Xw).permute(0,2,1).contiguous()
        return X_cnn, starts, Xn.shape[0]

    @torch.no_grad()
    def _win_err(model, X_cnn):
        ld = DataLoader(TensorDataset(X_cnn, X_cnn), batch_size=256, shuffle=False, drop_last=False)
        arr = []
        for xb,_ in ld:
            xb = xb.to(device)
            xhat,_ = model(xb)
            e = ((xb-xhat)**2).mean(dim=1).detach().cpu().numpy()
            arr.append(e)
        return np.concatenate(arr, axis=0)  # [Nw,L]

    def _overlap(E, starts, T):
        Nw,L = E.shape
        ssum = np.zeros(T, dtype=np.float64)
        cnt  = np.zeros(T, dtype=np.int32)
        for i,s in enumerate(starts):
            ssum[s:s+L] += E[i]; cnt[s:s+L] += 1
        spe = np.full(T, np.nan, dtype=np.float64)
        m = cnt>0; spe[m] = ssum[m]/cnt[m]
        return spe, cnt

    # ---- 1) 构造“200正常+800故障” × 每个已学工况，并分段计算窗口误差
    E_list, S_list = [], []
    segs, offset = [], 0
    T_total = 0
    for mode, (p_norm,v_norm), (p_fault,v_fault) in zip(learned_modes, normals, faults):
        mean, std = MODE_STATS[mode]
        Xn = tep_data_load(p_norm, v_norm)[:200,:]
        Xf = tep_data_load(p_fault, v_fault)[:800,:]
        Xseg = np.vstack([Xn, Xf])

        X_cnn, starts, T = _make_seg_windows(Xseg, mean, std, L, hop)
        Ew = _win_err(combo_model, X_cnn)

        E_list.append(Ew)
        S_list.append(starts + offset)                # 映射到全局时间
        segs.append((offset, offset+len(Xn)-1,        # (norm_start, norm_end,
                     offset+len(Xn), offset+len(Xn)+len(Xf)-1))  # fault_start, fault_end)
        offset += T
        T_total += T

    E_all = np.vstack(E_list)
    S_all = np.concatenate(S_list)
    spe_ts, cnt = _overlap(E_all, S_all, T_total)

    # ---- 2) 逐时间步全局 CL：用已学工况的“正常全段”，各自用自家 mean/std
    all_spe = []
    for mode, (p_norm,v_norm) in zip(learned_modes, normals):
        mean, std = MODE_STATS[mode]
        Xn_full = tep_data_load(p_norm, v_norm)
        X_cnn, starts, T = _make_seg_windows(Xn_full, mean, std, L, hop)
        Ew = _win_err(combo_model, X_cnn)
        spe_norm, _ = _overlap(Ew, starts, T)
        all_spe.append(spe_norm[L-1:])   # 避开左边界
    CL = float(np.nanquantile(np.concatenate(all_spe), alpha))

    # ---- 3) 指标
    alarm = spe_ts > CL
    pre_mask = np.zeros(T_total, dtype=bool)
    post_mask= np.zeros(T_total, dtype=bool)
    first_alarm = None
    for (ns,ne,fs,fe) in segs:
        pre_mask[ns:ne+1]  = True
        post_mask[fs:fe+1] = True
        idx = np.where(alarm[fs:fe+1])[0]
        if first_alarm is None and idx.size>0:
            first_alarm = fs + int(idx[0])
    FAR = float((alarm & pre_mask).sum()) / max(1, pre_mask.sum())
    TPR = float((alarm & post_mask).sum()) / max(1, post_mask.sum())

    # ---- 4) 画图
    xs = np.arange(T_total)
    plt.figure(figsize=(11,4))
    plt.plot(xs, spe_ts, label='SPE (per-time)', linewidth=1.2)
    plt.axhline(CL, linestyle='--', label='Global CL (per-time)')
    ymax = np.nanpercentile(spe_ts, 99.5) * 1.1 if np.isfinite(spe_ts).all() else np.nanmax(spe_ts)*1.1
    for k,(ns,ne,fs,fe) in enumerate(segs, start=1):
        plt.axvline(ns, color='k', linestyle=':', alpha=0.4)
        plt.axvline(fs, color='r', linestyle=':', alpha=0.5)
        plt.text((ns+ne)//2, ymax*0.05, f'M{k}-Norm', ha='center', va='bottom', fontsize=8)
        plt.text((fs+fe)//2, ymax*0.10, f'M{k}-Fault',ha='center', va='bottom', fontsize=8)
    plt.title(f'Session-{session_id} Monitoring (Per-time)')
    plt.xlabel('Time'); plt.ylabel('SPE (per-time)')
    plt.legend(); plt.tight_layout()
    plt.savefig(f'artifacts/monitor_S{session_id}.png', dpi=150)
    plt.close()

    metrics = dict(FAR=FAR, TPR=TPR, first_alarm=first_alarm)
    return CL, metrics, spe_ts, segs



def main():

    os.makedirs('artifacts', exist_ok=True)

    # =======================
    # 会话 1：M1
    # =======================
    # 1) 载入 M1（示例：无故障 m1d00；你可在此处合并 M1 的故障集）
    X_M1 = tep_data_load('TE_data/M1/m1d00.mat', 'm1d00')   # [T, P]
    print("M1 原始形状：", X_M1.shape)

    # 2) 窗口化（按时间切分 val）
    train_loader_M1, val_loader_M1, (mean1, std1) = prep_windows_compat(
        X_M1, L=L, hop=HOP, batch_size=BATCH, val_ratio=VAL_RATIO
    )
    MODE_STATS['M1'] = (mean1, std1)

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

    # === 会话1结束后（只评 M1）===
    combo_model = CombinedAE(backbone, decoder)
    CL1, m1, _, _ = eval_after_session(
        session_id=1, combo_model=combo_model,
        learned_modes=['M1'],
        normals=NORMALS[:1], faults=FAULTS[:1],
        L=L, hop=HOP, alpha=0.995, device=DEVICE
    )
    print('S1:', m1, 'CL=', CL1)

    # =======================
    # 会话 2：M2
    # =======================
    # 1) 载入 M2（示例：无故障 m2d00；可自行加入 M2 故障数据）
    #    若没有 M2 数据，请注释本段。这里示意流程。
    try:
        X_M2 = tep_data_load('TE_data/M2/m2d00.mat', 'm2d00')
        print("M2 原始形状：", X_M2.shape)

        train_loader_M2, val_loader_M2, (mean2, std2) = prep_windows_compat(
            X_M2, L=L, hop=HOP, batch_size=BATCH, val_ratio=VAL_RATIO
        )
        MODE_STATS['M2'] = (mean2, std2)

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
        # === 会话2结束后（评 M1+M2）===
        CL2, m2, _, _ = eval_after_session(
            session_id=2, combo_model=combo_model,
            learned_modes=['M1', 'M2'],
            normals=NORMALS[:2], faults=FAULTS[:2],
            L=L, hop=HOP, alpha=0.995, device=DEVICE
        )
        print('S2:', m2, 'CL=', CL2)




    except Exception as e:
        print("跳过 M2 会话（未找到数据或出错）：", e)
        CL_M2 = None

    # =======================
    # 保存工件（Artifacts）
    # =======================
    os.makedirs('artifacts', exist_ok=True)

    # 1️⃣ 模型参数（共享 CNN 编码器 + 解码器）
    torch.save(backbone.state_dict(), 'artifacts/backbone.pt')
    torch.save(decoder.state_dict(), 'artifacts/decoder.pt')

    # 2️⃣ 每个工况的 CL（控制限）
    # 例如：CL_dict = {'M1': CL_M1, 'M2': CL_M2, ...}
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

    print("\n✅ Artifacts saved in ./artifacts")
    print("   包含: backbone.pt, decoder.pt, cls.json, standardize_params.npz, meta.json")


if __name__ == "__main__":
    main()
