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
def eval_after_session(session_id, combo_model, mean, std, learned_normal_paths, learned_fault_paths,
                       L=100, hop=10, device=None, alpha=0.995):
    """
    learned_normal_paths / learned_fault_paths: 只放“到当前会话为止”的工况顺序子集。
      例如 Session-1 用 [(M1_norm)], [(M1_fault)]
           Session-2 用 [(M1_norm,M2_norm)], [(M1_fault,M2_fault)] 以此类推
    """
    # ① 构造多工况测试序列（200 正常 + 800 故障）× 已学工况
    X_test, segs = build_multi_mode_sequence(learned_normal_paths, learned_fault_paths, mean, std, L, hop)
    # ② 逐时间步 SPE（方案B）
    X_cnn, starts, T = make_windows(X_test, mean, std, L, hop)
    E_win = window_time_errors(combo_model, X_cnn, device=device)
    spe_ts, cnt = overlap_avg(E_win, starts, T)
    # ③ 全局 CL（逐时间步口径，基于“已学工况”的正常数据集合）
    learned_normals = []
    for (pn, vn) in learned_normal_paths:
        from tep_data_load import tep_data_load
        learned_normals.append(tep_data_load(pn, vn))
    CL = compute_global_CL_per_time(combo_model, learned_normals, mean, std, L, hop, alpha=alpha, device=device)
    # ④ 指标
    m = metrics_per_time(spe_ts, CL, segs)
    # ⑤ 画图
    plot_monitor(spe_ts, CL, segs, f'Session-{session_id} Monitoring (Per-time)', f'artifacts/monitor_S{session_id}.png')
    return CL, m, spe_ts, segs




def main():
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

    # === 会话1结束后（只评 M1）===
    CL_S1, metrics_S1, spe_ts_S1, segs_S1 = eval_after_session(
        session_id=1,
        combo_model=combo_model,
        mean=mean, std=std,
        learned_normal_paths=NORMALS[:1],
        learned_fault_paths=FAULTS[:1],
        L=L, hop=HOP, device=DEVICE, alpha=0.995
    )
    print("S1 Global-CL:", CL_S1, "Metrics:", metrics_S1)

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
        # === 会话2结束后（评 M1+M2）===
        CL_S2, metrics_S2, spe_ts_S2, segs_S2 = eval_after_session(
            session_id=2,
            combo_model=combo_model,
            mean=mean, std=std,
            learned_normal_paths=NORMALS[:2],
            learned_fault_paths=FAULTS[:2],
            L=L, hop=HOP, device=DEVICE, alpha=0.995
        )
        print("S2 Global-CL:", CL_S2, "Metrics:", metrics_S2)



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
