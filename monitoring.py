import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def make_windows(X, mean, std, L=100, hop=10):
    """Normalize X, slice sliding windows, and return CNN-ready tensor.
    Returns: X_cnn [N, P, L], starts, T
    """
    Xn = (X - mean) / (std + 1e-8)
    starts = np.arange(0, Xn.shape[0] - L + 1, hop, dtype=int)
    X_win = np.empty((len(starts), L, Xn.shape[1]), dtype=np.float32)
    for i, s in enumerate(starts):
        X_win[i] = Xn[s : s + L, :]
    X_cnn = torch.tensor(X_win).permute(0, 2, 1).contiguous()  # [N,P,L]
    return X_cnn, starts, Xn.shape[0]


@torch.no_grad()
def _window_time_errors(model, X_cnn, batch=256, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    loader = DataLoader(TensorDataset(X_cnn, X_cnn), batch_size=batch, shuffle=False, drop_last=False)
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device)
        xhat, _ = model(xb)
        e = ((xb - xhat) ** 2).mean(dim=1).detach().cpu().numpy()  # [B,L]
        chunks.append(e)
    E_win = np.concatenate(chunks, axis=0)  # [Nw,L]
    return E_win


def overlap_avg(E_win, starts, T):
    """将逐窗口误差叠加到原始时间轴，并对每个时刻基于“有效值”求平均。
    - 对含 NaN 的窗口时间步不计入该时刻的计数（逐时刻忽略 NaN）。
    返回: spe_ts, cnt（逐时刻有效计数）
    """
    Nw, L = E_win.shape
    ssum = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int32)
    for i, s in enumerate(starts):
        seg = E_win[i]
        mask = np.isfinite(seg)
        # 仅累计有限值，NaN/Inf 跳过
        if mask.any():
            ssum[s : s + L][mask] += seg[mask]
            cnt[s : s + L][mask] += 1
    spe_ts = np.full(T, np.nan, dtype=np.float64)
    m = cnt > 0
    spe_ts[m] = ssum[m] / cnt[m]
    return spe_ts, cnt


def build_multi_mode_sequence(norm_paths, fault_paths, mean, std, L=100, hop=10):
    """Build concatenated multi-mode sequence and segments.
    Returns X_all and segs = [(t_norm_start,t_norm_end,t_fault_start,t_fault_end), ...]
    """
    from tep_data_load import tep_data_load
    assert len(norm_paths) == len(fault_paths)
    X_list, segs = [], []
    t = 0
    for (pn, vn), (pf, vf) in zip(norm_paths, fault_paths):
        Xn = tep_data_load(pn, vn)[:200, :]
        Xf = tep_data_load(pf, vf)[:800, :]
        X_list.extend([Xn, Xf])
        segs.append((t, t + len(Xn) - 1, t + len(Xn), t + len(Xn) + len(Xf) - 1))
        t += len(Xn) + len(Xf)
    X_all = np.vstack(X_list)
    return X_all, segs


def compute_global_CL_per_time(combo_model, learned_normal_sets, mean, std, L=100, hop=10, alpha=0.995, device=None):
    """Global per-time CL from normal sets of learned modes (robust to NaN)."""
    all_spe = []
    for Xn in learned_normal_sets:
        X_cnn, starts, T = make_windows(Xn, mean, std, L, hop)
        E_win = _window_time_errors(combo_model, X_cnn, device=device)
        spe_ts, _ = overlap_avg(E_win, starts, T)
        part = spe_ts[L - 1 :]
        if part.size:
            all_spe.append(part[np.isfinite(part)])  # 仅保留有效值
    cat = np.concatenate(all_spe, axis=0) if all_spe else np.array([], dtype=float)
    CL = float(np.nanquantile(cat, alpha)) if cat.size > 0 else float('nan')
    return CL


def metrics_per_time(spe_ts, CL, segs):
    """Compute FAR/TPR/first alarm using per-time criterion."""
    T = len(spe_ts)
    alarm = spe_ts > CL
    pre_mask = np.zeros(T, dtype=bool)
    post_mask = np.zeros(T, dtype=bool)
    first_alarm = None
    for (ns, ne, fs, fe) in segs:
        pre_mask[ns : ne + 1] = True
        post_mask[fs : fe + 1] = True
        after_idx = np.where(alarm[fs : fe + 1])[0]
        if first_alarm is None and after_idx.size > 0:
            first_alarm = fs + int(after_idx[0])
    FAR = float((alarm & pre_mask).sum()) / max(1, pre_mask.sum())
    TPR = float((alarm & post_mask).sum()) / max(1, post_mask.sum())
    delay = None if first_alarm is None else int(first_alarm - segs[0][2])
    return dict(FAR=FAR, TPR=TPR, first_alarm=first_alarm, delay=delay, alarm=alarm)


def plot_monitor(spe_ts, CL, segs, title, out_png):
    T = len(spe_ts)
    xs = np.arange(T)
    plt.figure(figsize=(11, 4))
    plt.plot(xs, spe_ts, label='SPE (per-time)', linewidth=1.3)
    plt.axhline(CL, linestyle='--', label='Global CL (per-time)')
    for k, (ns, ne, fs, fe) in enumerate(segs, start=1):
        plt.axvline(ns, color='k', linestyle=':', alpha=0.5)
        plt.axvline(fs, color='r', linestyle=':', alpha=0.6)
        ymax = np.nanmax(spe_ts)
        plt.text((ns + ne) // 2, ymax * 0.05, f'M{k}-Norm', ha='center', va='bottom', fontsize=8)
        plt.text((fs + fe) // 2, ymax * 0.10, f'M{k}-Fault', ha='center', va='bottom', fontsize=8)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('SPE (per-time)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def eval_after_session(session_id, combo_model, learned_modes,
                       normals, faults, L=100, hop=10, alpha=0.995, device=None,
                       mode_stats=None):
    """
    Evaluate after each session using per-time monitoring.
    learned_modes: list like ['M1'] or ['M1','M2'] up to current session
    normals/faults: aligned lists of (path, varname)
    mode_stats: dict like {'M1': (mean, std), ...}. If None, uses global MODE_STATS if imported.
    Returns: CL, metrics, spe_ts, segs
    """
    from tep_data_load import tep_data_load

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    combo_model = combo_model.to(device).eval()

    # Optional MODE_STATS passthrough to avoid global dependency
    if mode_stats is None:
        try:
            from main import MODE_STATS as _MODE_STATS  # fallback to global
            mode_stats = _MODE_STATS
        except Exception:
            raise RuntimeError("mode_stats must be provided if not importing from main.MODE_STATS")

    # Helpers local to this evaluation to avoid cross-mode normalization mixing
    def _make_seg_windows(X, mean, std, L, hop):
        Xn = (X - mean) / (std + 1e-8)
        starts = np.arange(0, Xn.shape[0] - L + 1, hop, dtype=int)
        Xw = np.empty((len(starts), L, Xn.shape[1]), dtype=np.float32)
        for i, s in enumerate(starts):
            Xw[i] = Xn[s : s + L, :]
        X_cnn = torch.tensor(Xw).permute(0, 2, 1).contiguous()
        return X_cnn, starts, Xn.shape[0]

    @torch.no_grad()
    def _win_err(model, X_cnn):
        ld = DataLoader(TensorDataset(X_cnn, X_cnn), batch_size=256, shuffle=False, drop_last=False)
        arr = []
        for xb, _ in ld:
            xb = xb.to(device)
            xhat, _ = model(xb)
            e = ((xb - xhat) ** 2).mean(dim=1).detach().cpu().numpy()
            arr.append(e)
        return np.concatenate(arr, axis=0)  # [Nw,L]

    def _overlap(E, starts, T):
        Nw, L = E.shape
        ssum = np.zeros(T, dtype=np.float64)
        cnt = np.zeros(T, dtype=np.int32)
        for i, s in enumerate(starts):
            ssum[s : s + L] += E[i]
            cnt[s : s + L] += 1
        spe = np.full(T, np.nan, dtype=np.float64)
        m = cnt > 0
        spe[m] = ssum[m] / cnt[m]
        return spe, cnt

    # 1) Build per-mode segments and compute window errors
    E_list, S_list = [], []
    segs, offset = [], 0
    T_total = 0
    for mode, (p_norm, v_norm), (p_fault, v_fault) in zip(learned_modes, normals, faults):
        mean, std = mode_stats[mode]
        Xn = tep_data_load(p_norm, v_norm)[:200, :]
        Xf = tep_data_load(p_fault, v_fault)[:800, :]
        Xseg = np.vstack([Xn, Xf])

        X_cnn, starts, T = _make_seg_windows(Xseg, mean, std, L, hop)
        Ew = _win_err(combo_model, X_cnn)

        E_list.append(Ew)
        S_list.append(starts + offset)
        segs.append((offset, offset + len(Xn) - 1,
                     offset + len(Xn), offset + len(Xn) + len(Xf) - 1))
        offset += T
        T_total += T

    E_all = np.vstack(E_list)
    S_all = np.concatenate(S_list)
    spe_ts, cnt = _overlap(E_all, S_all, T_total)

    # 2) Global per-time CL using full normal segments of learned modes (robust)
    all_spe = []
    for mode, (p_norm, v_norm) in zip(learned_modes, normals):
        mean, std = mode_stats[mode]
        Xn_full = tep_data_load(p_norm, v_norm)
        X_cnn, starts, T = _make_seg_windows(Xn_full, mean, std, L, hop)
        Ew = _win_err(combo_model, X_cnn)
        spe_norm, _ = _overlap(Ew, starts, T)
        part = spe_norm[L - 1 :]
        if part.size:
            all_spe.append(part[np.isfinite(part)])
    cat = np.concatenate(all_spe, axis=0) if all_spe else np.array([], dtype=float)
    CL = float(np.nanquantile(cat, alpha)) if cat.size > 0 else float('nan')

    # 3) Metrics
    alarm = spe_ts > CL
    pre_mask = np.zeros(T_total, dtype=bool)
    post_mask = np.zeros(T_total, dtype=bool)
    first_alarm = None
    for (ns, ne, fs, fe) in segs:
        pre_mask[ns : ne + 1] = True
        post_mask[fs : fe + 1] = True
        idx = np.where(alarm[fs : fe + 1])[0]
        if first_alarm is None and idx.size > 0:
            first_alarm = fs + int(idx[0])
    FAR = float((alarm & pre_mask).sum()) / max(1, pre_mask.sum())
    TPR = float((alarm & post_mask).sum()) / max(1, post_mask.sum())

    # 4) Plot
    xs = np.arange(T_total)
    plt.figure(figsize=(11, 4))
    plt.plot(xs, spe_ts, label='SPE (per-time)', linewidth=1.2)
    if np.isfinite(CL):
        plt.axhline(CL, linestyle='--', label='Global CL (per-time)')
    finite_mask = np.isfinite(spe_ts)
    if finite_mask.any():
        ymax = float(np.nanpercentile(spe_ts[finite_mask], 99.5) * 1.1)
    else:
        ymax = 1.0  # 防御：全 NaN 时给默认值并尽量少画
    for k, (ns, ne, fs, fe) in enumerate(segs, start=1):
        plt.axvline(ns, color='k', linestyle=':', alpha=0.4)
        plt.axvline(fs, color='r', linestyle=':', alpha=0.5)
        if np.isfinite(ymax):
            plt.text((ns + ne) // 2, ymax * 0.05, f'M{k}-Norm', ha='center', va='bottom', fontsize=8)
            plt.text((fs + fe) // 2, ymax * 0.10, f'M{k}-Fault', ha='center', va='bottom', fontsize=8)
    plt.title(f'Session-{session_id} Monitoring (Per-time)')
    plt.xlabel('Time')
    plt.ylabel('SPE (per-time)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'artifacts/monitor_S{session_id}.png', dpi=150)
    plt.close()

    metrics = dict(FAR=FAR, TPR=TPR, first_alarm=first_alarm)
    return CL, metrics, spe_ts, segs
