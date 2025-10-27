import numpy as np
import torch


def _extract_x(batch):
    """Helper to pull input tensors from different batch structures."""
    if isinstance(batch, dict):
        xb = batch.get('x') or batch.get('input') or batch.get('data')
        if xb is None:
            return batch
        return xb
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def compute_metrics_timestep(t_fault_start, spe_ts, CL):
    """
    基于逐时刻误差 spe_ts 与常数 CL 的指标（可视化口径）。
    返回:
      num_alarm_ts: 报警时刻数量
      first_alarm_time_ts: 首次报警时刻（>= t_fault_start）
      delay_ts: 检测延迟（逐时刻口径）
      FAR_ts: 预故障误报率（t < t_fault_start）
      TPR_ts: 故障段检出率（t >= t_fault_start）
      alarm_ts: [T] 布尔序列
    """
    T = len(spe_ts)
    alarm_ts = spe_ts > CL

    # 首次报警（故障之后）
    idx_after = np.where(np.arange(T) >= t_fault_start)[0]
    first_alarm_time_ts = None
    if idx_after.size > 0:
        aft_mask = alarm_ts[idx_after]
        if aft_mask.any():
            first_alarm_time_ts = int(idx_after[np.argmax(aft_mask)])  # 第一个 True 的全局时刻

    delay_ts = None
    if first_alarm_time_ts is not None:
        delay_ts = max(0, first_alarm_time_ts - int(t_fault_start))

    # FAR / TPR
    pre_mask  = np.arange(T) <  t_fault_start
    post_mask = np.arange(T) >= t_fault_start
    FAR_ts = float((alarm_ts & pre_mask).sum()) / max(1, pre_mask.sum())
    TPR_ts = float((alarm_ts & post_mask).sum()) / max(1, post_mask.sum())

    return {
        "num_alarm_ts": int(alarm_ts.sum()),
        "first_alarm_time_ts": first_alarm_time_ts,
        "delay_ts": None if delay_ts is None else int(delay_ts),
        "FAR_ts": FAR_ts,
        "TPR_ts": TPR_ts,
        "alarm_ts": alarm_ts
    }

def overlap_average_to_timeseries(E_win, starts, T):
    """
    将逐窗口逐时间步误差 E_win 叠加到原始时间轴，并做覆盖窗口的平均：
      spe_ts[t] = mean_i E_win[i, t - starts[i]]  (对所有覆盖 t 的窗口 i)
    返回：
      spe_ts: [T]，每个时刻的误差
      cnt_ts: [T]，每个时刻被多少窗口覆盖（用于自检）
    """
    Nw, L = E_win.shape
    spe_sum = np.zeros(T, dtype=np.float64)
    cnt_ts  = np.zeros(T, dtype=np.int32)
    for i in range(Nw):
        s = int(starts[i])
        e_seg = E_win[i]              # 长度 L
        spe_sum[s:s+L] += e_seg
        cnt_ts[s:s+L]  += 1
    # 避免除零（按你的参数，0..T-1 都有覆盖，这里只是保险）
    mask = cnt_ts > 0
    spe_ts = np.full(T, np.nan, dtype=np.float64)
    spe_ts[mask] = spe_sum[mask] / cnt_ts[mask]
    return spe_ts, cnt_ts

@torch.no_grad()
def compute_window_time_errors(model, loader):
    """
    返回：
      E_win: [Nw, L]，每个窗口、每个时间步的通道均方误差
      Nw, L: 便于检查
    说明：
      - loader 必须按窗口顺序（shuffle=False）
      - 模型输入/输出形状 [N, P, L]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    chunks = []
    for batch in loader:
        xb = _extract_x(batch).to(device)
        xhat, _ = model(xb)
        # 对“通道维”求均方误差，保留时间维：得到 [batch, L]
        # (xb - xhat)^2: [N, P, L]  -> mean(dim=1) -> [N, L]
        e = ((xb - xhat) ** 2).mean(dim=1).detach().cpu().numpy()
        chunks.append(e)
    E_win = np.concatenate(chunks, axis=0)  # [Nw, L]
    return E_win, E_win.shape[0], E_win.shape[1]

