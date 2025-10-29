import os

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from tep_data_load import *
from prepare_windows_for_cnn import *
from CNNAE import *
from compute_cl_and_spe import *

from compute_window_time_errors import *

import matplotlib.pyplot as plt



if __name__ == "__main__":
    # 测试数据加载功能,查看输出形状,加载数据
    X=tep_data_load('TE_data/M2/m2d00.mat','m2d00')
    # 保留X的1~600个样本
    X=X[:2000,:]
    print("用于训练的样本数：", X.shape[0])

    train_epoch=200


    # 把连续时序 X 规范化并切成固定长度的滑动窗口，再整理成 CNN 需要的张量形状。
    train_loader, val_loader, mean, std = prepare_windows_for_cnn(X,L=100,hop=10,batch_size=64,val_ratio=0.2)
    print("训练集批次数量：", len(train_loader))
    for xb in train_loader:
        print("一个批次的形状：", xb[0].shape)  # xb 是一个元组，包含输入张量
        break

    P=X.shape[1]
    print("变量数 P =", P)
    xb=next(iter(train_loader))
    #print("输入张量形状：", xb.shape)  # xb 是一个元组，包含输入张量

    model = CNNAE(in_ch=P, latent_ch=32)
    model,hist=fit_cnn_ae(model,train_loader,val_loader,epochs=train_epoch,lr=1e-3,patience=8)
    # 将历史拆分为数组
    train_curve = np.array([h[0] for h in hist], dtype=float)
    val_curve = np.array([h[1] for h in hist], dtype=float)
    epochs_axis = np.arange(1, len(hist) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_axis, train_curve, label='train_mse', linewidth=1.8)
    plt.plot(epochs_axis, val_curve, label='val_mse', linewidth=1.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training & Validation MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    plt.show()

    print("学习曲线已保存：loss_curve.png")




    CL,spe_train=compute_cl_and_spe(model,train_loader)
    print("训练集 SPE 控制限 CL =", CL)

    # 加载测试数据集
    # 1️⃣ 拼接：前200个正常 + 前400个故障
    X_fault_in_normal=tep_data_load('TE_data/M2/m2d00.mat','m2d00')
    os.makedirs('results', exist_ok=True)

    fault_ids = range(1, 22)  # m1d01 .. m1d21
    t_fault_start = 200
    L, hop = 100, 10
    T = 1000

    for idx in fault_ids:
        name = f"m2d{idx:02d}"
        path = os.path.join('TE_data', 'M2', f"{name}.mat")
        print(f"处理故障样本: {name}  文件: {path}")

        # 加载故障数据
        X_fault = tep_data_load(path, name)
        # 拼接：前200个正常 + 前400个故障（与原逻辑一致）
        X_fault_seq = np.vstack([X_fault_in_normal[:200, :], X_fault[:800, :]])
        print(f"拼接后数据形状: {X_fault_seq.shape}")

        # 标准化（用训练阶段的 mean/std）
        Xn = (X_fault_seq - mean) / (std + 1e-8)

        # 构造滑窗
        starts = np.arange(0, Xn.shape[0] - L + 1, hop)
        X_win = np.empty((len(starts), L, Xn.shape[1]), dtype=np.float32)
        for i, s in enumerate(starts):
            X_win[i] = Xn[s:s + L, :]

        # 转为 [N, P, L] 并构造 DataLoader
        X_cnn = torch.tensor(X_win).permute(0, 2, 1).contiguous()
        eval_loader = DataLoader(TensorDataset(X_cnn, X_cnn), batch_size=256, shuffle=False)


        # 方案 B：逐时间步误差重叠平均
        E_win, Nw, Lw = compute_window_time_errors(model, eval_loader)
        print("E_win shape =", E_win.shape)  # 期望: (len(starts), L)
        spe_ts, cnt_ts = overlap_average_to_timeseries(E_win, starts, T)
        print("spe_ts shape =", spe_ts.shape, " min(cnt)=", cnt_ts.min(), " max(cnt)=", cnt_ts.max())
        metrics_ts = compute_metrics_timestep(t_fault_start=t_fault_start, spe_ts=spe_ts, CL=CL)
        print("[逐时刻] 报警数:", metrics_ts["num_alarm_ts"])
        print("[逐时刻] 首次报警时间:", metrics_ts["first_alarm_time_ts"])
        print("[逐时刻] 检测延迟:", metrics_ts["delay_ts"])
        print("[逐时刻] FAR:", metrics_ts["FAR_ts"], "TPR:", metrics_ts["TPR_ts"])
        plt.figure(figsize=(11, 4))
        plt.plot(np.arange(T), spe_ts, label='SPE (per-time, overlap-avg)', linewidth=1.3)
        plt.axhline(CL, color='r', linestyle='--', linewidth=2, label='Control Limit')
        plt.axvline(t_fault_start, color='k', linestyle=':', label='Fault Start')
        alarm_idx = np.where(metrics_ts["alarm_ts"])[0]
        if alarm_idx.size > 0:
            plt.scatter(alarm_idx, spe_ts[alarm_idx], s=12, marker='o', label='Alarm (ts)', zorder=3)
        plt.xlabel('Time index')
        plt.ylabel('SPE (per-time)')
        plt.title(f'CNN-AE Monitoring — {name} (Overlap Avg)')
        plt.legend()
        plt.tight_layout()
        plt.show()
