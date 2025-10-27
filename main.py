import os

import numpy as np
from dask.array import vstack, append

from tep_data_load import *
from prepare_windows_for_cnn import *
from CNNAE import *
from compute_cl_and_spe import *
from compute_test_spe import *
from expand_spe_to_time_series import  *
from compute_metrics_window import *

import matplotlib.pyplot as plt



if __name__ == "__main__":
    # 测试数据加载功能,查看输出形状,加载数据
    X=tep_data_load('TE_data/M1/m1d00.mat','m1d00')
    # 保留X的1~600个样本


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
    model,hist=fit_cnn_ae(model,train_loader,val_loader,epochs=50,lr=1e-3,patience=8)

    CL,spe_train=compute_cl_and_spe(model,train_loader)
    print("训练集 SPE 控制限 CL =", CL)

    # 加载测试数据集
    # 1️⃣ 拼接：前200个正常 + 前400个故障
    X_fault_in_normal=tep_data_load('TE_data/M1/m1d00.mat','m1d00')
    os.makedirs('results', exist_ok=True)

    fault_ids = range(1, 22)  # m1d01 .. m1d21
    t_fault_start = 200
    L, hop = 100, 10
    T = 600

    for idx in fault_ids:
        name = f"m1d{idx:02d}"
        path = os.path.join('TE_data', 'M1', f"{name}.mat")
        print(f"处理故障样本: {name}  文件: {path}")

        # 加载故障数据
        X_fault = tep_data_load(path, name)
        # 拼接：前200个正常 + 前400个故障（与原逻辑一致）
        X_fault_seq = np.vstack([X_fault_in_normal[:200, :], X_fault[:400, :]])
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

        # 推理并计算窗口级 SPE
        spe_test = compute_spe(model, eval_loader)
        t_points = starts + (L - 1)

        # 展开为时间序列并计算指标
        spe_full_step, cl_full_step = expand_spe_to_time_series(spe_test, t_points, T, mode='step', cl=CL)
        metrics = compute_metrics_window(t_points, spe_test, CL, t_fault_start)

        print(
            f"[{name}] 报警数: {metrics['num_alarms']}, 首次报警: {metrics['first_alarm_time']}, 延迟: {metrics['delay']}, FAR: {metrics['FAR']:.4f}, TPR: {metrics['TPR']:.4f}")

        # 可视化并保存
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(T), spe_full_step, label='SPE (expanded, step)', linewidth=1.2)
        if cl_full_step is not None:
            plt.plot(np.arange(T), cl_full_step, 'r--', linewidth=2, label='Control Limit')
        plt.axvline(t_fault_start, color='k', linestyle=':', label='Fault Start')
        plt.xlabel('Time index')
        plt.ylabel('SPE')
        plt.title(f'CNN-AE Monitoring - {name}')
        plt.legend()
        plt.tight_layout()
        outpath = os.path.join('results', f'{name}_spe.png')
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"保存图像: {outpath}")