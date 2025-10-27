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
    X_fault=tep_data_load("TE_data/M1/m1d04.mat",'m1d04')
    print(X_fault_in_normal.shape)
    print(X_fault.shape)
    X_fault_seq=np.vstack([X_fault_in_normal[:200,:],X_fault[:400,:]])
    t_fault_start=200;
    print("拼接后数据形状:", X_fault_seq.shape)

    # 2️⃣ 标准化（用训练阶段的 mean/std）
    Xn = (X_fault_seq - mean) / (std + 1e-8)
    # 3️⃣ 构造滑窗
    L, hop = 100, 10  # 窗口长度 L 和滑动步长 hop
    starts = np.arange(0, Xn.shape[0] - L + 1, hop)  # 计算所有窗口的起始索引（包含最后一个完整窗口）
    X_win = np.empty((len(starts), L, Xn.shape[1]), dtype=np.float32)  # 预分配数组存放所有窗口，形状为 (窗口数, 窗口长度, 变量数)
    for i, s in enumerate(starts):  # 遍历每个起始索引
        X_win[i] = Xn[s:s + L, :]  # 从标准化后的序列中按起始索引切出长度为 L 的子序列并保存到 X_win

    # 转为 \[N, P, L\]
    X_cnn = torch.tensor(X_win).permute(0, 2, 1).contiguous()  # 转为张量并将轴顺序改为 (N, P, L)，使用 contiguous 保证内存连续
    eval_loader = DataLoader(TensorDataset(X_cnn, X_cnn), batch_size=256, shuffle=False)  # 构造评估用 DataLoader，输入与目标相同（用于重构/推断）

    # 4️⃣ 推理并计算 SPE

    spe_test = compute_test_spe(model, eval_loader)
    print(f"SPE_test 形状: {spe_test.shape}")

    # 5️⃣ 映射窗口时间轴
    t_points = starts + (L - 1)   # 窗口末端时间点
    alarm_mask = spe_test > CL
    t_alarm = t_points[alarm_mask]
    print(f"触发报警数量: {alarm_mask.sum()}")


    T = 600
    spe_full_step, cl_full_step = expand_spe_to_time_series(spe_test, t_points, T, mode='step', cl=CL)
    spe_full_lin,  cl_full_lin  = expand_spe_to_time_series(spe_test, t_points, T, mode='linear', cl=CL)




    metrics = compute_metrics_window(t_points, spe_test, CL, t_fault_start)
    print(f"[窗口口径] 报警数: {metrics['num_alarms']}")
    print(f"[窗口口径] 首次报警时间: {metrics['first_alarm_time']}")
    print(f"[窗口口径] 检测延迟: {metrics['delay']} 样本")
    print(f"[窗口口径] 预故障误报率 FAR: {metrics['FAR']:.4f}")
    print(f"[窗口口径] 故障段检出率 TPR: {metrics['TPR']:.4f}")

    plt.figure(figsize=(10,4))
    plt.plot(np.arange(600), spe_full_step, label='SPE (expanded, step)', linewidth=1.2)
    if cl_full_step is not None:
        plt.plot(np.arange(600), cl_full_step, 'r--', linewidth=2, label='Control Limit')
    plt.axvline(t_fault_start, color='k', linestyle=':', label='Fault Start')
    plt.xlabel('Time index')
    plt.ylabel('SPE (per-step, expanded)')
    plt.title('CNN-AE Monitoring (Expanded for Visualization)')
    plt.legend()
    plt.tight_layout()
    plt.show()
