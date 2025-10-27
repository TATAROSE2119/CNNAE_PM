import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset



def prepare_windows_for_cnn(X:np.array,L:int=10,hop:int=10,batch_size:int=64,val_ratio:float=0.2):
    # X: 原始时序数据，形状 [N, P]
    # L: 窗口长度
    # hop: 窗口滑动步长
    # batch_size: 批大小
    # val_ratio: 验证集比例

    # （0）自检形状 & 自动转置
    if X.ndim != 2: #.ndim 获取数组的维度数量
        raise ValueError("输入数据 X 应为二维数组，形状 [N, P] 但是检测到形状：{}".format(X.shape))
    T, P = X.shape
    if T < P:
        print("⚠️ 警告：输入数据行数小于列数，可能需要转置。正在自动转置...")
        X = X.T
        T, P = X.shape
        print("转置后形状：", X.shape)
    # (1) 标准化
    mean=X.mean(axis=0,keepdims=True)#axis=0表示按列计算均值，keepdims保持二维形状
    std=X.std(axis=0,keepdims=True)#axis=0表示按列计算标准差，keepdims保持二维形状
    Xn=(X-mean)/std

    # (2) 切滑动窗口 [N_win, L, P]
    idx_starts=np.arange(0,T-L+1,hop,dtype=int)#生成窗口起始索引,dtype=int表示索引为整数类型
    N_windows=len(idx_starts) #窗口总数
    X_windows=np.empty((N_windows,L,P),dtype=np.float32)#初始化窗口数组
    for i,idx in enumerate(idx_starts):#enumerate()函数用于同时获取索引和值
        X_windows[i]=Xn[idx:idx+L,:]

    # (3)转成 CNN 需要的形状 [N_win, C, L]，把 P 当作“通道”
    X_cnn=torch.tensor(X_windows).permute(0,2,1).contiguous() # tensor().permute()改变维度顺序，contiguous()确保内存连续性

    # (4) 划分训练/验证集，无监督重构任务，输入等于输出
    N = X_cnn.size(0)
    n_train = int(N * (1.0 - val_ratio))
    train_idx = torch.arange(0, n_train)  # 按时间前段
    val_idx = torch.arange(n_train, N)  # 按时间后段

    X_train = X_cnn[train_idx]
    X_val = X_cnn[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, X_train),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, X_val),
        batch_size=batch_size, shuffle=False, drop_last=False  # 验证：不打乱、不丢尾
    )
    return train_loader, val_loader,(mean.astype(np.float32)),(std.astype(np.float32))#返回训练/验证数据加载器及标准化参数