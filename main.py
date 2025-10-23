from tep_data_load import *
from prepare_windows_for_cnn import *
from CNNAE import *

if __name__ == "__main__":
    # 测试数据加载功能,查看输出形状,加载数据
    X=tep_data_load('TE_data/M1/m1d00.mat','m1d00')
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


