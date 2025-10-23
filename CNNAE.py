import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class CNNAE(nn.Module):
    def __init__(self,in_ch:int,latent_ch:int=32):
        """
        1D-CNN 自编码器，用于时序数据的特征提取与重构
         输入形状 [N, in_ch, L]，输出形状 [N, in_ch, L]
        :param in_ch:  输入通道数（变量数）P
         例如 TE 数据的 52 个测点
         latent_ch: 潜在特征通道数（瓶颈层通道数）
        :param latent_ch:  潜在特征通道数（瓶颈层通道数）
        """
        super().__init__() #初始化父类
        # 编码器：逐步下采样提取特征 [N, in_ch, L] -> [N, latent_ch, L/4]
        self.enc=nn.Sequential( #编码器部分,nn.Sequential()用于将多个层组合在一起,下采样两次，步长为2
            nn.Conv1d(in_ch,64,kernel_size=5,stride=2,padding=2), nn.ReLU(), #卷积层+激活函数ReLU，kernel_size卷积核大小，stride步长，padding填充
            nn.Conv1d(64,32,kernel_size=5,stride=2,padding=2), nn.ReLU(), #卷积层+激活函数ReLU
            nn.Conv1d(32,latent_ch,kernel_size=3,stride=1,padding=1), nn.ReLU() #卷积层+激活函数ReLU
        )

        self.dec=nn.Sequential( #解码器部分,上采样还原
            nn.ConvTranspose1d(latent_ch,64,kernel_size=3,stride=1,padding=1), nn.ReLU(), #反卷积层+激活函数ReLU
            nn.Upsample(scale_factor=2,mode='nearest'), #上采样，scale_factor放大倍数
            nn.ConvTranspose1d(64,64,kernel_size=3,stride=1,padding=1), nn.ReLU(), #反卷积层+激活函数ReLU
            nn.Upsample(scale_factor=2,mode='nearest'), #上采样，
            nn.ConvTranspose1d(64,in_ch,kernel_size=3,stride=1,padding=1) #反卷积层，输出 [N, in_ch, L]
        )

    def forward(self,x:torch.Tensor):
        """
        前向传播
        :param x:  输入张量，形状 [N, in_ch, L]
        :return:  输出重构张量 xhat，形状 [N, in_ch, L]；
                    潜在特征张量 z，形状 [N, latent_ch, L/4]
        """
        z=self.enc(x)  #编码器提取潜在特征
        xhat=self.dec(z)  #解码器重构输入
        # 安全对齐：确保xhat与x形状一致（处理上采样可能引入的长度偏差）
        L=x.size(-1)  #获取输入长度
        if xhat.size(-1) > L:
            print("⚠️ 警告：解码器输出长度大于输入长度，正在截断多余部分...")
            xhat=xhat[...,:L]  #截断多余部分
        elif xhat.size(-1) < L:
            print("⚠️ 警告：解码器输出长度小于输入长度，正在补齐缺失部分...")
            pad_size=L - xhat.size(-1)  #计算需要补齐的长度
            xhat=torch.nn.functional.pad(xhat,(0,pad_size),'replicate')  #补齐缺失部分，使用复制边缘值方式
        return xhat, z  #返回重构张量和潜在特征张


def train_epoch(model,loader,optimizer,device):
    """
    训练一个 epoch
    :param model:  待训练模型
    :param loader:  数据加载器
    :param optimizer:  优化器
    :param device:  计算设备（CPU 或 GPU）
    :return:  平均训练损失
    """
    model.train()#设置模型为训练模式
    crit = nn.MSELoss()#均方误差损失函数,用于回归任务
    total_loss,n=0.0,0 #初始化总损失和样本数
    for batch in loader:# xb,yb是一个批次的数据和标签
        # 兼容多种 batch 格式： (xb,yb) | (xb,) | xb | dict{ 'x','y' }
        if isinstance(batch, dict):
            xb = batch.get('x') or batch.get('input') or batch.get('data')
            yb = batch.get('y') or batch.get('target') or xb
            if xb is None:
                # 回退：将整个 dict 当作 xb（不太常见）
                xb = batch
                yb = xb
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                xb = batch[0]
                yb = xb
        else:
            xb = batch
            yb = xb


        xb,yb=xb.to(device),yb.to(device) #将数据和标签移动到指定设备,如GPU
        optimizer.zero_grad() #清空梯度
        xhat,_=model(xb) #前向传播，获取重构结果
        loss=crit(xhat,yb) #计算损失
        loss.backward() #反向传播计算梯度
        optimizer.step() #更新模型参数
        bs=xb.size(0) #获取当前批次大小
        total_loss+=loss.item()*bs #累加总损失
        n+=bs #累加样本数
    return total_loss/max(1,n) #返回平均损失

@torch.no_grad()#装饰器，表示该函数在执行时不计算梯度
def eval_epoch(model,loader,device):
    """
    评估一个 epoch
    :param model:  待评估模型
    :param loader:  数据加载器
    :param device:  计算设备（CPU 或 GPU）
    :return:  平均评估损失
    """
    model.eval() #设置模型为评估模式
    crit = nn.MSELoss() #均方误差损失函数
    total_loss,n=0.0,0 #初始化总损失和样本数
    for batch  in loader: # xb,yb是一个批次的数据和标签
        # 兼容多种 batch 格式： (xb,yb) | (xb,) | xb | dict{ 'x','y' }
        if isinstance(batch, dict):
            xb = batch.get('x') or batch.get('input') or batch.get('data')
            yb = batch.get('y') or batch.get('target') or xb
            if xb is None:
                # 回退：将整个 dict 当作 xb（不太常见）
                xb = batch
                yb = xb
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                xb = batch[0]
                yb = xb
        else:
            xb = batch
            yb = xb


        xb,yb=xb.to(device),yb.to(device) #将数据和标签移动到指定设备,如GPU
        xhat,_=model(xb) #前向传播，获取重构结果
        loss=crit(xhat,yb) #计算损失
        bs=xb.size(0) #获取当前批次大小
        total_loss+=loss.item()*bs #累加总损失
        n+=bs #累加样本数
    return total_loss/max(1,n) #返回平均损失

def fit_cnn_ae(model,train_loader,val_loader,epochs=50,lr=1e-3,weight_decay=0.0,patience=8,device=None):
    """
    训练 1D-CNN 自编码器模型，包含早停机制
    早停机制：如果验证集损失在连续 `patience` 个 epoch
    :param model:   待训练模型
    :param train_loader:  训练数据加载器
    :param val_loader:  验证数据加载器
     没有改善，则停止训练以防止过拟合
    :param epochs:  最大训练 epoch 数
    :param lr:   学习率
    :param weight_decay: 权重衰减（L2 正则化系数）
    :param patience:  早停耐心值
     在多少个 epoch 内验证损失没有改善则停止训练
     如果设置为 0，则不使用早停机制
     如果设置为负数，则无限期训练直到达到最大 epoch 数
    :param device:  计算设备（CPU 或 GPU）
     如果为 None，则自动选择可用的 GPU，否则使用 CPU
    :return:  训练好的模型
    """
    if torch.cuda.is_available(): # 检查是否有可用的 GPU
        device=torch.device("cuda") # 使用 GPU 进行计算
    else: # 没有可用的 GPU
        device=torch.device("cpu") # 使用 CPU 进行计算
    model.to(device) # 将模型移动到指定设备
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay) # Adam 优化器

    best_val=math.inf # 初始化最佳验证损失为无穷大
    best_sd=None # 最佳模型参数状态字典
    wait=0 # 早停计数器
    history=[] # 训练历史记录

    for epoch in range(1,epochs+1): # 迭代每个 epoch
        tr=train_epoch(model,train_loader,optimizer,device) # 训练一个 epoch 并获取训练损失
        val=eval_epoch(model,val_loader,device) # 评估验证集损失
        history.append((tr,val))# 记录训练和验证损失

        improved=val< best_val - 1e-6 # 检查验证损失是否有改善
        if improved:
            best_val=val
            best_sd={k:v.cpu().clone() for k,v in model.state_dict().items()} # 保存最佳模型参数
            wait=0 # 重置早停计数器
        else:
            wait+=1 # 增加早停计数器
        print(f"[Epoch {epoch:03d}] train_mse={tr:.6f}  val_mse={val:.6f}  {'*' if improved else ''}")# 打印训练和验证损失

        if wait>patience:
            print(f"⚠️ 早停触发：验证损失在连续 {patience} 个 epoch 内没有改善，停止训练。")
            break
    if best_sd is not None: # 如果存在最佳模型参数
        model.load_state_dict(best_sd) # 加载最佳模型参数
    return model,history  # 返回训练好的模型