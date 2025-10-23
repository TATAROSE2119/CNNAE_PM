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