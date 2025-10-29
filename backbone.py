# backbone.py
import torch.nn as nn
# 从 CNNAE 里拆编码器
class CNNBackbone(nn.Module):
    def __init__(self, in_ch: int, feat_ch: int = 64, latent_ch: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, feat_ch, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # L -> L/2
            nn.Conv1d(feat_ch, feat_ch, kernel_size=5, stride=2, padding=2), nn.ReLU(),# L/2 -> L/4
            nn.Conv1d(feat_ch, latent_ch, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
    def forward(self, x):  # x: [N,P,L]
        return self.enc(x) # z: [N,latent_ch,L/4]
