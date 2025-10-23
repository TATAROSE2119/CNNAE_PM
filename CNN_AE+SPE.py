# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ========= 1) 数据准备 =========
# X_train_norm: 正常工况训练数据，形状 [N_train, L, P] （样本数, 时间窗长度, 变量数）
# X_test: 待监测数据（正常+故障），形状 [N_test, L, P]
# 你需要把自己的 TE 数据切成固定窗口。下面用随机数占位：
P, L = 52, 100         # 例如 TE 的 52 个测点，每窗 100 步
N_train, N_test = 2000, 800
np.random.seed(0)
X_train_norm = np.random.randn(N_train, L, P).astype(np.float32)
X_test = np.random.randn(N_test, L, P).astype(np.float32)

# 转成 [N, C, L] 的 CNN 输入（把变量 P 当作“通道”更合理；若相反也可以）
X_train_t = torch.tensor(X_train_norm).permute(0,2,1)  # [N, P, L]
X_test_t = torch.tensor(X_test).permute(0,2,1)         # [N, P, L]

train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=64, shuffle=True)

# ========= 2) 模型定义：1D-CNN 自编码器 =========
class CNNAE(nn.Module):
    def __init__(self, in_ch=P, latent_ch=16):
        super().__init__()
        # Encoder: [N, P, L] -> [N, latent_ch, L/4]（步长2下采样两次）
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(32, latent_ch, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )
        # Decoder: 上采样还原
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(latent_ch, 32, kernel_size=4, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, in_ch, kernel_size=4, stride=2, padding=1) # 输出 [N,P,L]
        )
    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z

model = CNNAE()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ========= 3) 训练（仅正常数据） =========
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

for epoch in range(10):  # 真实训练建议 50~200 epoch
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        xhat, _ = model(xb)
        loss = crit(xhat, yb)
        loss.backward()
        opt.step()
        running += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}, recon MSE={running/len(train_loader.dataset):.6f}")

# ========= 4) 统计量计算：SPE/Q =========
@torch.no_grad()
def compute_spe(model, X):
    model.eval()
    X = X.to(device)
    xhat, _ = model(X)
    # MSE over channels & time: [N,P,L] -> [N]
    spe = ((X - xhat)**2).mean(dim=(1,2)).detach().cpu().numpy()
    return spe

spe_train = compute_spe(model, X_train_t)
spe_test = compute_spe(model, X_test_t)

# ========= 5) 控制限（CL）估计 =========
alpha = 0.995  # 99.5% 控制限
CL = np.quantile(spe_train, alpha)

# ========= 6) 画“统计量 + 控制限”监测图 =========
plt.figure(figsize=(10,4))
plt.plot(spe_test, label='SPE (Q) statistic')
plt.axhline(CL, linestyle='--', label=f'CL @ {alpha*100:.1f}%', linewidth=2)
alarm_idx = np.where(spe_test > CL)[0]
plt.scatter(alarm_idx, spe_test[alarm_idx], marker='x', label='Alarm', zorder=3)
plt.xlabel('Window index'); plt.ylabel('Reconstruction MSE')
plt.title('CNN-AE Monitoring (SPE with Control Limit)')
plt.legend(); plt.tight_layout()
plt.show()
