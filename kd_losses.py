# kd_losses.py
import torch
import torch.nn.functional as F
#最小蒸馏 + 重构联合
def feat_distill(z_student, z_teacher, w=1.0):
    return w * F.mse_loss(z_student, z_teacher)

def logits_distill(logits_s, logits_t, T=2.0, w=0.5):
    ps = F.log_softmax(logits_s / T, dim=1)
    pt = F.softmax(logits_t / T, dim=1)
    return w * (T*T) * F.kl_div(ps, pt, reduction='batchmean')

def recon_loss(x_hat, x, w=1.0):
    return w * F.mse_loss(x_hat, x)
