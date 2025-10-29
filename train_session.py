# train_session.py
import copy, torch
from torch.utils.data import DataLoader, TensorDataset
from kd_losses import feat_distill, logits_distill, recon_loss

def train_one_session(
    mode_id,                        # 当前会话工况ID: 'M1'/'M2'/...
    model_backbone, decoder=None,   # 共享编码器 + （可选）解码器
    clf_head=None,                  # （可选）分类头
    teacher=None,                   # 上一会话冻结模型快照: {'bb':..,'dec':..,'clf':..}
    mem=None,                       # MemoryBank
    train_loader=None, val_loader=None,   # 本工况 DataLoader（按时间切分）
    epochs=80, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu',
    kd_w_feat=1.0, kd_w_logits=0.5, rec_w=1.0, sup_w=1.0
):
    model_backbone.to(device).train()
    if decoder is not None: decoder.to(device).train()
    if clf_head is not None: clf_head.to(device).train()
    # 将教师模型移动到目标设备，避免因设备不一致导致的 dtype 错误
    if teacher is not None:
        teacher = {
            key: (module.to(device) if module is not None else None)
            for key, module in teacher.items()
        }

    params = list(model_backbone.parameters())
    if decoder is not None: params += list(decoder.parameters())
    if clf_head is not None: params += list(clf_head.parameters())
    optim = torch.optim.Adam(params, lr=lr)

    history = []
    for ep in range(1, epochs+1):
        tr_loss = 0.0; n=0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = (None if yb is None else yb.to(device))
            optim.zero_grad()
            z = model_backbone(xb)
            loss = 0.0

            # 重构
            if decoder is not None:
                xhat = decoder(z)
                # 对齐长度（安全裁切/填充）
                if xhat.size(-1) > xb.size(-1): xhat = xhat[..., :xb.size(-1)]
                elif xhat.size(-1) < xb.size(-1):
                    pad = xb.size(-1) - xhat.size(-1)
                    xhat = torch.nn.functional.pad(xhat, (0,pad))
                loss = loss + recon_loss(xhat, xb, rec_w)

            # 监督（若有标签与分类头）
            if (yb is not None) and (clf_head is not None):
                logits = clf_head(z.mean(dim=-1))  # 简化：对时序做平均池化
                loss = loss + sup_w * torch.nn.functional.cross_entropy(logits, yb)

            # 记忆库蒸馏（稳住旧工况）
            if (teacher is not None) and (mem is not None):
                xb_old, yb_old = mem.sample_mixed(except_mode=mode_id, n=xb.size(0))
                if xb_old is not None:
                    xb_old = xb_old.to(device)
                    with torch.no_grad():
                        z_t = teacher['bb'](xb_old)
                        logits_t = None
                        if (clf_head is not None) and (teacher.get('clf', None) is not None):
                            logits_t = teacher['clf'](z_t.mean(dim=-1))
                    z_s = model_backbone(xb_old)
                    loss = loss + feat_distill(z_s, z_t, kd_w_feat)
                    if (clf_head is not None) and (teacher.get('clf', None) is not None):
                        logits_s = clf_head(z_s.mean(dim=-1))
                        loss = loss + logits_distill(logits_s, logits_t, T=2.0, w=kd_w_logits)

            loss.backward()
            optim.step()
            tr_loss += loss.item() * xb.size(0); n += xb.size(0)

        # 验证（重构或分类MSE/CE的平均）
        model_backbone.eval()
        if decoder is not None: decoder.eval()
        if clf_head is not None: clf_head.eval()
        import torch.nn.functional as F
        va_loss=0.0; m=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = (None if yb is None else yb.to(device))
                z = model_backbone(xb)
                loss = 0.0
                if decoder is not None:
                    xhat = decoder(z)
                    if xhat.size(-1) > xb.size(-1): xhat = xhat[..., :xb.size(-1)]
                    elif xhat.size(-1) < xb.size(-1):
                        pad = xb.size(-1) - xhat.size(-1)
                        xhat = torch.nn.functional.pad(xhat, (0,pad))
                    loss += F.mse_loss(xhat, xb)
                if (yb is not None) and (clf_head is not None):
                    logits = clf_head(z.mean(dim=-1))
                    loss += F.cross_entropy(logits, yb)
                va_loss += loss.item() * xb.size(0); m += xb.size(0)
        history.append((tr_loss/max(1,n), va_loss/max(1,m)))

        model_backbone.train()
        if decoder is not None: decoder.train()
        if clf_head is not None: clf_head.train()

    # 返回当前会话的“教师快照”与曲线
    teacher_next = {
        'bb': copy.deepcopy(model_backbone).cpu().eval(),
        'dec': None if decoder is None else copy.deepcopy(decoder).cpu().eval(),
        'clf': None if clf_head is None else copy.deepcopy(clf_head).cpu().eval(),
    }
    return teacher_next, history
