
import numpy as np
import torch


@torch.no_grad()
def compute_test_spe(model, loader, device=None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    spe_all = []
    for batch in loader:
        # 兼容多种 batch 格式： (xb,yb) | (xb,) | xb | dict{ 'x','y' }
        if isinstance(batch, dict):
            xb = batch.get('x') or batch.get('input') or batch.get('data')
            if xb is None:
                xb = batch
        elif isinstance(batch, (list, tuple)):
            xb = batch[0]
        else:
            xb = batch

        xb = xb.to(device)
        xhat, _ = model(xb)
        spe = ((xb - xhat) ** 2).mean(dim=(1, 2)).cpu().numpy()
        spe_all.append(spe)
    return np.concatenate(spe_all, axis=0)
