import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def make_eval_loader_from_dataset(tds: TensorDataset, batch_size: int = 256):
    """Create a deterministic, non-dropping DataLoader for evaluation.
    Expects a TensorDataset compatible with (x, y) or (x, x) pairs.
    """
    return DataLoader(tds, batch_size=batch_size, shuffle=False, drop_last=False)


def plot_history(hist, title: str, out_png: str):
    """Plot train/val loss curves and save to file."""
    tr = np.array([h[0] for h in hist], dtype=float)
    va = np.array([h[1] for h in hist], dtype=float)
    ep = np.arange(1, len(hist) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ep, tr, label='train_loss', linewidth=1.8)
    plt.plot(ep, va, label='val_loss', linewidth=1.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE/CE)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def prep_windows_compat(X, L: int, hop: int, batch_size: int, val_ratio: float):
    """Wrapper to normalize return signature from prepare_windows_for_cnn.
    Returns: train_loader, val_loader, (mean, std)
    """
    from prepare_windows_for_cnn import prepare_windows_for_cnn

    out = prepare_windows_for_cnn(
        X, L=L, hop=hop, batch_size=batch_size, val_ratio=val_ratio
    )
    if isinstance(out, tuple) and len(out) == 4:
        train_loader, val_loader, mean, std = out
    elif isinstance(out, tuple) and len(out) == 3:
        train_loader, val_loader, ms = out
        mean, std = ms
    else:
        raise RuntimeError(
            f"prepare_windows_for_cnn 返回未知签名：len={len(out)}"
        )
    return train_loader, val_loader, (mean, std)

