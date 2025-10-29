# memory.py
import random
from collections import defaultdict
import torch

class MemoryBank:
    def __init__(self, total_cap=300):
        self.total_cap = total_cap
        self.bank = defaultdict(list)  # mode_id -> list of (x,y or None)

    def add(self, mode_id, xs, ys=None, k=50):
        buf = self.bank[mode_id]
        for i in range(min(k, xs.size(0))):
            buf.append((xs[i].cpu(), None if ys is None else ys[i].cpu()))
        self._rebalance()

    def sample_mixed(self, except_mode=None, n=64):
        pool = []
        for m, items in self.bank.items():
            if m == except_mode: continue
            pool.extend(items)
        if not pool: return None, None
        picks = random.sample(pool, min(n, len(pool)))
        x = torch.stack([p[0] for p in picks], 0)
        y = None if picks[0][1] is None else torch.stack([p[1] for p in picks], 0)
        return x, y

    def _rebalance(self):
        # 均分到每个已学工况
        modes = list(self.bank.keys())
        if not modes: return
        per = max(1, self.total_cap // len(modes))
        for m in modes:
            self.bank[m] = self.bank[m][-per:]
