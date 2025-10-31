import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, x, y, w, h, text, fontsize=10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        fc="#f5f5f5", ec="#333333", lw=1.0
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha='center', va='center', fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='-|>', mutation_scale=12,
                          lw=1.0, ec="#333333", fc="#333333")
    ax.add_patch(arr)


def make_flowchart(out_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_axis_off()

    # Left column: training pipeline per session
    w, h = 0.28, 0.12
    x0 = 0.07
    y = [0.85, 0.68, 0.51, 0.34, 0.17]
    texts = [
        "Data Load / Preprocess\n(TE, column filter, truncate, per-mode standardization)",
        "Sliding Windows\nL=100, hop=10 → [N, P, L]",
        "Session Training (Mi)\nReconstruction MSE + Memory Distillation",
        "Window-wise CL (α=0.99)\nRecord loss, CL; Update MemoryBank",
        "Save Artifacts\nbackbone.pt, decoder.pt,\nstandardize_params.npz, cls.json"
    ]
    for i, yy in enumerate(y):
        draw_box(ax, x0, yy, w, h, texts[i], fontsize=10)
        if i > 0:
            arrow(ax, x0 + w / 2.0, y[i - 1], x0 + w / 2.0, yy + h)

    # Right column: per-time monitoring and evaluation
    xr = 0.58
    yR = [0.74, 0.57, 0.40, 0.23]
    textsR = [
        "Build Concatenated Sequences\nper learned mode: Normal(≈200)+Fault(≈800)",
        "Per-window errors → Overlap-average\n→ SPE per-time",
        "Global Per-time CL (α=0.995)\nfrom full normal segments",
        "Metrics & Plots\nFAR, TPR, First Alarm, Delay\nmonitor_S{session}.png"
    ]
    for i, yy in enumerate(yR):
        draw_box(ax, xr, yy, w, h, textsR[i], fontsize=10)
        if i > 0:
            arrow(ax, xr + w / 2.0, yR[i - 1], xr + w / 2.0, yy + h)

    # Cross connections
    arrow(ax, x0 + w, y[2] + h / 2.0, xr, yR[0] + h / 2.0)
    arrow(ax, xr + w / 2.0, yR[-1], x0 + w / 2.0, y[-1] + h)

    # Loop to next session
    ax.text(x0 + w / 2.0, 0.08, "Next Session (Mi+1)", ha='center', va='center', fontsize=9, color="#333")
    arrow(ax, x0 + w / 2.0, 0.17, x0 + w / 2.0, 0.12)
    arrow(ax, x0 + w / 2.0, 0.12, x0 + w / 2.0, 0.97)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_pseudocode(out_path):
    code = r'''procedure Main()
  set SEED, DEVICE; set {L, hop, batch, epochs, lr}
  MODE_STATS ← {}
  backbone, decoder ← init_models(P from M1)
  mem ← MemoryBank(total_cap=300); teacher ← None
  for mode in [M1, M2, M3]:
    X ← load_normal(mode)
    train_loader, val_loader, (mean, std) ← windowize_and_norm(X, L, hop)
    MODE_STATS[mode] ← (mean, std)
    teacher, hist ← train_one_session(
      mode, backbone, decoder, teacher, mem,
      train_loader, val_loader, epochs, lr, DEVICE)
    plot_history(hist, "loss_"+mode)
    combo ← CombinedAE(backbone, decoder)
    CL_window ← quantile(window_errors(combo, train_loader), 0.99)
    mem.add(mode, sample(train_loader), k=50)
    learned ← modes up to current
    (CL_time, metrics, spe_ts, segs) ← eval_after_session(
      session_id, combo, learned, normals, faults,
      L, hop, alpha=0.995, mode_stats=MODE_STATS)
    plot_monitor(spe_ts, CL_time, segs, "S"+session_id)
  save_artifacts(backbone, decoder, MODE_STATS, {CL_window})

procedure train_one_session(mode, backbone, decoder, teacher, mem,
  train_loader, val_loader, epochs, lr, device)
  for ep in 1..epochs:
    for xb in train_loader:
      z_s ← backbone(xb); xhat ← decoder(z_s)
      loss ← mse(xhat, xb)
      if teacher and mem:
        xb_old ← mem.sample(except=mode)
        if xb_old:
          z_t ← teacher.bb(xb_old)
          loss ← loss + mse(backbone(xb_old), z_t)
      update(loss)
    validate on val_loader
  return snapshot(backbone, decoder) as teacher, history

procedure eval_after_session(session_id, combo, learned, normals, faults, L, hop, alpha, mode_stats)
  E_all, starts_all, T_total ← []
  segs, offset ← [], 0
  for mode in learned:
    (mean, std) ← mode_stats[mode]
    X_seg ← concat(normal[:200], fault[:800])
    E_win ← window_errors(combo, windowize(X_seg, mean, std, L, hop))
    collect E_win, starts+offset, update segs, offset
  spe_ts ← overlap_average(E_all, starts_all, T_total)
  CL_time ← quantile(concat spe_ts on full normal segments, alpha)
  metrics ← FAR/TPR/first_alarm/delay from spe_ts > CL_time
  save monitor_S{session_id}.png; return CL_time, metrics, spe_ts, segs
'''
    fig_w, fig_h = 12, 16
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.text(0.02, 0.98, code, ha='left', va='top', family='monospace', fontsize=9, wrap=True)
    fig.tight_layout(pad=0.6)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    make_flowchart("artifacts/flowchart_pipeline.png")
    make_pseudocode("artifacts/pseudocode_pipeline.png")
    print("Generated artifacts/flowchart_pipeline.png and artifacts/pseudocode_pipeline.png")

