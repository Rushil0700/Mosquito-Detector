"""M2 visual check -> results/figures/m2_features.png

Row per target species (+ background): one real training window, one synthetic window (log-mel), and the median
log-mel band profile of all real vs synthetic training windows of that class.

Usage: python -m src.report_m2
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REAL, SYN = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
NAMES = {"ae_aegypti": "Ae. aegypti", "ae_albopictus": "Ae. albopictus", "an_stephensi": "An. stephensi",
         "cx_quinquefasciatus": "Cx. quinquefasciatus", "background": "Background (no mosquito)"}


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def main():
    cfg = yaml.safe_load(open("configs/features.yaml"))
    d = cfg["out_dir"]
    R, Rm = np.load(f"{d}/train.npy", mmap_mode="r"), pd.read_csv(f"{d}/train_meta.csv", keep_default_na=False)
    S, Sm = np.load(f"{d}/synth_train.npy", mmap_mode="r"), pd.read_csv(f"{d}/synth_train_meta.csv", keep_default_na=False)
    mel_hz = librosa.mel_frequencies(cfg["n_mels"] + 2, fmin=cfg["fmin"], fmax=cfg["fmax"])[1:-1]
    rows = ["ae_aegypti", "ae_albopictus", "an_stephensi", "cx_quinquefasciatus", "background"]
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.6 * len(rows)), facecolor=SURF,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.2], hspace=0.6, wspace=0.25))
    extent = [0, cfg["window_s"], 0, len(mel_hz)]
    for i, cls in enumerate(rows):
        ri = np.flatnonzero((Rm.presence == 0).values if cls == "background" else (Rm.species == cls).values)
        si = np.flatnonzero((Sm.species == cls).values) if cls != "background" else np.array([], int)
        ex = R[rng.choice(ri)].astype(np.float32)
        ax = axes[i, 0]
        ax.imshow(ex, origin="lower", aspect="auto", cmap="magma", extent=extent, vmin=ex.max() - 70, vmax=ex.max())
        style(ax, f"{NAMES[cls]}: real window")
        ax.set_yticks([0, 16, 32, 48, 63], [f"{mel_hz[min(k, 63)]:.0f}" for k in (0, 16, 32, 48, 63)])
        ax.set_ylabel("Hz (mel)", fontsize=7, color=MUTED)
        ax = axes[i, 1]
        if len(si):
            ex = S[rng.choice(si)].astype(np.float32)
            ax.imshow(ex, origin="lower", aspect="auto", cmap="magma", extent=extent, vmin=ex.max() - 70, vmax=ex.max())
            style(ax, f"{NAMES[cls]}: synthetic window")
            ax.set_yticks([0, 16, 32, 48, 63], [f"{mel_hz[min(k, 63)]:.0f}" for k in (0, 16, 32, 48, 63)])
        else:
            ax.axis("off")
            ax.text(0.02, 0.5, "no synthetic background\n(household noise is added in M5)", fontsize=8, color=MUTED, transform=ax.transAxes)
        ax = axes[i, 2]
        prof = lambda X, idx: np.median(np.asarray(X[np.sort(rng.choice(idx, min(len(idx), 2000), replace=False))], np.float32).mean(2), 0)
        pr = prof(R, ri)
        ax.plot(mel_hz, pr - pr.max(), color=REAL, lw=2, label=f"real train (n={len(ri):,})")
        if len(si):
            ps = prof(S, si)
            ax.plot(mel_hz, ps - ps.max(), color=SYN, lw=2, label=f"synthetic (n={len(si):,})")
        ax.set_xlabel("Frequency (Hz)", fontsize=7, color=MUTED)
        ax.set_ylabel("dB re. peak", fontsize=7, color=MUTED)
        ax.grid(color=GRID, lw=0.6)
        style(ax, f"{NAMES[cls]}: median band profile")
        ax.legend(frameon=False, fontsize=7)
    for ax in axes[-1, :2]:
        ax.set_xlabel("Time (s)", fontsize=7, color=MUTED)
    fig.suptitle("M2 model input: 1 s log-mel windows (64 bands, 100-4000 Hz), real training data vs synthetic v1.1",
                 x=0.05, ha="left", fontsize=12, color=INK)
    fig.savefig("results/figures/m2_features.png", dpi=110, facecolor=SURF, bbox_inches="tight")
    print("wrote results/figures/m2_features.png")


if __name__ == "__main__":
    main()
