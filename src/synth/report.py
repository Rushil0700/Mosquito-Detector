"""Synthetic vs real summary figure on the held-out TEST split (results/figures/synth_vs_real.png).

Rows 1-2 per class: median spectrum (band = middle 50 %), f0 histogram, one real and one synthetic
spectrogram at similar f0. Row 3: fair AUC per version vs the real-vs-real floor (mean over seeds),
tone SNR and pitch steadiness distributions. Also writes results/synth_report.csv.

Usage: python -m src.synth.report --real data/processed/manifest_funchal.csv [--seeds 7 11 13]
"""
import argparse
import json
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.synth.calibrate import log_spectrum
from src.synth.generate import NoiseBank
from src.synth.validate import compare, real_floor

REAL, SYN, SYN_OLD = "#2a78d6", "#eb6834", "#f4b8a0"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
NAMES = {"ae_aegypti": "Ae. aegypti", "cx_quinquefasciatus": "Cx. quinquefasciatus"}
# Earlier versions, test split, as reported at the time (experiments.md M1d): v0.1 external measurement,
# v0.2 mean of 3 seeds. (Re-run with the current code, v0.2 scores 0.83 / 0.78 on test: M1g.)
HISTORY = {"ae_aegypti": {"v0.1": 0.93, "v0.2": 0.80}, "cx_quinquefasciatus": {"v0.1": 0.99, "v0.2": 0.82}}


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def spectra(wavs, sr):
    f = None
    out = []
    for x in wavs:
        f, s = log_spectrum((x - x.mean()) / (x.std() + 1e-12), sr, 1024)
        out.append(s)
    return f, np.array(out)


def spectrogram(ax, x, sr, title):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=256, hop_length=32)), ref=np.max)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-60, vmax=0,
              extent=[0, len(x) / sr, 0, sr / 2])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel("Hz", fontsize=8, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--config", default="configs/synth.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--version", default="v0.3")
    ap.add_argument("--out", default="results/figures/synth_vs_real.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sr = cfg["sample_rate"]
    bank = json.load(open(cfg["calibration_file"]))
    man = pd.read_csv(args.real)
    real = man[man.split == args.split]
    nb = NoiseBank(bank, sr, cfg["noise"]["pool_by"])

    rows, tables = [], None
    for seed in args.seeds:
        res, tab = compare(real, cfg, bank, nb, args.n, np.random.default_rng(seed), seed=seed)
        tables = tables or tab
        for r in res:
            rows.append(dict(cls=r["cls"], seed=seed, auc_fair=r["auc_fair"], auc_all=r["auc_all"],
                             floor=real_floor(man, args.split, r["cls"], sr, args.n, seed)))
    df = pd.DataFrame(rows)
    summ = df.groupby("cls").agg(auc_fair=("auc_fair", "mean"), auc_fair_sd=("auc_fair", "std"),
                                 auc_all=("auc_all", "mean"), floor=("floor", "mean"), floor_sd=("floor", "std"))
    summ["gap"] = summ.auc_fair - summ.floor
    print(summ.round(3).to_string())
    summ.round(4).to_csv("results/synth_report.csv")

    classes = list(tables)
    fig = plt.figure(figsize=(16, 13.5), facecolor=SURF)
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)
    for row, cls in enumerate(classes):
        Fr, Fs, rw, sw = tables[cls]
        ax = fig.add_subplot(gs[row, 0])
        for wavs, c, lab in [(rw, REAL, "real (median, band = middle 50%)"), (sw, SYN, "synthetic (median, band = middle 50%)")]:
            f, S = spectra(wavs, sr)
            ax.fill_between(f, *np.percentile(S, [25, 75], 0), color=c, alpha=0.12, lw=0)
            ax.plot(f, np.median(S, 0), color=c, lw=2, label=lab)
        ax.set_xlim(0, sr / 2)
        ax.set_xlabel("Frequency (Hz)", fontsize=8, color=MUTED)
        ax.set_ylabel("Level, RMS-normalised (dB)", fontsize=8, color=MUTED)
        style(ax, f"{NAMES.get(cls, cls)}: spectrum")
        ax.legend(frameon=False, fontsize=8)

        ax = fig.add_subplot(gs[row, 1])
        bins = np.arange(300, 810, 10)
        for F, c, lab in [(Fr, REAL, "real"), (Fs, SYN, "synthetic")]:
            v = F["f0"].dropna()
            ax.hist(v, bins=bins, histtype="step", color=c, lw=2, label=f"{lab}  median {v.median():.0f} Hz")
            ax.hist(v, bins=bins, color=c, alpha=0.1)
        ax.set_xlabel("Fundamental f0 (Hz)", fontsize=8, color=MUTED)
        ax.set_ylabel("Clips", fontsize=8, color=MUTED)
        style(ax, f"{NAMES.get(cls, cls)}: wingbeat pitch (f0)")
        ax.legend(frameon=False, fontsize=8)

        sub = gs[row, 2].subgridspec(2, 1, hspace=0.6)
        fr, fs = Fr["f0"].values, Fs["f0"].values
        i = int(np.nanargmin(np.abs(fr - np.nanmedian(fr))))
        j = int(np.nanargmin(np.abs(fs - fr[i])))
        spectrogram(fig.add_subplot(sub[0]), rw[i], sr, f"{NAMES.get(cls, cls)}: real example, f0 {fr[i]:.0f} Hz")
        a = fig.add_subplot(sub[1])
        spectrogram(a, sw[j], sr, f"synthetic example, f0 {fs[j]:.0f} Hz")
        a.set_xlabel("Time (s)", fontsize=8, color=MUTED)

    # AUC bars per version vs floor
    ax = fig.add_subplot(gs[2, 0])
    y, labels, ticks = 0, [], []
    for cls in classes:
        bars = [(k, v, SYN_OLD) for k, v in HISTORY.get(cls, {}).items()]
        bars.append((args.version, summ.loc[cls, "auc_fair"], SYN))
        bars.append(("real vs real (floor)", summ.loc[cls, "floor"], REAL))
        ax.text(0.505, y - 0.7, NAMES.get(cls, cls), fontsize=9, style="italic", color=INK)
        for name, v, c in bars:
            ax.barh(y, v - 0.5, left=0.5, color=c, height=0.62)
            ax.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
            ticks.append(y)
            labels.append(name if "floor" in name else f"{name} synthetic vs real")
            y += 1
        y += 1.2
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 1.6, -1.5)  # inverted; headroom for the first species label
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("ROC-AUC   (0.5 = indistinguishable, 1.0 = always told apart)", fontsize=8, color=MUTED)
    style(ax, "How easily a classifier tells them apart (fair features)")
    ax.grid(axis="y", visible=False)

    for col, (feat, title, ylab) in enumerate([("tone_snr", "Tone SNR (loudest harmonic above noise floor)", "dB"),
                                               ("sharp_8hz", "Pitch steadiness (energy within ±8 Hz of f0 peak)", "fraction")], start=1):
        ax = fig.add_subplot(gs[2, col])
        data, pos, cols = [], [], []
        for k, cls in enumerate(classes):
            Fr, Fs, _, _ = tables[cls]
            for d, (F, c) in enumerate([(Fr, REAL), (Fs, SYN)]):
                data.append(F[feat].dropna().values)
                pos.append(k * 3 + d)
                cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                        medianprops=dict(color=SURF, lw=2), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, edgecolor=c)
        ax.set_xticks([k * 3 + 0.5 for k in range(len(classes))], [NAMES.get(c, c) for c in classes])
        ax.set_ylabel(ylab, fontsize=8, color=MUTED)
        style(ax, title)

    fig.suptitle(f"Synthetic vs real mosquito flight tone (synthesizer {args.version}; real = held-out Funchal {args.split} clips)",
                 x=0.07, y=0.965, ha="left", fontsize=14, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REAL), plt.Rectangle((0, 0), 1, 1, color=SYN)]
    fig.legend(handles, [f"real (held-out {args.split} clips)", "synthetic"], loc="upper right", bbox_to_anchor=(0.93, 0.975),
               ncol=2, frameon=False, fontsize=10)
    n = {c: len(tables[c][0]) for c in classes}
    fig.text(0.07, 0.035, f"Fair AUC and floor = mean of seeds {args.seeds} (n = " + " / ".join(str(v) for v in n.values())
             + " clips per side); floor = real calib split vs real test split. "
             + "; ".join(f"{NAMES.get(c, c)} gap to floor {summ.loc[c, 'gap']:+.2f}" for c in classes) + ".",
             fontsize=9, color=MUTED)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, facecolor=SURF)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
