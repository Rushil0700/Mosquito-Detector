"""Realism check: synthetic clips vs HELD-OUT real clips (split != calib).

Reports, per class:
  auc_all   discriminator ROC-AUC on all features (incl. level / quantisation / DC / <150 Hz)
  auc_fair  discriminator ROC-AUC on 'fair' features: RMS-normalised log-mel 150-4000 Hz + tone-shape stats
  top per-feature univariate AUCs (folded to >= 0.5), so every run shows what still gives clips away.
0.5 = indistinguishable. Synthetic clips are generated in memory from the calibration bank.
Also reports the real-vs-real floor (real calib split vs the same held-out split, same features and n):
the synthesizer's target is that floor, not 0.5. Metrics are averaged over --seeds.

Usage: python -m src.synth.validate --real data/processed/manifest_funchal.csv [--split test] [--seeds 7 11 13]
"""
import argparse
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.synth.calibrate import analyse_clip, log_spectrum, peak_sharpness
from src.synth.generate import DriftMap, Equalizer, NoiseBank, class_cfg, synthesize

REAL, SYN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3dd"
N_MELS, FMIN, FMAX = 32, 150, 4000
UNFAIR = {"peak_dbfs", "n_distinct", "dc", "low150_db"}


def clip_features(x, sr):
    """Named features for one clip. Level/device features are listed in UNFAIR."""
    feats = {}
    xn = (x - x.mean()) / (x.std() + 1e-12)
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=xn, sr=sr, n_fft=512, hop_length=128,
                                                           n_mels=N_MELS, fmin=FMIN, fmax=FMAX))
    centers = librosa.mel_frequencies(N_MELS + 2, fmin=FMIN, fmax=FMAX)[1:-1]
    for c, mu, sd in zip(centers, m.mean(1), m.std(1)):
        feats[f"mel_mean_{c:.0f}Hz"] = mu
        feats[f"mel_std_{c:.0f}Hz"] = sd
    rms = librosa.feature.rms(y=x, frame_length=256, hop_length=128)[0]
    feats["env_cv"] = rms.std() / (rms.mean() + 1e-12)
    feats["crest"] = np.max(np.abs(x - x.mean())) / (x.std() + 1e-12)
    feats["flatness"] = float(np.mean(librosa.feature.spectral_flatness(y=xn, n_fft=512, hop_length=128)))
    a = analyse_clip(x, sr)
    f, s = log_spectrum(x, sr, 2048)
    if a:
        f0 = a["f0"]
        feats["sharp_8hz"] = peak_sharpness(x, sr, f0)
        feats["tone_snr"] = a["snr_db"]
        h = a["harmonic_db"] + [np.nan] * 3
        feats["h2_h1"], feats["h3_h1"] = h[1] - h[0], h[2] - h[0]
        feats["f0"] = f0
    feats["peak_dbfs"] = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
    feats["n_distinct"] = np.log10(len(np.unique(x)))
    feats["dc"] = x.mean() / (x.std() + 1e-12)
    p = 10 ** (s / 10)
    feats["low150_db"] = 10 * np.log10(p[f < 150].sum() / (p.sum() + 1e-20) + 1e-20)
    return feats


def discriminator_auc(F, y):
    F = F.fillna(F.median())
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    p = cross_val_predict(clf, F.values, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def univariate_aucs(F, y):
    out = {}
    for c in F.columns:
        v = F[c].values
        ok = np.isfinite(v)
        if ok.sum() > 10 and len(np.unique(y[ok])) == 2:
            a = roc_auc_score(y[ok], v[ok])
            out[c] = max(a, 1 - a)
    return pd.Series(out).sort_values(ascending=False)


_REAL_CACHE = {}


def real_set(real_df, cls, sr, n, seed):
    """(features, waveforms) of up to n real clips of `cls` (cached: tuning re-scores the same clips)."""
    r = real_df[real_df.cls == cls]
    r = r.sample(min(n, len(r)), random_state=seed)
    key = (sr, tuple(r.path))
    if key not in _REAL_CACHE:
        wavs = [x for x in (librosa.load(p, sr=sr)[0] for p in r.path) if len(x) >= 0.2 * sr]  # skip empty/truncated
        _REAL_CACHE[key] = (pd.DataFrame([clip_features(x, sr) for x in wavs]), wavs)
    return _REAL_CACHE[key]


def score_pair(Fa, Fb):
    """(auc_all, auc_fair, univariate AUCs) for telling table A (label 0) from table B (label 1)."""
    F = pd.concat([Fa, Fb], ignore_index=True)
    y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]
    fair = [c for c in F.columns if c not in UNFAIR]
    return discriminator_auc(F, y), discriminator_auc(F[fair], y), univariate_aucs(F, y)


def synth_set(cfg, bank, noise_bank, cls, n, rng, eq=None):
    ccfg = class_cfg(cfg, cls)
    entries = [e for es in bank[cls].values() for e in es]
    dm = DriftMap(ccfg)
    syn = [synthesize(entries[rng.integers(len(entries))], ccfg, rng, noise_bank, drift_map=dm, cls=cls, eq=eq)[0]
           for _ in range(n)]
    return pd.DataFrame([clip_features(x, cfg["sample_rate"]) for x in syn]), syn


def compare(real_df, cfg, bank, noise_bank, n, rng, seed=0, classes=None, eq=None):
    """Return per-class result dicts and per-class (real, synth) feature tables."""
    sr = cfg["sample_rate"]
    if eq is None:
        eq = Equalizer(cfg.get("eq", {}).get("file"))
    results, tables = [], {}
    for cls in classes or sorted(set(real_df.cls) & set(bank)):
        Fr, real = real_set(real_df, cls, sr, n, seed)
        Fs, syn = synth_set(cfg, bank, noise_bank, cls, len(real), rng, eq)
        auc_all, auc_fair, uni = score_pair(Fr, Fs)
        results.append(dict(cls=cls, n=len(real), auc_all=auc_all, auc_fair=auc_fair, top=uni.head(6).round(2).to_dict()))
        tables[cls] = (Fr, Fs, real, syn)
    return results, tables


def real_floor(man, split, cls, sr, n, seed):
    """Fair AUC of real calib clips vs real `split` clips: the best a synthesizer fitted on calib can do."""
    Fa, _ = real_set(man[man.split == "calib"], cls, sr, n, seed)
    Fb, _ = real_set(man[man.split == split], cls, sr, n, seed)
    k = min(len(Fa), len(Fb))
    return score_pair(Fa.iloc[:k], Fb.iloc[:k])[1]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot(tables, sr, out):
    fig, axes = plt.subplots(len(tables), 2, figsize=(10, 3.2 * len(tables)), facecolor="#fcfcfb")
    axes = np.atleast_2d(axes)
    for row, (cls, (Fr, Fs, real, syn)) in enumerate(tables.items()):
        ax = axes[row, 0]
        for wavs, c, lab in [(real, REAL, "real (held-out)"), (syn, SYN, "synthetic")]:
            specs = []
            for x in wavs:
                f, sp = log_spectrum((x - x.mean()) / (x.std() + 1e-12), sr, 1024)
                specs.append(sp)
            ax.plot(f, np.median(specs, 0), color=c, lw=2, label=lab)
        ax.set_xlim(0, 4000)
        ax.set_xlabel("Frequency (Hz)", fontsize=8, color=MUTED)
        ax.set_ylabel("Median level, RMS-normalised (dB)", fontsize=8, color=MUTED)
        style(ax, f"{cls}: median spectrum")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[row, 1]
        bins = np.arange(300, 800, 10)
        for F, c, lab in [(Fr, REAL, "real (held-out)"), (Fs, SYN, "synthetic")]:
            v = F["f0"].dropna()
            ax.hist(v, bins=bins, color=c, alpha=0.55, label=f"{lab}  median {v.median():.0f} Hz", edgecolor="#fcfcfb", linewidth=1)
        ax.set_xlabel("Fundamental f0 (Hz)", fontsize=8, color=MUTED)
        ax.set_ylabel("Clips", fontsize=8, color=MUTED)
        style(ax, f"{cls}: f0 distribution")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default="configs/synth.yaml")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--out", default="results/figures/synth_validation.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    bank = json.load(open(cfg["calibration_file"]))
    man = pd.read_csv(args.real)
    real = man[man.split == args.split]
    nb = NoiseBank(bank, cfg["sample_rate"], cfg["noise"]["pool_by"])
    rows = []
    for seed in args.seeds:
        results, tables = compare(real, cfg, bank, nb, args.n, np.random.default_rng(seed), seed=seed)
        for r in results:
            r.update(seed=seed, floor_fair=real_floor(man, args.split, r["cls"], cfg["sample_rate"], args.n, seed))
            print(f"seed {seed} {r['cls']:22s} n={r['n']}  auc_all={r['auc_all']:.3f}  auc_fair={r['auc_fair']:.3f}"
                  f"  floor={r['floor_fair']:.3f}", flush=True)
            print("   top features:", ", ".join(f"{k} {v}" for k, v in r["top"].items()))
            rows.append(r)
    df = pd.DataFrame(rows)
    summ = df.groupby("cls")[["auc_all", "auc_fair", "floor_fair"]].agg(["mean", "std"]).round(3)
    summ[("gap", "mean")] = (summ[("auc_fair", "mean")] - summ[("floor_fair", "mean")]).round(3)
    print(f"\n{args.split} split, mean over seeds {args.seeds}:\n{summ.to_string()}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot(tables, cfg["sample_rate"], args.out)
    df.to_csv(Path(args.out).parent.parent / "synth_validation.csv", index=False)


if __name__ == "__main__":
    main()
