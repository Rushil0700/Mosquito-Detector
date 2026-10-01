"""M5: robustness to noise the model has NEVER heard in training (every noise source here is test-only).

Two tests of a trained M4 model:
  A. accuracy vs SNR: real TEST-split mosquito windows (held-out sessions / labs) + a held-out noise at a controlled SNR,
     measured as energy in 250-3900 Hz (the definition used for the synthetic training data). SNR from -10 to 30 dB plus
     "no added noise". The mosquito window keeps its level; the noise is scaled. Note: the original recordings already
     contain their own background, so "no added noise" is the recording as it is, not a clean mosquito.
     Reports detection recall at the operating thresholds and species / genus accuracy.
  B. false alarms: 1 s noise-only windows (hop 0.5 s) of every source; false alarms per hour at the operating thresholds,
     raw and with 2-of-3 smoothing (a run of consecutive positive windows = one alarm).

Noise sources (none of them used in training):
  DEMAND held-out  DLIVING, OOFFICE, PCAFETER, STRAFFIC, TCAR (training used the 12 other DEMAND environments)
  ESC-50           2,000 clips, 50 classes in 5 groups (CC BY-NC: evaluation only); includes flying insects and crickets
  MUSAN            noise / music / speech, once data/noise/public/musan/musan/ is extracted (else skipped)
  custom           data/noise/custom/<category>/*.wav|flac|mp3 -- your own (e.g. Mumbai) recordings, when present

    python -m src.eval_noise                                   # latest final M4 model
    python -m src.eval_noise --run results/m4_runs/<run> --per-class 300
Writes <run>/m5_noise/: run.log, snr_curve.csv, snr_by_class.csv, false_alarms.csv, false_alarms_by_class.csv, noise_robustness.png
"""
import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import yaml
from scipy import signal

from src.evaluate import INK, MUTED, SURF, decisions, false_alarms_per_hour, style
from src.evaluate_species import NAMES
from src.features import CFG, logmel
from src.models import DEFAULT_SPECIES  # noqa: F401  (registers SpectralWhitening for model loading)

SR = CFG["sample_rate"]
N = int(CFG["window_s"] * SR)
HOP = int(CFG["hop_s"] * SR)
BAND = (250, 3900)
SNRS = [-10, -5, 0, 5, 10, 15, 20, 30]
THRESHOLDS = (0.5, 0.9, 0.98)
DEMAND_HELDOUT = ["DLIVING", "OOFFICE", "PCAFETER", "STRAFFIC", "TCAR"]
ESC_GROUPS = ["animals", "natural", "human", "domestic", "urban"]  # ESC-50 target // 10
GENUS_OF = {"ae_aegypti": "Aedes", "ae_albopictus": "Aedes", "an_stephensi": "Anopheles", "cx_quinquefasciatus": "Culex"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7", "#52514e", "#0f8b8d"]
AUDIO = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
log = logging.getLogger("m5")
_SOS = signal.butter(4, [BAND[0] / (SR / 2), BAND[1] / (SR / 2)], "bandpass", output="sos")


def setup_logging(out):
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


def band_energy(x):
    return float(np.mean(signal.sosfilt(_SOS, x) ** 2)) + 1e-20


def load(path, offset=0.0, duration=None):
    warnings.filterwarnings("ignore")
    x, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=duration)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- noise
def noise_sources(max_musan_files):
    """[(group, class, source_id, audio)] for every available test-only noise recording."""
    out = []
    root = Path("data/noise/public")
    for env in DEMAND_HELDOUT:
        p = root / "demand" / env / "ch01.wav"
        if p.exists():
            out.append(("DEMAND held-out", env, str(p), load(p)))
    meta = root / "esc50" / "ESC-50-master" / "meta" / "esc50.csv"
    if meta.exists():
        esc = pd.read_csv(meta)
        for r in esc.itertuples():
            out.append((f"ESC-50 {ESC_GROUPS[r.target // 10]}", r.category, r.filename,
                        load(meta.parent.parent / "audio" / r.filename)))
    else:
        log.info("  ESC-50 not found (data/noise/public/esc50/ESC-50-master) -- skipped")
    musan = root / "musan" / "musan"
    if musan.exists():
        rng = np.random.default_rng(0)
        for kind in ("noise", "music", "speech"):
            files = sorted((musan / kind).rglob("*.wav"))
            for p in rng.permutation(files)[:max_musan_files]:
                out.append((f"MUSAN {kind}", kind, str(p), load(p, duration=60)))
    else:
        log.info("  MUSAN not extracted (data/noise/public/musan/musan) -- skipped")
    for p in sorted(Path("data/noise/custom").rglob("*")):
        if p.suffix.lower() in AUDIO:
            cat = p.parent.name if p.parent.name != "custom" else "custom"
            out.append(("custom (your recordings)", cat, str(p), load(p)))
    return [s for s in out if len(s[3]) >= N // 2]


def excerpt(audio, rng, tries=20):
    """A random 1 s excerpt that is not (near) digital silence (ESC-50 clips are zero-padded)."""
    for _ in range(tries):
        a = int(rng.integers(0, max(1, len(audio) - N + 1)))
        x = audio[a:a + N]
        x = np.resize(x, N) if len(x) < N else x
        if band_energy(x) > 1e-9:
            return x
    return x


# ------------------------------------------------------------------------------------------------------------ mosquitoes
def mosquito_windows(per_class, species, seed=0):
    """Up to `per_class` real TEST-split mosquito windows per species class (<= 3 per segment, spread over recordings)."""
    seg = pd.read_csv("data/processed/segments_split.csv", keep_default_na=False)
    seg = seg[(seg.split == "test") & (seg.presence == 1) & (seg.dur >= CFG["min_segment_s"])]
    seg = seg.assign(cls=np.where(seg.species.isin(species), seg.species, "other_mosquito"))
    rng = np.random.default_rng(seed)
    X, rows = [], []
    for cls, g in seg.groupby("cls"):
        got = 0
        for r in g.sample(frac=1, random_state=seed).itertuples():
            if got >= per_class:
                break
            x = load(r.path, offset=r.start_s, duration=r.dur)
            if len(x) < N:
                x = np.resize(x, N)
            starts = rng.permutation(np.arange(0, len(x) - N + 1, HOP))[:min(3, per_class - got)]
            for s in starts:
                X.append(x[s:s + N])
                rows.append(dict(cls=cls, species=r.species, site=r.site, src=r.path))
            got += len(starts)
    return np.stack(X), pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------- model
class Model:
    def __init__(self, run):
        cfg = yaml.safe_load(open(Path(run) / "config.yaml"))
        self.species = cfg.get("species_classes") or DEFAULT_SPECIES
        self.m = tf.keras.models.load_model(Path(run) / "model.keras")

    def __call__(self, wav):
        F = np.concatenate([logmel(wav[i:i + 2048]) for i in range(0, len(wav), 2048)]).astype(np.float32)
        P = self.m.predict(F, batch_size=1024, verbose=0)
        return P["presence"][:, 0], P["species"]


def species_scores(M, S, species):
    pred = np.array(species)[S.argmax(1)]
    tgt = M.cls.isin(GENUS_OF).values
    per = {c: float((pred[(M.cls == c).values] == c).mean()) for c in M.cls.unique()}
    return dict(species_acc_targets=float((pred[tgt] == M.cls.values[tgt]).mean()),
                genus_acc_targets=float(np.mean([GENUS_OF.get(p, "-") == GENUS_OF[t] for p, t in zip(pred[tgt], M.cls.values[tgt])])),
                balanced_species_acc=float(np.mean(list(per.values()))), **{f"recall_{c}": v for c, v in per.items()})


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/m4_runs/20260930_114325_real_plus_synth")
    ap.add_argument("--per-class", type=int, default=300, help="real test mosquito windows per species class")
    ap.add_argument("--max-musan-files", type=int, default=150, help="per MUSAN kind (first 60 s of each)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.run) / "m5_noise"
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    t0 = time.time()
    model = Model(args.run)
    log.info(f"M5 noise robustness | model {args.run} | species classes {model.species}")

    log.info("\nloading test-only noise sources ...")
    noise = noise_sources(args.max_musan_files)
    nz = pd.DataFrame([(g, c, s, len(a) / SR) for g, c, s, a in noise], columns=["group", "cls", "src", "sec"])
    log.info("\n" + nz.groupby("group").agg(files=("src", "size"), classes=("cls", "nunique"),
                                            minutes=("sec", lambda s: round(s.sum() / 60, 1))).to_string())

    log.info(f"\nloading real TEST mosquito windows (<= {args.per_class} per class) ...")
    Xm, Mm = mosquito_windows(args.per_class, model.species, args.seed)
    log.info("\n" + Mm.groupby("cls").agg(windows=("src", "size"), recordings=("src", "nunique"), sites=("site", "nunique")).to_string())

    # ---- A. accuracy vs SNR
    log.info("\nA. accuracy vs SNR (mosquito window + held-out noise)")
    rows, by_class = [], []
    p0, S0 = model(Xm)
    base = dict(snr_db="none", **{f"recall@{t}": float((p0 >= t).mean()) for t in THRESHOLDS}, **species_scores(Mm, S0, model.species))
    for grp in ["(no added noise)"]:
        rows.append(dict(group=grp, **base))
    groups = nz.groupby("group").indices
    for gi, (grp, idx) in enumerate(groups.items()):
        rng = np.random.default_rng(args.seed + gi)
        # one noise excerpt per mosquito window, from a random file of this group; the same excerpt at every SNR
        pick = rng.choice(idx, len(Xm))
        E = np.stack([excerpt(noise[k][3], rng) for k in pick])
        cls_of = nz.cls.values[pick]
        ex, en = np.array([band_energy(x) for x in Xm]), np.array([band_energy(e) for e in E])
        for snr in SNRS:
            g = np.sqrt(ex / (en * 10 ** (snr / 10)))
            p, S = model(Xm + g[:, None] * E)
            r = dict(group=grp, snr_db=snr, **{f"recall@{t}": float((p >= t).mean()) for t in THRESHOLDS},
                     **species_scores(Mm, S, model.species))
            rows.append(r)
            for c in np.unique(cls_of):
                k = cls_of == c
                by_class.append(dict(group=grp, noise_class=c, snr_db=snr, windows=int(k.sum()), **{f"recall@{t}": float((p[k] >= t).mean()) for t in THRESHOLDS}))
        log.info(f"  {grp:28s} done ({time.time() - t0:.0f} s)")
    snr = pd.DataFrame(rows)
    snr.to_csv(out / "snr_curve.csv", index=False)
    pd.DataFrame(by_class).to_csv(out / "snr_by_class.csv", index=False)
    pd.set_option("display.width", 220)
    for col, title in (("recall@0.9", "mosquito detected (threshold 0.9)"), ("species_acc_targets", "species correct, 4 target species"),
                       ("genus_acc_targets", "genus correct, 4 target species")):
        tab = snr[snr.snr_db != "none"].pivot(index="group", columns="snr_db", values=col)
        tab["no noise"] = base[col]
        log.info(f"\n  {title} -- by SNR (dB)")
        log.info("\n" + tab.map(lambda v: f"{v:.0%}").to_string())

    # ---- B. false alarms on noise alone
    log.info("\nB. false alarms on noise alone (1 s windows, hop 0.5 s)")
    fa_rows, meta = [], []
    wins = []
    for k, (grp, cls, src, a) in enumerate(noise):
        a = np.resize(a, N) if len(a) < N else a
        for s in range(0, len(a) - N + 1, HOP):
            wins.append(a[s:s + N])
            meta.append(dict(group=grp, cls=cls, src=src, win_start_s=s / SR, presence=0))
    Mn = pd.DataFrame(meta)
    pn, Sn = model(np.stack(wins))
    Mn["p"] = pn
    Mn["species_guess"] = np.array(model.species)[Sn.argmax(1)]
    for level, keys in (("group", ["group"]), ("class", ["group", "cls"])):
        for key, g in Mn.groupby(keys):
            G = g.reset_index(drop=True)
            r = dict(zip(keys, key if isinstance(key, tuple) else (key,)), hours=round(len(G) * CFG["hop_s"] / 3600, 3))
            for t in THRESHOLDS:
                r[f"FA/h@{t}"] = false_alarms_per_hour(decisions(G.p.values, G, t), G)[0]
                r[f"FA/h@{t} 2of3"] = false_alarms_per_hour(decisions(G.p.values, G, t, 2, 3), G)[0]
            if level == "class":
                hit = G.p >= 0.9
                r["most common species guess when firing"] = (G.species_guess[hit].map(NAMES).value_counts().index[0] if hit.any() else "")
            fa_rows.append((level, r))
    fa = pd.DataFrame([r for lv, r in fa_rows if lv == "group"])
    fac = pd.DataFrame([r for lv, r in fa_rows if lv == "class"]).sort_values("FA/h@0.9 2of3", ascending=False)
    fa.to_csv(out / "false_alarms.csv", index=False)
    fac.to_csv(out / "false_alarms_by_class.csv", index=False)
    fmt = lambda d: d.map(lambda v: f"{v:.1f}" if isinstance(v, float) else v)
    log.info("\n" + fmt(fa).to_string(index=False))
    log.info("\n  worst 15 noise classes (false alarms / hour, threshold 0.9, 2-of-3):")
    log.info("\n" + fmt(fac.head(15)).to_string(index=False))

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
    xs = SNRS
    for ax, col, title in zip(axes[:2], ("recall@0.9", "species_acc_targets"),
                              ("Mosquito detected (threshold 0.9)", "Species correct (4 target species)")):
        for i, grp in enumerate(groups):
            s = snr[snr.group == grp]
            ax.plot(xs, s[col].values, color=PALETTE[i % len(PALETTE)], lw=2, marker="o", ms=4, label=grp)
        ax.axhline(base[col], color=MUTED, lw=1, ls="--", label="no added noise")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("SNR in 250-3900 Hz (dB): mosquito vs added noise", fontsize=8, color=MUTED)
        style(ax, title)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    ax = axes[2]
    f = fa.set_index("group")
    y = np.arange(len(f))
    ax.barh(y, f["FA/h@0.9 2of3"].values, color=[PALETTE[list(groups).index(g) % len(PALETTE)] for g in f.index], height=0.6)
    for yi, v in zip(y, f["FA/h@0.9 2of3"].values):
        ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, f.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("False alarms per hour of noise alone (threshold 0.9, 2-of-3)", fontsize=8, color=MUTED)
    style(ax, "False alarms on noise alone")
    fig.suptitle("M5: robustness to noise never heard in training", x=0.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "noise_robustness.png", dpi=120, facecolor=SURF)
    log.info(f"\nfigure {out / 'noise_robustness.png'} | tables in {out} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
