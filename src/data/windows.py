"""Cut real mosquito recordings of the target species into 1 s windows for synthesizer calibration.

Sources: HumBugDB mosquito clips (already labelled segments) and Abuzz mosquito recordings (whole files).
Funchal clips (0.3 s) are too short for 1 s windows; UFRGS is kept aside as an external test.
Windows are non-overlapping, at most 40 per recording, raised (in steps of 10) until the species has ~1,500 windows or runs out of audio (random, seeded), saved at 8 kHz:
16-bit for native 8 kHz sources (keeps the real quantisation grid), float otherwise.

Each window gets a quick f0 / tone-SNR estimate (harmonic sum, 200-1100 Hz) used later for sex assignment.

Realism split (calib / tune / test) — for measuring the synthesizer only, NOT the model's train/test split:
  * species with >= MIN_SESSIONS sessions: whole sessions are assigned (~60/20/20 of windows);
  * otherwise (one dominant session): contiguous time blocks inside each recording (first 60 % calib,
    next 20 % tune, last 20 % test). Same session on both sides, so that floor is optimistic; flagged
    in the `split_mode` column.

Usage: python -m src.data.windows
       python -m src.data.windows --species an_arabiensis an_funestus ... --train-only --per-class 600 \
              --out data/processed/manifest_windows_other.csv   # other species, TRAINING sessions only (split = calib)
"""
import argparse
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.synth.calibrate import harmonic_levels, harmonic_sum_f0, log_spectrum

TARGET = ["ae_aegypti", "ae_albopictus", "an_stephensi", "cx_quinquefasciatus"]
SR, WIN, MAX_PER_FILE, TARGET_PER_CLASS, MIN_SESSIONS = 8000, 1.0, 40, 1500, 5
OUT_DIR = Path("data/processed/windows")
OUT = Path("data/processed/manifest_windows.csv")


def quick_f0(x):
    f, s = log_spectrum(x, SR, 2048)
    f0, floor = harmonic_sum_f0(f, s, 200, 1100)
    return f0, harmonic_levels(f, s, f0, floor)[1]


def session_split(df, rng):
    """Assign whole sessions to calib/tune/test so that each gets ~60/20/20 % of the windows."""
    sizes = df.groupby("session").size().sample(frac=1, random_state=int(rng.integers(1e9)))
    cum = sizes.cumsum() / sizes.sum()
    return {s: ("calib" if c <= 0.6 or i == 0 else "tune" if c <= 0.8 else "test")
            for i, (s, c) in enumerate(cum.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*", default=TARGET)
    ap.add_argument("--train-only", action="store_true", help="only sessions in the model's TRAINING split; all rows split=calib")
    ap.add_argument("--per-class", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    m = pd.read_csv("data/processed/manifest_files.csv")
    m = m[(m.cls.isin(args.species)) & (m.sound_type == "mosquito") & (m.dataset.isin(["humbugdb", "abuzz"]))]
    if args.train_only:
        seg = pd.read_csv("data/processed/segments_split.csv")
        train_sessions = set(seg[seg.split == "train"].session)
        m = m[m.session.isin(train_sessions)]
    # per-recording cap: 40, raised for species with few recordings so rare species keep all their audio
    avail = (m.duration_s // WIN).astype(int)
    cap = {}
    for c, a in avail.groupby(m.cls):
        c_cap = MAX_PER_FILE
        while np.minimum(a, c_cap).sum() < args.per_class and c_cap < a.max():
            c_cap += 10
        cap[c] = c_cap
    rows = []
    for r in m.itertuples():
        x, _ = librosa.load(r.path, sr=SR)
        n = int(WIN * SR)
        k_all = np.arange(len(x) // n)
        if not len(k_all):
            continue
        ks = np.sort(rng.permutation(k_all)[:cap[r.cls]])
        d = OUT_DIR / r.cls
        d.mkdir(parents=True, exist_ok=True)
        for k in ks:
            w = x[k * n:(k + 1) * n]
            if np.std(w) == 0:
                continue
            f0, snr = quick_f0(w)
            # short hash of the full source path: Abuzz reuses file names ("As.wav") across phone folders
            tag = hashlib.md5(r.path.encode()).hexdigest()[:8]
            p = d / f"{r.dataset}_{tag}_{Path(r.path).stem.replace(' ', '_')}_{k:04d}.wav"
            sf.write(p, w, SR, subtype="PCM_16" if int(r.orig_sr) == 8000 else "FLOAT")
            rows.append(dict(path=str(p), cls=r.cls, sex_label=r.sex, dataset=r.dataset, orig_sr=int(r.orig_sr),
                             site=r.site, device=r.device, session=r.session, src_file=r.path, start_s=k * WIN,
                             pos_in_file=(k + 0.5) / len(k_all), f0_quick=round(float(f0), 1),
                             snr_quick=round(float(snr), 1) if np.isfinite(snr) else np.nan))
    w = pd.DataFrame(rows)
    assert w.path.is_unique, "window paths must be unique"

    if args.train_only:
        w["split"], w["split_mode"] = "calib", "train_sessions_only"
    else:
        split, mode = [], []
        for cls, g in w.groupby("cls"):
            if g.session.nunique() >= MIN_SESSIONS:
                s_ = session_split(g, rng)
                split += [(i, s_[v]) for i, v in g.session.items()]
                mode += [(i, "session") for i in g.index]
            else:
                split += [(i, "calib" if p < 0.6 else "tune" if p < 0.8 else "test") for i, p in g.pos_in_file.items()]
                mode += [(i, "time_block") for i in g.index]
        w["split"] = pd.Series(dict(split))
        w["split_mode"] = pd.Series(dict(mode))
    w.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(w.groupby(["cls", "split_mode", "split"]).agg(windows=("path", "size"), sessions=("session", "nunique")).unstack("split"))
    print("\nwindows by dataset / sex label:")
    print(w.pivot_table(index=["cls", "dataset"], columns="sex_label", values="path", aggfunc="size", fill_value=0))
    print(f"\nwrote {args.out} ({len(w)} windows)")


if __name__ == "__main__":
    main()
