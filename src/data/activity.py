"""Find flight (wingbeat) segments in long, weakly-labelled recordings (Abuzz).

A hand-made harmonicity score (tone SNR + harmonic count) was tried first and rejected: it fired on
99 % of HumBugDB background frames (hum / fans / rumble are harmonic too). Instead, a small frame
classifier (gradient boosting on log-mel statistics, 1 s frames, hop 0.5 s) is trained on HumBugDB's
labelled mosquito vs background/other clips. Its accuracy is estimated with GroupKFold by SESSION,
the threshold is set to a target false-positive rate on held-out sessions, and it is checked on Abuzz's
own background recordings. Active frames are merged into segments (gap <= 0.5 s, min 1 s).
Caveat: segments are the ones a HumBugDB-trained detector accepts, so they may under-represent
unusual-sounding flight in Abuzz.

Usage: python -m src.data.activity [--fpr 0.02]
Writes data/processed/abuzz_segments.csv and results/activity_threshold.json.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


SR, WIN, HOP = 8000, 1.0, 0.5


def frame_features(x, sr=SR):
    """Per 1 s frame: mean and std of log-mel (40 bands, 100-4000 Hz, RMS-normalised) + flatness + level."""
    w, h = int(WIN * sr), int(HOP * sr)
    starts, feats = [], []
    for i in range(0, max(len(x) - w, 0) + 1, h):
        seg = x[i:i + w]
        if len(seg) < w // 2 or np.std(seg) == 0:
            continue
        seg = np.pad(seg, (0, w - len(seg))) if len(seg) < w else seg
        xn = (seg - seg.mean()) / (seg.std() + 1e-12)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=xn, sr=sr, n_fft=512, hop_length=256, n_mels=40, fmin=100, fmax=4000))
        fl = librosa.feature.spectral_flatness(y=xn, n_fft=512, hop_length=256)
        feats.append(np.r_[m.mean(1), m.std(1), fl.mean(), 20 * np.log10(seg.std() + 1e-9)])
        starts.append(i / sr)
    return np.array(starts), np.array(feats)


def humbugdb_frames(max_frames=8, seed=0):
    m = pd.read_csv("data/processed/manifest_files.csv")
    h = m[m.dataset == "humbugdb"]
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for r in h.itertuples():
        x, _ = librosa.load(r.path, sr=SR, duration=60.0)
        _, F = frame_features(x)
        if not len(F):
            continue
        F = F[rng.permutation(len(F))[:max_frames]]
        X.append(F)
        y += [int(r.sound_type == "mosquito")] * len(F)
        groups += [r.session] * len(F)
    return np.vstack(X), np.array(y), np.array(groups)


def train_detector(fpr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    X, y, g = humbugdb_frames()
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    thr = float(np.quantile(oof[y == 0], 1 - fpr))          # FP rate on held-out sessions
    rec = float(np.mean(oof[y == 1] >= thr))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
    return clf, thr, rec, int((y == 0).sum()), int((y == 1).sum())


def segments(starts, active):
    segs, cur = [], None
    for t in starts[active]:
        if cur and t <= cur[1] + HOP:  # contiguous / small gap
            cur[1] = t + WIN
        else:
            if cur:
                segs.append(cur)
            cur = [t, t + WIN]
    if cur:
        segs.append(cur)
    return [(a, b) for a, b in segs if b - a >= 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.02)
    args = ap.parse_args()
    clf, thr, rec, nb, nm = train_detector(args.fpr)
    print(f"frame detector (GroupKFold by session on HumBugDB): threshold p>={thr:.3f} -> held-out FP/frame {args.fpr:.3f} "
          f"({nb} neg frames), recall/frame {rec:.3f} ({nm} pos frames)", flush=True)

    m = pd.read_csv("data/processed/manifest_files.csv")
    ab = m[m.dataset == "abuzz"]
    rows, bg_fp, fu = [], [], []
    for r in ab.itertuples():
        x, _ = librosa.load(r.path, sr=SR)
        starts, F = frame_features(x)
        if not len(F):
            continue
        active = clf.predict_proba(F)[:, 1] >= thr
        if r.sound_type == "background":
            bg_fp += list(active)
            continue
        for a, b in segments(starts, active):
            rows.append(dict(path=r.path, start_s=round(a, 2), end_s=round(min(b, len(x) / SR), 2)))
    # sanity check on a dataset the detector never saw: Funchal clips are all mosquito (0.3 s, padded to 1 frame)
    for p in m[m.dataset == "funchal"].path.sample(300, random_state=0):
        x, _ = librosa.load(p, sr=SR)
        _, F = frame_features(np.pad(x, (0, max(0, SR - len(x)))))
        if len(F):
            fu += list(clf.predict_proba(F)[:, 1] >= thr)
    seg = pd.DataFrame(rows)
    seg.to_csv("data/processed/abuzz_segments.csv", index=False)
    info = dict(method="HistGradientBoosting on log-mel frame stats", threshold_p=thr, target_fpr=args.fpr,
                humbugdb_heldout_recall=rec, abuzz_bg_fp=float(np.mean(bg_fp)), abuzz_bg_frames=len(bg_fp),
                funchal_recall=float(np.mean(fu)))
    Path("results").mkdir(exist_ok=True)
    json.dump(info, open("results/activity_threshold.json", "w"), indent=1)
    print(f"Abuzz background recordings: FP/frame {info['abuzz_bg_fp']:.3f} ({len(bg_fp)} frames) | Funchal recall {info['funchal_recall']:.3f}")
    seg["dur"] = seg.end_s - seg.start_s
    tot = ab[ab.sound_type == "mosquito"].duration_s.sum()
    print(f"segments: {len(seg)}, active {seg.dur.sum() / 60:.1f} min of {tot / 60:.1f} min mosquito-file duration")


if __name__ == "__main__":
    main()
