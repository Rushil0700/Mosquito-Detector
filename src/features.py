"""M2: log-mel features for real segments (per split) and synthetic clips (training only).

Writes to `out_dir` (configs/features.yaml):
  <split>.npy          float16 [N, n_mels, frames] log-mel (dB) windows
  <split>_meta.csv     one row per window: source path, window start, labels (presence / species / genus / sex),
                       session, site, dataset, padded flag
  synth_train.npy/.csv synthetic clips (all from training-session calibration), same format
  norm.json            per-mel-band mean / std over REAL training windows (applied at training time)

Usage: python -m src.features [--synthetic data/synthetic/manifest_clips_mumbai_v1_1.csv]
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(open("configs/features.yaml"))
GENUS = {"ae": "Aedes", "an": "Anopheles", "cx": "Culex", "cs": "Culiseta", "ma": "Mansonia"}
TARGET = ["ae_aegypti", "ae_albopictus", "an_stephensi", "cx_quinquefasciatus"]


def logmel(windows):
    """[n, samples] -> [n, n_mels, frames] log-mel in dB (absolute level kept; normalisation happens at training)."""
    c = CFG
    m = librosa.feature.melspectrogram(y=windows, sr=c["sample_rate"], n_fft=c["n_fft"], hop_length=c["hop_length"],
                                       n_mels=c["n_mels"], fmin=c["fmin"], fmax=c["fmax"], center=True)
    return librosa.power_to_db(m, ref=1.0, amin=1e-10, top_db=None).astype(np.float16)


def cut_windows(x, sr, cap, rng):
    """Window starts (s) and stacked 1 s windows of one segment; wrap-pads 0.4-1 s segments."""
    n, hop = int(CFG["window_s"] * sr), int(CFG["hop_s"] * sr)
    padded = False
    if len(x) < n:
        if len(x) < CFG["min_segment_s"] * sr:
            return [], np.zeros((0, n), np.float32), False
        x = np.resize(x, n)  # repeat the audio (no silence, which would itself be a giveaway)
        padded = True
    starts = np.arange(0, len(x) - n + 1, hop)
    if cap is not None and len(starts) > cap:
        starts = np.sort(rng.choice(starts, cap, replace=False))
    return (starts / sr).tolist(), np.stack([x[s:s + n] for s in starts]).astype(np.float32), padded


def _segment_job(job):
    warnings.filterwarnings("ignore")
    i, path, start, dur, cap, seed = job
    sr = CFG["sample_rate"]
    try:
        x, _ = librosa.load(path, sr=sr, offset=start, duration=dur)
    except Exception as e:
        return i, [], None, False, repr(e)
    starts, W, padded = cut_windows(x, sr, cap, np.random.default_rng(seed))
    return i, starts, (logmel(W) if len(W) else None), padded, None


def extract_real(workers, splits=None):
    seg = pd.read_csv("data/processed/segments_split.csv", keep_default_na=False)
    out = Path(CFG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for split, g in seg.groupby("split"):
        if splits and split not in splits:
            continue
        cap = CFG["max_windows_per_segment"].get(split)
        # the cap stops long MOSQUITO recordings dominating; background is never capped (the detector needs it)
        jobs = [(i, r.path, r.start_s, r.dur, cap if r.presence == 1 else None, CFG["seed"] * 1_000_003 + i)
                for i, r in zip(g.index, g.itertuples())]
        feats, meta, errors = [], [], 0
        with ProcessPoolExecutor(workers) as ex:
            for i, starts, F, padded, err in ex.map(_segment_job, jobs, chunksize=4):
                if err:
                    errors += 1
                    continue
                if F is None:
                    continue
                r = seg.loc[i]
                feats.append(F)
                for s in starts:
                    meta.append(dict(src=r.path, seg_start_s=r.start_s, win_start_s=round(r.start_s + s, 3), padded=padded,
                                     presence=r.presence, species=r.species, genus=r.genus, sex=r.sex, session=r.session,
                                     site=r.site, dataset=r.dataset, device=r.device, split=split, synthetic=False))
        X = np.concatenate(feats) if feats else np.zeros((0, CFG["n_mels"], 63), np.float16)
        np.save(out / f"{split}.npy", X)
        M = pd.DataFrame(meta)
        M.to_csv(out / f"{split}_meta.csv", index=False)
        report.append(dict(split=split, windows=len(M), mosquito=int(M.presence.sum()), background=int((M.presence == 0).sum()),
                           padded=int(M.padded.sum()), read_errors=errors, hours=round(len(M) * CFG["hop_s"] / 3600, 2)))
        print(report[-1], flush=True)
    return pd.DataFrame(report)


def _synth_job(job):
    warnings.filterwarnings("ignore")
    i, path = job
    x, _ = librosa.load(path, sr=CFG["sample_rate"])
    n = int(CFG["window_s"] * CFG["sample_rate"])
    x = np.resize(x, n) if len(x) < n else x[:n]
    return i, logmel(x[None, :])[0]


def extract_synthetic(manifest, workers, name="synth_train"):
    """Synthetic clips -> <name>.npy / <name>_meta.csv. v1 manifests are all mosquito; v2 manifests carry `presence`
    (noise-only clips = 0) and `snr_band_db`."""
    s = pd.read_csv(manifest)
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_synth_job, list(zip(s.index, s.path)), chunksize=64))
    X = np.stack([res[i] for i in s.index])
    out = Path(CFG["out_dir"])
    np.save(out / f"{name}.npy", X)
    presence = s.presence if "presence" in s else pd.Series(1, index=s.index)
    cls = s.cls.fillna("") if "cls" in s else pd.Series("", index=s.index)
    # species label as in the real data: the 4 target species by name, every other mosquito species = other_mosquito
    species = cls.where(cls.isin(TARGET) | (cls == ""), "other_mosquito")
    M = pd.DataFrame(dict(src=s.path, seg_start_s=0.0, win_start_s=0.0, padded=False, presence=presence, species=species,
                          species_raw=cls, genus=cls.str[:2].map(GENUS).fillna(""), sex=s.sex.fillna(""),
                          sex_source=s.sex_source.fillna(""), synth_source=s.source,
                          snr_band_db=s.snr_band_db if "snr_band_db" in s else np.nan,
                          noise=s.noise if "noise" in s else "", session="synthetic", site="synthetic",
                          dataset=s.dataset if "dataset" in s else "synthetic", device="synthetic", split="train", synthetic=True))
    M.to_csv(out / f"{name}_meta.csv", index=False)
    print(f"{name}: {len(M)} windows ({int(M.presence.sum())} mosquito, {int((M.presence == 0).sum())} noise-only)", flush=True)


def norm_stats():
    out = Path(CFG["out_dir"])
    X = np.load(out / "train.npy", mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(X), min(len(X), 50000), replace=False)
    S = X[np.sort(idx)].astype(np.float32)
    stats = dict(mean=S.mean(axis=(0, 2)).round(4).tolist(), std=S.std(axis=(0, 2)).round(4).tolist(),
                 n_windows=int(len(idx)), note="per mel band, REAL training windows only")
    json.dump(stats, open(out / "norm.json", "w"), indent=1)
    print("norm.json written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--synthetic", default=None)
    ap.add_argument("--synthetic-name", default="synth_train")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--splits", nargs="*", help="only these real splits (default: all)")
    args = ap.parse_args()
    if not args.skip_real:
        rep = extract_real(args.workers, args.splits)
        path = Path(CFG["out_dir"]) / "report.csv"
        if args.splits and path.exists():  # keep the other splits' rows
            old = pd.read_csv(path)
            rep = pd.concat([old[~old.split.isin(args.splits)], rep], ignore_index=True)
        rep.to_csv(path, index=False)
        norm_stats()
    if args.synthetic:
        extract_synthetic(args.synthetic, args.workers, args.synthetic_name)


if __name__ == "__main__":
    main()
