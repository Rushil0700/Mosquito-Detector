"""Apply the calibration quality gate to held-out (tune / test) windows, so the realism check compares synthetic
clips with real windows that actually contain a clear flight tone (many Abuzz windows are silence or handling noise).

Gate = the same test calibration uses: tone SNR >= 8 dB, f0 inside the class search range, harmonics consistent.
Writes data/processed/manifest_windows_gated.csv (tune + test rows that pass, with their f0).

Usage: python -m src.data.tone_gate --config configs/synth_mumbai.yaml
"""
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import yaml

from src.synth.calibrate import analyse_clip, plausible_range


def _gate(job):
    warnings.filterwarnings("ignore")
    path, lo, hi = job
    x, _ = librosa.load(path, sr=8000)
    r = analyse_clip(x, 8000, f0_range=(lo, hi), check_harmonics=True)
    return r["f0"] if r else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_mumbai.yaml")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    w = pd.read_csv("data/processed/manifest_windows.csv")
    w = w[w.split.isin(["tune", "test"])].copy()
    jobs = [(p, *plausible_range(cfg["priors"], c, "unknown")) for p, c in zip(w.path, w.cls)]
    with ProcessPoolExecutor(args.workers) as ex:
        w["f0_gate"] = list(ex.map(_gate, jobs, chunksize=16))
    kept = w[w.f0_gate.notna()]
    kept.to_csv("data/processed/manifest_windows_gated.csv", index=False)
    s = w.assign(ok=w.f0_gate.notna()).groupby(["cls", "split"]).ok.agg(["sum", "size"])
    s["pass_%"] = (100 * s["sum"] / s["size"]).round(0)
    print(s.to_string())


if __name__ == "__main__":
    main()
