"""Fit the measured correction filter: per class, median real / median synthetic spectrum on the TUNE split.

Spectra are RMS-normalised log-power (Welch), medians over clips, averaged over seeds; the dB difference is
smoothed (moving average over `eq.smooth_hz`) and limited to +-`eq.max_db`. Applied by generate.synthesize
when `eq.enabled`, by default to the synthetic tone only (the noise is a real recording), so a gain on the
tone moves the mixed spectrum less than 1:1 where noise dominates: the fit is iterated (--iters), each
round adding the remaining real/synthetic difference. Check the effect on the TEST split only.

Usage: python -m src.synth.fit_eq --real data/processed/manifest_funchal.csv
"""
import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.synth.calibrate import log_spectrum
from src.synth.generate import Equalizer, NoiseBank
from src.synth.validate import real_set, synth_set


def median_spectrum(wavs, sr):
    return np.median([log_spectrum((x - x.mean()) / (x.std() + 1e-12), sr, 1024)[1] for x in wavs], 0)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--config", default="configs/synth.yaml")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sr, e = cfg["sample_rate"], cfg["eq"]
    raw = copy.deepcopy(cfg)
    raw["eq"]["enabled"] = False
    bank = json.load(open(cfg["calibration_file"]))
    nb = NoiseBank(bank, sr, cfg["noise"]["pool_by"])
    real = pd.read_csv(args.real)
    real = real[real.split == "tune"]
    f = log_spectrum(np.random.default_rng(0).standard_normal(sr), sr, 1024)[0]
    df = f[1] - f[0]
    width = max(1, int(round(e["smooth_hz"] / df)))
    smooth = lambda d: np.convolve(np.pad(d, width, mode="edge"), np.ones(width) / width, mode="same")[width:-width]
    gain = {cls: np.zeros_like(f) for cls in sorted(set(real.cls) & set(bank))}
    on = copy.deepcopy(cfg)
    on["eq"]["enabled"] = True
    for it in range(args.iters):
        eq = Equalizer(curves={c: {"freq": f.tolist(), "gain_db": g.tolist()} for c, g in gain.items()})
        for cls in gain:
            diffs = []
            for seed in args.seeds:
                _, rw = real_set(real, cls, sr, args.n, seed)
                _, sw = synth_set(on if it else raw, bank, nb, cls, len(rw), np.random.default_rng(seed), eq)
                diffs.append(median_spectrum(rw, sr) - median_spectrum(sw, sr))
            d = smooth(np.mean(diffs, 0))
            gain[cls] = np.clip(gain[cls] + d, -e["max_db"], e["max_db"])
            print(f"iter {it + 1} {cls}: remaining |real - synth| mean {np.abs(d).mean():.2f} dB, max {np.abs(d).max():.2f} dB", flush=True)
    curves = {}
    bands = [(0, 300), (300, 1000), (1000, 2500), (2500, 3500), (3500, 4000)]
    for cls, g in gain.items():
        curves[cls] = {"freq": f[::4].round(2).tolist(), "gain_db": g[::4].round(3).tolist()}
        print(f"{cls}: mean gain per band " + ", ".join(f"{a}-{b} Hz {g[(f >= a) & (f < b)].mean():+.1f} dB" for a, b in bands)
              + f"; max |gain| {np.abs(g).max():.1f} dB")
    Path(e["file"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(curves, open(e["file"], "w"))
    print(f"wrote {e['file']}")


if __name__ == "__main__":
    main()
