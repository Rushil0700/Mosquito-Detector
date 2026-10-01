"""Synthetic set v2: synthetic flight tones on RANDOMISED backgrounds + matching no-mosquito clips (training only).

Why: in v1 each species' synthetic clips carried background noise from that species' own lab, so they reinforced the
model's shortcut "sounds like Oxford -> Culex". Here every clip's noise comes from a random training site / device or a
training household environment, and the no-mosquito clips use the same pool and the same device randomisation.

Usage: python -m src.synth.generate_v2 [--config configs/synth_v2.yaml] [--workers 8]
       python -m src.synth.generate_v2 --config configs/synth_v2_other.yaml   # the 16 other mosquito species
Writes <out_dir>/<class>_<sex>/*.wav, <out_dir>/negative/*.wav and data/synthetic/manifest_clips_v2.csv.
"""
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy import signal

from src.synth.generate import DriftMap, class_cfg, make_tone, pick_entry
from src.synth.wingbeat import augment_channel

SR = 8000
BAND = (250, 3900)


class NoisePool:
    """Random 1 s noise beds: real training background windows (any site) or DEMAND training environments."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        seg = pd.read_csv("data/processed/segments_split.csv")
        bg = seg[(seg.split == "train") & (seg.presence == 0) & (seg.dur >= cfg["clip_seconds"])]
        # sample a SITE uniformly first, then a segment (weighted by duration): no single site dominates the noise
        self.bg = {site: g[["path", "start_s", "dur", "site"]].values for site, g in bg.groupby("site")}
        self.bg_w = {site: g.dur.values / g.dur.sum() for site, g in bg.groupby("site")}
        root = Path(cfg["noise"]["household_dir"])
        self.house = [(env, p) for env in cfg["noise"]["household_train_envs"] for p in sorted((root / env).glob("ch*.wav"))]
        assert not set(cfg["noise"]["household_train_envs"]) & set(cfg["noise"]["household_test_envs"])

    def draw(self, rng, n):
        kind = rng.choice(list(self.cfg["noise"]["mix_prob"]), p=list(self.cfg["noise"]["mix_prob"].values()))
        if kind == "background":
            site = list(self.bg)[rng.integers(len(self.bg))]
            path, start, dur, site = self.bg[site][rng.choice(len(self.bg[site]), p=self.bg_w[site])]
            off = start + rng.uniform(0, dur - n / SR)
            x, _ = librosa.load(path, sr=SR, offset=off, duration=n / SR)
            src = f"background:{site}"
        else:
            env, path = self.house[rng.integers(len(self.house))]
            info = sf.info(path)
            k = int(n * info.samplerate / SR)
            a = int(rng.integers(0, info.frames - k))
            y, fs = sf.read(path, start=a, frames=k, dtype="float32")
            x = librosa.resample(y, orig_sr=fs, target_sr=SR)
            src = f"household:{env}"
        x = np.resize(x, n) if len(x) < n else x[:n]
        return x.astype(np.float64), src


def band_energy(x):
    sos = signal.butter(4, [BAND[0] / (SR / 2), BAND[1] / (SR / 2)], "bandpass", output="sos")
    return float(np.mean(signal.sosfilt(sos, x) ** 2)) + 1e-20


def noise_bed(pool, cfg, rng, n):
    x, src = pool.draw(rng, n)
    if rng.random() < cfg["noise"]["second_layer_prob"]:
        y, src2 = pool.draw(rng, n)
        g = 10 ** (rng.uniform(*cfg["noise"]["second_layer_db"]) / 20)
        x = x + g * y * np.std(x) / (np.std(y) + 1e-12)
        src += "+" + src2
    return x, src


def device(x, cfg, rng):
    d = cfg["device"]
    x = augment_channel(x, SR, dict(highpass_hz=d["highpass_hz"], lowpass_hz=d["lowpass_hz"], n_peaking_eq=d["n_peaking_eq"],
                                    peaking_gain_db=d["peaking_gain_db"], self_noise_db=d["self_noise_db"],
                                    clip_probability=d["clip_probability"]), rng)
    x = x / (np.max(np.abs(x)) + 1e-12) * 10 ** (rng.uniform(*d["peak_dbfs"]) / 20)
    q = rng.random() < d["int16_probability"]
    if q:
        x = np.round(x * 32767) / 32767
    return np.clip(x, -1, 1).astype(np.float32), q


def _job(job):
    warnings.filterwarnings("ignore")
    kind, cls, sex, ids, seed, cfg_path = job
    cfg = yaml.safe_load(open(cfg_path))
    tone_cfg = yaml.safe_load(open(cfg["tone_config"]))
    bank = json.load(open(tone_cfg["calibration_file"]))
    rng = np.random.default_rng(seed)
    pool = NoisePool(cfg, rng)
    n = int(cfg["clip_seconds"] * SR)
    out = Path(cfg["out_dir"]) / (f"{cls}_{sex}" if kind == "mosquito" else "negative")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if kind == "mosquito":
        ccfg = class_cfg(tone_cfg, cls)
        dm = DriftMap(ccfg)
    for i in ids:
        noise, nsrc = noise_bed(pool, cfg, rng, n)
        meta = dict(presence=0, cls="", sex="", source="noise_only", sex_source="", snr_band_db=np.nan, f0_hz=np.nan)
        x = noise
        if kind == "mosquito":
            if sex == "any":  # species without a sex model: all its calibrated windows form one pool
                allent = [e for es in bank[cls].values() for e in es]
                entry, male_prior = allent[rng.integers(len(allent))], False
            else:
                entry, male_prior = pick_entry(bank, cls, sex, rng, tone_cfg.get("min_real_males", 1) if sex == "male" else 1)
            mult = rng.uniform(*tone_cfg["male_female_ratio"]) if male_prior else 1.0
            tone, _, f0, params = make_tone(entry, ccfg, rng, n, mult, dm)
            snr = rng.uniform(*cfg["snr_db"])
            g = np.sqrt(band_energy(noise) * 10 ** (snr / 10) / band_energy(tone))
            x = g * tone + noise
            meta = dict(presence=1, cls=cls, sex=sex, source="prior" if male_prior else "calibrated",
                        sex_source="prior" if male_prior else ("none" if sex == "any" else entry.get("sex_source", "label")),
                        snr_band_db=round(snr, 1), f0_hz=round(params.f0_hz, 1))
        x, q16 = device(x, cfg, rng)
        p = out / f"v2_{kind}_{cls or 'none'}_{sex or 'none'}_{i:05d}.wav"
        sf.write(p, x, SR, subtype="PCM_16" if q16 else "FLOAT")
        rows.append(dict(path=str(p), dataset="synthetic_v2", noise=nsrc, int16=q16, **meta))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_v2.yaml")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tone_cfg = yaml.safe_load(open(cfg["tone_config"]))
    bank = json.load(open(tone_cfg["calibration_file"]))
    chunk = 250
    jobs, seed = [], cfg["seed"] * 1000
    for cls in sorted(bank):
        for sex in cfg.get("sexes", ["female", "male"]):
            for a in range(0, cfg["n_per_class_sex"], chunk):
                seed += 1
                jobs.append(("mosquito", cls, sex, list(range(a, min(a + chunk, cfg["n_per_class_sex"]))), seed, args.config))
    for a in range(0, cfg["n_negative"], chunk):
        seed += 1
        jobs.append(("negative", "", "", list(range(a, min(a + chunk, cfg["n_negative"]))), seed, args.config))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for part in ex.map(_job, jobs) for r in part]
    man = pd.DataFrame(rows)
    mpath = cfg.get("manifest", "data/synthetic/manifest_clips_v2.csv")
    man.to_csv(mpath, index=False)
    pd.set_option("display.width", 200)
    print(man.groupby(["presence", "cls", "sex", "source"], dropna=False).size().to_string())
    print("\nnoise bed sources (share):")
    print(man.noise.str.split(":").str[0].value_counts(normalize=True).round(3).to_string())
    print(f"\nwrote {len(man)} clips -> {cfg['out_dir']}, manifest {mpath}")


if __name__ == "__main__":
    main()
