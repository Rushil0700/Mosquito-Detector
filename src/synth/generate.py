"""Generate synthetic mosquito flight-tone clips from the calibrated bank.

Each clip = synthetic tone (parameters of one real calibration clip) + a real recording residual from
the noise bank (same dataset and original sample rate), mixed to the measured tone SNR, then passed
through that source's device chain (level, 16-bit grid for native 8 kHz). Clip length = residual length.

`source` = "calibrated" (tone parameters from real clips of that class/sex) or "prior" (e.g. males:
female parameters x literature male/female f0 ratio). Synthetic clips are for TRAINING ONLY.

Usage: python -m src.synth.generate --n 500 [--classes ...] [--sex unknown male]
"""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from src.synth.calibrate import blind_tone_snr, peak_sharpness
from src.synth.wingbeat import ToneParams, augment_channel, device_output, mix_at_snr, mix_at_tnr, synth_tone


def class_cfg(cfg, cls):
    """Config for one class: `species.<cls>` blocks (tuning / flight / room / source) override the top level."""
    out = copy.deepcopy(cfg)
    for block, over in (cfg.get("species", {}).get(cls) or {}).items():
        if isinstance(over, dict):
            out.setdefault(block, {}).update(over)
        else:
            out[block] = over
    return out


class Equalizer:
    """Measured per-class correction: smoothed, limited median(real) / median(synthetic) spectrum ratio.

    Fitted on the tune split by src/synth/fit_eq.py; applied as a zero-phase gain to the synthetic tone
    before mixing (`eq.apply_to: tone`, default) or to the whole mix (`mix`).
    """

    def __init__(self, path=None, curves=None):
        self.curves = curves if curves is not None else (json.load(open(path)) if path and Path(path).exists() else {})

    def __call__(self, x, sr, cls):
        c = self.curves.get(cls)
        if not c:
            return x
        f = np.fft.rfftfreq(len(x), 1 / sr)
        g = 10 ** (np.interp(f, c["freq"], c["gain_db"]) / 20)
        return np.fft.irfft(np.fft.rfft(x) * g, len(x))


class NoiseBank:
    """Real recording residuals, pooled by `pool_by` keys of the source clip; cached in memory.

    Pooling by (dataset, orig_sr, cls) keeps residuals within one recording batch: in Funchal each
    species was recorded separately, so its residuals carry that batch's rumble and handling noise.
    """

    def __init__(self, bank, sr, pool_by=("dataset", "orig_sr", "cls")):
        self.sr, self.pool_by = sr, tuple(pool_by)
        self.by_key = defaultdict(list)
        for cls, by_sex in bank.items():
            for entries in by_sex.values():
                for e in entries:
                    self.by_key[self._key(e, cls)].append(e["residual"])
        self._cache = {}

    def _key(self, e, cls):
        return tuple(cls if k == "cls" else e[k] for k in self.pool_by)

    def load(self, path):
        if path not in self._cache:
            self._cache[path] = librosa.load(path, sr=self.sr)[0]
        return self._cache[path]

    def pick(self, entry, pairing, rng, cls=None):
        if pairing == "same_clip":
            path = entry["residual"]
        else:
            pool = self.by_key[self._key(entry, cls)]
            path = pool[rng.integers(len(pool))]
        return self.load(path), path


class DriftMap:
    """Pitch-drift std (Hz) that reproduces a clip's measured peak sharpness, found by simulating tones.

    The frame-wise pitch-track CV is smoothed and misses fast wobble, so for `drift_from: sharpness`
    each synthetic clip gets the drift that makes its fundamental as sharp as the real source clip's.
    """
    GRID_HZ = np.array([0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 55])

    def __init__(self, cfg, n_sims=10):
        self.cfg, self.n_sims, self.tables = cfg, n_sims, {}

    def table(self, n):
        if n not in self.tables:
            rng, sr, t = np.random.default_rng(0), self.cfg["sample_rate"], self.cfg["tuning"]
            f0, vals = 500.0, []
            for sig in self.GRID_HZ:
                sh = []
                for _ in range(self.n_sims):
                    p = ToneParams(f0, np.array([0.0, -3.0, -12.0]), np.zeros(3), sig / f0, t["drift_tau_s"],
                                   4.5 * t["am_scale"], t["am_tau_s"], t["harmonic_jitter_db"])
                    sh.append(peak_sharpness(synth_tone(p, n, sr, self.cfg["flight"], 3990, rng)[0], sr, f0))
                vals.append(np.mean(sh))
            self.tables[n] = np.minimum.accumulate(np.array(vals))  # enforce monotone decreasing
        return self.tables[n]

    MAX_CV = 0.08  # physiological cap (Arthur 2014: per-flight CV ~0.03 F / ~0.04 M)

    def drift_cv(self, target_sharp, f0, n):
        t = self.table(n)
        return min(float(np.interp(target_sharp, t[::-1], self.GRID_HZ[::-1])) / f0, self.MAX_CV)


def pick_entry(bank, cls, sex, rng, min_real=1):
    """Return (bank entry, needs_male_prior) for the requested class/sex.

    Uses a real entry of that sex with probability min(1, n_real / min_real); otherwise a female (or, for banks
    without sex groups, unknown) entry, flagged so the caller scales its f0 by the literature male/female ratio.
    `unassigned` entries (sex unknown, f0 ambiguous) are never tone sources - their residuals are still noise.
    """
    by_sex = bank.get(cls, {})
    real = by_sex.get(sex, [])
    # enough real entries -> always real; a few -> real with probability n / min_real, else derived from a female
    if real and rng.random() < len(real) / min_real:
        return real[rng.integers(len(real))], False
    pool = by_sex.get("female") or by_sex.get("unknown")
    if not pool:
        return None, False
    return pool[rng.integers(len(pool))], sex == "male"


def make_tone(entry, cfg, rng, n, f0_mult=1.0, drift_map=None):
    """Clean synthetic flight tone (no noise, no device) of n samples from one calibration entry.

    Returns (tone, trajectory, per-sample f0, ToneParams). `cfg` = class config (`class_cfg(cfg, cls)`).
    """
    sr, s, t = cfg["sample_rate"], cfg["source"], cfg["tuning"]
    levels = entry.get("harmonic_energy_db") if s.get("harmonic_levels", "energy") == "energy" else None
    levels = entry["harmonic_db"] if levels is None else levels
    n_h = len(levels)
    harm = (np.nan_to_num(np.array(levels, float), nan=-40.0)
            + t["harmonic_tilt_db"] * np.arange(n_h)          # dB per harmonic above the 1st (+ or -)
            + rng.normal(0, s["harmonic_profile_jitter_db"], n_h))
    phase = np.array(entry["harmonic_phase"], float) if s["use_measured_phase"] else rng.uniform(-np.pi, np.pi, n_h)
    params = ToneParams(
        f0_hz=entry["f0"] * f0_mult * (1 + rng.normal(0, s["f0_scale_jitter"])),
        harmonic_db=harm,
        harmonic_phase=phase,
        drift_cv=t["drift_scale"] * (drift_map.drift_cv(entry["sharp_8hz"], entry["f0"], n)
                                     if t["drift_from"] == "sharpness" and drift_map else entry["track_cv"]),
        drift_tau_s=t["drift_tau_s"],
        am_db=8.686 * entry["am_cv"] * t["am_scale"],
        am_tau_s=t["am_tau_s"],
        harmonic_jitter_db=t["harmonic_jitter_db"],
        fm_fast_cv=t.get("fast_fm_scale", 1.0) * entry.get("fm_fast_cv", 0.0),
        am_fast_cv=t.get("fast_am_scale", 1.0) * entry.get("am_fast_cv", 0.0),
        fast_bw_hz=t.get("fast_bw_hz", 80.0),
    )
    tone, path, f0 = synth_tone(params, n, sr, cfg["flight"], s["max_harmonic_hz"], rng, cfg.get("room"))
    return tone, path, f0, params


def synthesize(entry, cfg, rng, noise_bank, f0_mult=1.0, drift_map=None, cls=None, eq=None):
    """One synthetic clip from one calibration entry. Returns (float32 waveform, metadata).

    `cfg` should already be the class config (`class_cfg(cfg, cls)`).
    """
    sr, s, t = cfg["sample_rate"], cfg["source"], cfg["tuning"]
    noise, noise_path = noise_bank.pick(entry, cfg["noise"]["pairing"], rng, cls)
    n = len(noise)
    tone, path, f0, params = make_tone(entry, cfg, rng, n, f0_mult, drift_map)
    eq_on = eq is not None and cfg.get("eq", {}).get("enabled")
    if eq_on and cfg["eq"].get("apply_to", "tone") == "tone":
        tone = eq(tone, sr, cls)   # the noise is a real recording: correct only the synthetic part
    f0_mean = float(np.mean(f0))
    if cfg["noise"]["mix"] == "tnr":   # tone-to-noise energy ratio measured on the source clip (stable)
        target = entry["tnr_db"] + t.get("tnr_offset_db", 0.0) + rng.normal(0, t["snr_jitter_db"])
        x = mix_at_tnr(tone, noise, target)
        got = blind_tone_snr(x, sr, f0_mean)
    else:                              # peak-SNR estimator (noisy at low SNR)
        target = entry["snr_db"] + rng.normal(0, t["snr_jitter_db"])
        x, got = mix_at_snr(tone, noise, target, lambda y: blind_tone_snr(y, sr, f0_mean), tol_db=cfg["noise"]["snr_tol_db"])
    if eq_on and cfg["eq"].get("apply_to", "tone") == "mix":
        x = eq(x, sr, cls)
    if cfg["augment_channel"]["enabled"]:
        x = augment_channel(x, sr, cfg["augment_channel"], rng)
    x = device_output(x, entry["peak_dbfs"] + rng.normal(0, t["level_jitter_db"]), entry["orig_sr"])
    r = np.linalg.norm(path, axis=1)
    return x, dict(f0_hz=round(params.f0_hz, 1), drift_cv=round(params.drift_cv, 4), am_db=round(params.am_db, 2),
                   mix=cfg["noise"]["mix"], target_db=round(target, 1), measured_snr_db=round(float(got), 1),
                   dist_min_m=round(r.min(), 3), dist_max_m=round(r.max(), 3), orig_sr=entry["orig_sr"],
                   bank_file=entry["file"], noise_file=noise_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth.yaml")
    ap.add_argument("--n", type=int, default=500, help="clips per class/sex")
    ap.add_argument("--classes", nargs="*")
    ap.add_argument("--sex", nargs="*", default=["female", "male"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rng = np.random.default_rng(cfg["seed"])
    bank = json.load(open(cfg["calibration_file"]))
    noise_bank = NoiseBank(bank, cfg["sample_rate"], cfg["noise"]["pool_by"])
    eq = Equalizer(cfg.get("eq", {}).get("file"))
    out_dir = Path(cfg["out_dir"])

    rows = []
    for cls in args.classes or sorted(bank):
        ccfg = class_cfg(cfg, cls)
        drift_map = DriftMap(ccfg)
        for sex in args.sex:
            d = out_dir / f"{cls}_{sex}"
            d.mkdir(parents=True, exist_ok=True)
            for i in range(args.n):
                entry, male_prior = pick_entry(bank, cls, sex, rng, cfg.get("min_real_males", 1) if sex == "male" else 1)
                if entry is None:
                    print(f"skip {cls}/{sex}: no calibration data")
                    break
                mult = rng.uniform(*cfg["male_female_ratio"]) if male_prior else 1.0
                x, meta = synthesize(entry, ccfg, rng, noise_bank, mult, drift_map, cls, eq)
                p = d / f"syn_{cls}_{sex}_{i:05d}.wav"
                # Native 8 kHz sources are 16-bit; resampled sources stay float like the real loaded clips.
                sf.write(p, x, cfg["sample_rate"], subtype="PCM_16" if meta["orig_sr"] == 8000 else "FLOAT")
                rows.append(dict(path=str(p), cls=cls, sex=sex, dataset="synthetic",
                                 source="prior" if male_prior else "calibrated",
                                 sex_source="prior" if male_prior else entry.get("sex_source", "label"), **meta))
    man = pd.DataFrame(rows)
    man_path = out_dir.parent / f"manifest_{out_dir.name}.csv"  # one manifest per output set (never overwrite another set's)
    man.to_csv(man_path, index=False)
    print(man.groupby(["cls", "sex", "source", "sex_source"]).size().to_string())
    print(f"wrote {len(man)} clips -> {out_dir}, manifest {man_path}")


if __name__ == "__main__":
    main()
