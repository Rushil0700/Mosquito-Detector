"""Measure flight-tone parameters on REAL recordings -> empirical bank + noise bank for the synthesizer.

Per clip (only rows with split == "calib" when a split column is present):
  f0 searched inside the plausible range for the class (literature priors +-3 SD; unsexed clips accept
  the female or male range), which re-estimates octave / subharmonic errors; entries with weak or
  ambiguous harmonic combs are dropped;
  fine pitch track (4 ms coarse track refined from the phase of every demodulated harmonic);
  per-harmonic complex envelopes -> tone resynthesis; harmonic levels (spectral peak and tone energy,
  dB re. loudest) and relative phases psi_k = phi_k - k*phi_1; tone AM CV; tone SNR; peak level; DC.
  Residual = clip minus the resynthesised tone (tone-dominated STFT bins refilled from the same
  recording's noise, `refill`; weaker bins Wiener-subtracted); saved as the noise bank background.
  The leftover tone in the residual is checked per harmonic against the local noise floor.

Usage: python -m src.synth.calibrate --manifest data/processed/manifest_funchal.csv [--refill borrow]
Manifest columns: path, cls, sex, dataset, orig_sr[, split].
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy import signal

F0_RANGE = (200.0, 1100.0)
N_HARM_SCORE = 5
MIN_SNR_DB = 8.0
TRACK_WIN, TRACK_HOP = 256, 32    # coarse track: 32 ms / 4 ms at 8 kHz (then phase-refined per sample)
TRACK_BW_HZ = 200.0               # pitch-correction bandwidth: follows wingbeat-to-wingbeat wobble
ENV_BW_HZ = (60.0, 20.0)          # envelope bandwidth of harmonic k = a + b*k (FM spread grows with k)
FAST_SPLIT_HZ = 20.0              # pitch / level modulation above this counts as fast (wingbeat-to-wingbeat)
MIN_HARM_GAP_DB = 6.0             # consistency: fundamental and >= 1 of harmonics 2-3 this far above the local floor


def log_spectrum(x, sr, nperseg):
    f, p = signal.welch(x, sr, nperseg=min(nperseg, len(x)), nfft=8192, noverlap=None)
    return f, 10 * np.log10(p + 1e-20)


def harmonic_sum_f0(f, spec_db, lo, hi, step=0.5, n_harm=N_HARM_SCORE, max_hz=3900):
    """f0 maximising the mean spectral level (above floor) at its first `n_harm` harmonics."""
    floor = np.median(spec_db)
    rel = np.maximum(spec_db - floor, 0)
    cands = np.arange(lo, hi, step)
    score = np.zeros_like(cands)
    for k in range(1, n_harm + 1):
        fk = cands * k
        ok = fk < max_hz
        score[ok] += np.interp(fk[ok], f, rel) / k ** 0.5  # mild emphasis on low harmonics
    return cands[np.argmax(score)], floor


def harmonic_levels(f, spec_db, f0, floor, max_hz=3990, tol=0.025):
    levels = []
    k = 1
    while k * f0 < max_hz and k <= 12:
        band = (f > k * f0 * (1 - tol)) & (f < k * f0 * (1 + tol))
        levels.append(float(spec_db[band].max()) if band.any() else np.nan)
        k += 1
    levels = np.array(levels)
    if not len(levels) or np.all(np.isnan(levels)):  # f0 invalid or above Nyquist
        return [], float("nan")
    snr = float(np.nanmax(levels) - floor)
    return (levels - np.nanmax(levels)).round(2).tolist(), snr


def peak_sharpness(x, sr, f0):
    """Fraction of the energy within +-100 Hz of the fundamental peak that lies within +-8 Hz of it.

    The centre is re-located to the spectral maximum within +-10 % of `f0`, so the measure does not
    depend on how the f0 estimate was obtained (1.0 = pure steady tone).
    """
    f, s = log_spectrum(x, sr, 2048)
    p = 10 ** (s / 10)
    win = np.abs(f - f0) <= 0.1 * f0
    fc = f[win][np.argmax(p[win])]
    return float(p[np.abs(f - fc) <= 8].sum() / (p[np.abs(f - fc) <= 100].sum() + 1e-20))


def blind_tone_snr(x, sr, f0_hint):
    """Tone SNR measured exactly as in calibration: f0 re-estimated from the spectrum (+-10 % of hint)."""
    f, s = log_spectrum(x, sr, 2048)
    if not np.isfinite(f0_hint) or f0_hint <= 0:
        return float("nan")
    f0, floor = harmonic_sum_f0(f, s, 0.9 * f0_hint, 1.1 * f0_hint)
    return harmonic_levels(f, s, f0, floor)[1]


def tone_snr(x, sr, f0):
    """Tone SNR (loudest harmonic above median spectral floor, dB) for a known f0."""
    f, s = log_spectrum(x, sr, 2048)
    return harmonic_levels(f, s, f0, np.median(s))[1]


def harmonic_excess(x, sr, f0, n_harm=7, nperseg=1024):
    """Per-harmonic peak level (within +-max(15 Hz, 2 %)) above the LOCAL floor between harmonics (dB).

    The floor is the median level 0.25-0.5 f0 away from the harmonic, so steep low-frequency noise
    does not count as tone (unlike `tone_snr`, which uses the whole-spectrum median).
    """
    f, s = log_spectrum(x, sr, nperseg)
    out = []
    for k in range(1, n_harm + 1):
        fk = k * f0
        if fk > sr / 2 - 60:
            break
        band = np.abs(f - fk) <= max(15.0, 0.02 * fk)
        fl = (np.abs(f - fk) > 0.25 * f0) & (np.abs(f - fk) < 0.5 * f0) & (f > 50) & (f < sr / 2 - 30)
        out.append(float(s[band].max() - np.median(s[fl])))
    return out


def plausible_range(priors, cls, sex, n_sd=3.0):
    """f0 range from literature priors (mean +- n_sd*SD); unsexed clips accept either sex."""
    p = priors.get(cls)
    if not p:
        return F0_RANGE
    if "search" in p:  # explicit search range (literature + data), see configs/synth_mumbai.yaml
        return tuple(p["search"])
    sexes = [sex] if sex in ("female", "male") else ["female", "male"]
    lo = min(p[s][0] - n_sd * p[s][1] for s in sexes)
    hi = max(p[s][0] + n_sd * p[s][1] for s in sexes)
    return max(lo, F0_RANGE[0]), min(hi, F0_RANGE[1])


def harmonics_consistent(f, spec_db, f0, f0_free):
    """Reject weak or ambiguous combs; accept octave / subharmonic re-estimates.

    `f0` was searched in the plausible range, `f0_free` over the full range. The fundamental and at least
    one of harmonics 2-3 must stand out from the LOCAL floor (0.1-0.25 f0 away). If the range restriction
    moved f0, the move must be an octave / subharmonic correction (ratio 2, 3 or 1/2); anything else
    means another tone dominates the clip, and the entry is dropped.
    """
    def local(fr):
        ring = (np.abs(f - fr) > 0.1 * f0) & (np.abs(f - fr) < 0.25 * f0)
        return spec_db[np.abs(f - fr) <= max(8.0, 0.02 * fr)].max() - np.median(spec_db[ring])
    h = [local(k * f0) for k in (1, 2, 3)]
    if h[0] < MIN_HARM_GAP_DB or max(h[1:]) < MIN_HARM_GAP_DB:
        return False
    if abs(f0 - f0_free) <= 5:
        return True
    return any(abs(f0 / f0_free - q) / q < 0.03 for q in (2, 3, 0.5))


def lowpass(z, sr, bw):
    """Zero-phase low-pass of a complex baseband signal."""
    sos = signal.butter(4, min(bw, 0.45 * sr) / (sr / 2), output="sos")
    return signal.sosfiltfilt(sos, z.real) + 1j * signal.sosfiltfilt(sos, z.imag)


def pitch_track(x, sr, f0):
    """Coarse frame-wise f0 (+-10 % around the clip f0), 4 ms hop, median-smoothed, per sample."""
    centers, track = [], []
    for i in range(0, len(x) - TRACK_WIN + 1, TRACK_HOP):
        ff, ss = log_spectrum(x[i:i + TRACK_WIN], sr, TRACK_WIN)
        track.append(harmonic_sum_f0(ff, ss, f0 * 0.9, f0 * 1.1, step=0.25, n_harm=3)[0])
        centers.append(i + TRACK_WIN // 2)
    track = signal.medfilt(np.array(track), 5) if len(track) >= 5 else np.array(track)
    return np.interp(np.arange(len(x)), centers, track)


def harmonic_tracker(x, sr, f0, n_iter=4):
    """Fine pitch track and per-harmonic complex envelopes.

    Demodulates every harmonic along the running phase Phi(t) = 2*pi*int f0; the phase drift of each
    baseband signal (weighted by |Z_k|^2 k^2, i.e. by how precisely it pins f0) corrects f0(t).
    Returns per-sample f0, Phi, envelopes Z (K x n, tone = sum_k 2 Re(Z_k e^{jk Phi})) and bandwidths.
    """
    fi = pitch_track(x, sr, f0)
    K = int((sr / 2 - 60) // (f0 * 1.1))
    ks = np.arange(1, K + 1)
    sos = signal.butter(2, TRACK_BW_HZ / (sr / 2), output="sos")
    for _ in range(n_iter):
        phi = 2 * np.pi * np.cumsum(fi) / sr
        Z = np.array([lowpass(x * np.exp(-1j * k * phi), sr, TRACK_BW_HZ) for k in ks])
        d = np.angle(Z[:, 1:] * np.conj(Z[:, :-1])) * sr / (2 * np.pi * ks[:, None])
        w = np.abs(Z[:, 1:]) ** 2 * ks[:, None] ** 2
        df = np.r_[0.0, (w * d).sum(0) / (w.sum(0) + 1e-30)]
        fi = np.clip(fi + signal.sosfiltfilt(sos, df), 0.8 * f0, 1.2 * f0)
    phi = 2 * np.pi * np.cumsum(fi) / sr
    bws = np.minimum(ENV_BW_HZ[0] + ENV_BW_HZ[1] * ks, 0.42 * f0)
    Z = np.array([lowpass(x * np.exp(-1j * k * phi), sr, b) for k, b in zip(ks, bws)])
    return fi, phi, Z, bws


def tone_residual(x, tone, sr, fi, bws, rng, mode="borrow", dominance=1.0, n_fft=512, hop=128):
    """Remove the resynthesised tone from `x` without disturbing the recording's own noise.

    Only STFT bins inside the harmonic bands (+-bw_k around k*f0(t)) are touched. The local noise power N
    is the median of the 6 nearest bins outside all bands in the same frame. Where the tone dominates
    (|T|^2 > dominance * N) the bin is replaced with this recording's noise (borrow: the value of a nearby
    out-of-band bin of the same frame, keeping the real level bumps; random: magnitude sqrt(N), random
    phase). Elsewhere the tone is subtracted with a Wiener weight |T|^2 / (|T|^2 + N), so bins where the
    tone estimate is mostly noise stay almost untouched. (Refilling the whole band instead changes the
    noise texture above 2 kHz - where the bands of higher harmonics cover most of the spectrum.)
    """
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop, center=True)
    T = librosa.stft(tone, n_fft=n_fft, hop_length=hop, center=True)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f0_frames = np.interp(np.arange(X.shape[1]) * hop, np.arange(len(fi)), fi)
    out = X.copy()
    for j, f0 in enumerate(f0_frames):
        band = np.zeros(len(freqs), bool)
        for k, b in enumerate(bws, start=1):
            band |= np.abs(freqs - k * f0) <= b
        keep = np.flatnonzero(~band & (freqs > 60))
        if not len(keep):
            keep = np.flatnonzero(freqs > 60)
        for b in np.flatnonzero(band & (freqs > 0)):
            near = keep[np.argsort(np.abs(freqs[keep] - freqs[b]))[:6]]
            noise = np.median(np.abs(X[near, j]) ** 2)
            t2 = np.abs(T[b, j]) ** 2
            if t2 > dominance * noise:
                out[b, j] = (X[near[rng.integers(len(near))], j] if mode == "borrow"
                             else np.sqrt(noise) * np.exp(2j * np.pi * rng.random()))
            else:
                out[b, j] = X[b, j] - T[b, j] * t2 / (t2 + noise + 1e-30)
    return librosa.istft(out, hop_length=hop, length=len(x))


def notch_residual(x, sr, fi, rng, n_fft=512, hop=128):
    """v0.2 noise residual: notch the harmonic comb (+-(32 Hz + 1 % k f0)) along the pitch track and refill the
    notched STFT bins with the median magnitude of neighbouring bins (+-150 Hz), random phase. Keeps part of the
    real tone's broad skirts, which the synthetic tone does not reproduce - on Funchal this scored better than the
    clean residual (results/experiments.md, M1g)."""
    Z = librosa.stft(x, n_fft=n_fft, hop_length=hop, center=True)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f0_frames = np.interp(np.arange(Z.shape[1]) * hop, np.arange(len(fi)), fi)
    for j, f0 in enumerate(f0_frames):
        mask = np.zeros(len(freqs), bool)
        for k in range(1, int(sr / 2 / f0) + 1):
            mask |= np.abs(freqs - k * f0) <= 32 + 0.01 * k * f0
        keep = np.flatnonzero(~mask & (freqs > 0))
        for b in np.flatnonzero(mask):
            near = keep[np.abs(freqs[keep] - freqs[b]) < 150]
            mag = np.median(np.abs(Z[near, j])) if len(near) else np.median(np.abs(Z[keep, j]))
            Z[b, j] = mag * np.exp(2j * np.pi * rng.random())
    return librosa.istft(Z, hop_length=hop, length=len(x))


def band_energy_db(x, resid, sr, fi, bws, nperseg=512):
    """Tone energy per harmonic (dB re. loudest): energy of the clip minus the residual in each band."""
    f, px = signal.welch(x, sr, nperseg=nperseg, nfft=4096)
    _, pr = signal.welch(resid, sr, nperseg=nperseg, nfft=4096)
    f0 = float(np.mean(fi))
    e = []
    for k, b in enumerate(bws, start=1):
        band = np.abs(f - k * f0) <= b
        e.append(max(px[band].sum() - pr[band].sum(), 1e-6 * px[band].sum() + 1e-30))
    e = 10 * np.log10(np.array(e))
    return (e - e.max()).round(2).tolist()


def analyse_clip(x, sr, rng=None, with_residual=False, f0_range=F0_RANGE, check_harmonics=False, refill="borrow",
                 residual="clean"):
    """Tone measurements for one clip, or None if too short / too noisy / inconsistent harmonics."""
    if len(x) < sr * 0.15:
        return None
    f, s = log_spectrum(x, sr, 2048)
    f0, floor = harmonic_sum_f0(f, s, *f0_range)
    levels, snr = harmonic_levels(f, s, f0, floor)
    if not snr >= MIN_SNR_DB:
        return None
    if check_harmonics and not harmonics_consistent(f, s, f0, harmonic_sum_f0(f, s, *F0_RANGE)[0]):
        return None
    out = {"f0": round(float(f0), 1), "harmonic_db": levels, "snr_db": round(snr, 1)}
    if not with_residual:
        return out
    rng = rng or np.random.default_rng(0)
    fi, phi, Z, bws = harmonic_tracker(x, sr, f0)
    ks = np.arange(1, len(Z) + 1)
    tone = 2 * np.real((Z * np.exp(1j * ks[:, None] * phi)).sum(0))
    resid = tone_residual(x, tone, sr, fi, bws, rng, refill)
    # relative phase psi_k = arg Z_k - k arg Z_1, circular mean weighted by |Z_k|
    rel = Z * np.exp(-1j * ks[:, None] * np.angle(Z[0]))
    m = (rel / (np.abs(rel) + 1e-20) * np.abs(Z)).sum(1) / (np.abs(Z).sum(1) + 1e-20)
    n_lv = len(levels)
    env = np.abs(Z[0])
    env = signal.sosfiltfilt(signal.butter(2, 25 / (sr / 2), output="sos"), env)  # ~20 ms level changes
    slow = lambda v: signal.sosfiltfilt(signal.butter(2, FAST_SPLIT_HZ / (sr / 2), output="sos"), v)
    raw_env = np.abs(Z[0])
    edge = int(0.02 * sr)  # filter edge effects
    inner = slice(edge, -edge)
    out.update({
        "track_cv": round(float(slow(fi)[inner].std() / fi.mean()), 4),
        "am_cv": round(float(env[inner].std() / (env[inner].mean() + 1e-12)), 4),
        # wingbeat-to-wingbeat wobble above FAST_SPLIT_HZ: gives the broad skirts around each harmonic
        "fm_fast_cv": round(float((fi - slow(fi))[inner].std() / fi.mean()), 5),
        "am_fast_cv": round(float((raw_env - slow(raw_env))[inner].std() / (raw_env[inner].mean() + 1e-12)), 4),
        "harmonic_energy_db": (band_energy_db(x, resid, sr, fi, bws) + [-60.0] * n_lv)[:n_lv],
        "harmonic_phase": (np.round(np.angle(m), 3).tolist() + [0.0] * n_lv)[:n_lv],
        "phase_R": (np.round(np.abs(m), 3).tolist() + [0.0] * n_lv)[:n_lv],
        "sharp_8hz": round(peak_sharpness(x, sr, f0), 4),
        "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 2),
        "dc": float(np.mean(x)), "n": len(x),
        "tnr_db": round(float(10 * np.log10(np.mean((x - resid) ** 2) / (np.mean(resid ** 2) + 1e-20))), 2),
        # tnr is always measured against the clean residual (best tone-energy estimate); the saved noise can be
        # the clean residual or the v0.2 notch residual (`noise.residual`).
        "_residual": resid if residual == "clean" else notch_residual(x, sr, fi, rng),
    })
    return out


def _calibrate_one(job):
    """Worker: analyse one clip, write its residual, return ("ok", (entry, residual check)) or ("drop", info)."""
    import warnings
    warnings.filterwarnings("ignore")
    cls, sex, lo, hi, path, dataset, orig_sr, refill, residual, noise_dir, seed = job
    sr = 8000
    x, _ = librosa.load(path, sr=sr, mono=True)
    rng = np.random.default_rng(seed)
    r = analyse_clip(x, sr, rng, with_residual=True, f0_range=(lo, hi), check_harmonics=True, refill=refill, residual=residual)
    if not r:
        return "drop", dict(file=path, f0_unrestricted=(analyse_clip(x, sr) or {}).get("f0"))
    resid = r.pop("_residual")
    rp = Path(noise_dir) / dataset / cls / (Path(path).stem + ".wav")
    rp.parent.mkdir(parents=True, exist_ok=True)
    sf.write(rp, resid.astype(np.float32), sr, subtype="FLOAT")
    ex_t, ex_x = harmonic_excess(resid, sr, r["f0"]), harmonic_excess(x, sr, r["f0"])
    ex_row = {**{f"h{k}": v for k, v in enumerate(ex_t, 1)}, **{f"x{k}": v for k, v in enumerate(ex_x, 1)},
              "off": float(np.median(harmonic_excess(resid, sr, r["f0"] * 1.37)[:3]))}
    r.update(file=str(path), dataset=dataset, orig_sr=orig_sr, residual=str(rp))
    return "ok", (r, ex_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default="configs/synth.yaml")
    ap.add_argument("--max-per-class", type=int, default=100000)
    ap.add_argument("--refill", choices=["borrow", "random"], default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sr = cfg["sample_rate"]
    refill = args.refill or cfg["noise"].get("refill", "borrow")
    rng = np.random.default_rng(cfg["seed"])
    noise_dir = Path(cfg["noise_bank_dir"])

    man = pd.read_csv(args.manifest)
    if "split" in man:
        man = man[man.split == "calib"]
    residual = cfg["noise"].get("residual", "clean")
    jobs = []
    for (cls, sex), g in man.groupby(["cls", "sex"]):
        g = g.iloc[rng.permutation(len(g))[: args.max_per_class]]
        lo, hi = plausible_range(cfg["priors"], cls, sex)
        for i, row in enumerate(g.itertuples()):
            jobs.append((cls, sex, lo, hi, row.path, row.dataset, int(row.orig_sr), refill, residual, str(noise_dir),
                         cfg["seed"] * 100003 + len(jobs)))
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(_calibrate_one, jobs, chunksize=8))
    bank, summary, excess, dropped = {}, [], [], []
    for (cls, sex, lo, hi, *_), res in zip(jobs, results):
        kind, payload = res
        if kind == "drop":
            dropped.append(dict(cls=cls, sex=sex, **payload))
            continue
        r, ex_row = payload
        bank.setdefault(cls, {}).setdefault(sex, []).append(r)
        excess.append(dict(cls=cls, **ex_row))
    for cls, by_sex in bank.items():
        for sex, entries in by_sex.items():
            lo, hi = plausible_range(cfg["priors"], cls, sex)
            col = lambda k: np.array([e[k] for e in entries])
            summary.append(dict(cls=cls, sex=sex, f0_range=f"{lo:.0f}-{hi:.0f}", kept=len(entries),
                                dropped=sum(d["cls"] == cls and d["sex"] == sex for d in dropped),
                                f0_median=np.median(col("f0")), f0_iqr=f"{np.percentile(col('f0'),25):.0f}-{np.percentile(col('f0'),75):.0f}",
                                track_cv=np.median(col("track_cv")), am_cv=np.median(col("am_cv")),
                                snr_db=np.median(col("snr_db")), tnr_db=np.median(col("tnr_db")), peak_dbfs=np.median(col("peak_dbfs"))))

    out = Path(cfg["calibration_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(bank, open(out, "w"), indent=1)
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    ex = pd.DataFrame(excess)
    print("\nleftover tone per harmonic, median dB above local floor (h = residual, x = original clip, off = noise-only reference):")
    print(ex.groupby("cls").median().round(1).to_string())
    ex.to_csv(out.with_name(out.stem + "_residual_check.csv"), index=False)
    pd.DataFrame(dropped).to_csv(out.with_name(out.stem + "_dropped.csv"), index=False)
    print(f"\ndropped {len(dropped)} clips (low SNR or inconsistent harmonics); see {out.stem}_dropped.csv")
    print(f"wrote {out} and residuals under {noise_dir} (refill={refill}, saved residual={residual})")


if __name__ == "__main__":
    main()
