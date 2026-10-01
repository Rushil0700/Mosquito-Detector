"""Physics-informed synthesis of a single mosquito's flight tone as heard by one microphone.

Model (parameters per clip, measured on a real clip by src/synth/calibrate.py):
  source   s(t) = sum_k g_k(t) * cos(k * Phi(t) + psi_k),  Phi' = 2*pi * f0(t)
           f0(t)  = measured f0 * (1 + slow drift + fast wingbeat-to-wingbeat wobble); slow drift from the
                    measured peak sharpness or fine-track CV, fast wobble from the measured fast FM CV
           g_k    = measured harmonic level * slow + fast level modulation (measured) * per-harmonic jitter
           psi_k  = measured relative harmonic phase (fixed per clip -> real waveform shape)
  flight   3-D wander or fly-by (approach / pass / leave) around the mic; amplitude ~ 1/r^n
  room     optional image-source reflections of a box around the mic; every path is read from the
           source at t - r_i(t)/c, so Doppler and position-dependent comb colouring come from geometry
  noise    real recording residual (tone subtracted, bands refilled), mixed to the measured tone-to-noise ratio
  device   per source sample rate: measured peak level, 16-bit quantisation for native 8 kHz clips.
           Random EQ / filters / self-noise (`augment_channel`) are for training-time augmentation only.

The trajectory is returned so the same model can later drive a multi-microphone array (Stage 2).
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class ToneParams:
    f0_hz: float
    harmonic_db: np.ndarray          # level of harmonics 1..K (0 dB = loudest)
    harmonic_phase: np.ndarray       # relative phase psi_k (rad), psi_1 = 0
    drift_cv: float = 0.01
    drift_tau_s: float = 0.05
    am_db: float = 1.0               # std of common level modulation (orientation / directivity)
    am_tau_s: float = 0.03
    harmonic_jitter_db: float = 0.5  # independent per-harmonic level modulation
    fm_fast_cv: float = 0.0          # fast (wingbeat-to-wingbeat) pitch wobble, fraction of f0
    am_fast_cv: float = 0.0          # fast level wobble, fraction of amplitude
    fast_bw_hz: float = 80.0         # bandwidth of the fast wobble (sets the width of the harmonic skirts)


def smooth_noise(n, sr, tau, sigma, rng):
    """Zero-mean random modulation with std `sigma` and a Gaussian spectrum (correlation time ~tau).

    Unlike an OU process (Lorentzian 1/f^2 spectrum), the Gaussian spectrum dies off quickly, so the
    modulation does not smear broadband skirts around every harmonic (real flight tones have steep skirts).
    """
    f = np.fft.rfftfreq(n, 1 / sr)
    spec = np.fft.rfft(rng.standard_normal(n)) * np.exp(-0.5 * (2 * np.pi * f * tau) ** 2)
    spec[0] = 0  # zero mean; otherwise a long tau leaves ~only DC and the normalisation below explodes it
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-12) * sigma


def _smooth(x, sr, tau):
    """Gaussian smoothing of a control signal (removes interpolation kinks)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, tau * sr, mode="nearest")


def ou_process(n, dt, tau, sigma, rng):
    """Stationary Ornstein-Uhlenbeck process with std `sigma` and time constant `tau`."""
    a = np.exp(-dt / tau)
    noise = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    x = np.empty(n)
    x[0] = rng.standard_normal() * sigma
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x


def _upsample(x, n):
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def flight_trajectory(n, sr, cfg, rng, ctrl_hz=100):
    """3-D flight path (m) relative to a mic at the origin, at `ctrl_hz` then upsampled.

    With probability `flyby_prob` the mosquito flies a near-straight pass whose closest approach
    (`flyby_miss_m`) happens somewhere in or near the clip, so it can approach, pass or leave; otherwise
    it wanders (correlated heading changes at constant speed) from `start_distance_m`.
    """
    m = max(2, int(n / sr * ctrl_hz))
    dt = 1.0 / ctrl_hz
    speed = rng.uniform(*cfg["speed_m_s"])
    turn = cfg.get("turn_rate", 2.0)
    if rng.random() < cfg.get("flyby_prob", 0.0):
        u = rng.standard_normal(3)
        u /= np.linalg.norm(u)
        vel = rng.standard_normal(3)
        vel -= vel.dot(u) * u                     # travel perpendicular to the closest-approach direction
        vel *= speed / np.linalg.norm(vel)
        t_c = rng.uniform(-0.5, 1.5) * m * dt     # closest approach before, during or after the clip
        pos = u * rng.uniform(cfg["min_distance_m"], cfg.get("flyby_miss_m", 0.1)) - vel * t_c
        turn *= 0.3
    else:
        d0 = rng.uniform(*cfg["start_distance_m"])
        direction = rng.standard_normal(3)
        pos = direction / np.linalg.norm(direction) * d0
        vel = rng.standard_normal(3)
        vel *= speed / np.linalg.norm(vel)
    path = np.empty((m, 3))
    for i in range(m):
        # Correlated heading changes (mosquitoes zig-zag) at constant speed.
        vel += rng.standard_normal(3) * speed * turn * np.sqrt(dt)
        vel *= speed / max(np.linalg.norm(vel), 1e-6)
        pos = pos + vel * dt
        r = np.linalg.norm(pos)
        if r < cfg["min_distance_m"]:  # bounce off the mic
            pos *= cfg["min_distance_m"] / r
            vel = -vel
        path[i] = pos
    path = np.stack([_smooth(_upsample(path[:, j], n), sr, 0.01) for j in range(3)], axis=1)
    return path


def propagation_paths(path, room, rng):
    """Direct path plus image-source reflections of a rectangular room / cage around the mic.

    Returns a list of (distance per sample, reflection gain). The mic is placed at a random point in the
    box and the mosquito is kept inside it; reflections change with the mosquito's position, which gives
    the level and colour ripple real recordings show. `room=None` or disabled -> direct path only.
    """
    r = np.linalg.norm(path, axis=1)
    if not room or not room.get("enabled"):
        return [(r, 1.0)]
    dims = np.array([rng.uniform(*room["dims_m"]) for _ in range(3)])
    margin = np.minimum(0.02, dims / 4)
    mic = np.array([rng.uniform(mg, d - mg) for mg, d in zip(margin, dims)])
    src = np.clip(mic + path, margin, dims - margin)
    beta = rng.uniform(*room["reflection_coef"], size=(3, 2))  # per wall
    out = [(np.linalg.norm(src - mic, axis=1), 1.0)]
    for order_vec in _image_orders(room.get("order", 1)):
        img, g = src.copy(), 1.0
        for ax, q in enumerate(order_vec):
            if q == 0:
                continue
            # q reflections along this axis: alternate walls at 0 and dims[ax]
            n_lo, n_hi = (abs(q) + (q < 0)) // 2, (abs(q) + (q > 0)) // 2
            img[:, ax] = _image_coord(src[:, ax], dims[ax], q)
            g *= beta[ax, 0] ** n_lo * beta[ax, 1] ** n_hi
        out.append((np.linalg.norm(img - mic, axis=1), g))
    return out


def _image_coord(x, L, q):
    """Coordinate of the q-th image along one axis with walls at 0 and L (|q| reflections; q > 0: first
    bounce off the wall at L, q < 0: off the wall at 0). Images sit at x + 2mL and -x + 2mL."""
    return x + q * L if q % 2 == 0 else -x + (q + 1) * L


def _image_orders(order):
    rng_ = range(-order, order + 1)
    return [(a, b, c) for a in rng_ for b in rng_ for c in rng_
            if 0 < abs(a) + abs(b) + abs(c) <= order]


def synth_tone(params: ToneParams, n, sr, flight_cfg, max_harmonic_hz, rng, room_cfg=None):
    """Return (clean waveform at the mic, trajectory, per-sample f0).

    The source is synthesised at emission time from smooth control signals (phase, harmonic levels);
    each propagation path (direct + reflections) reads them at t - distance/c, so Doppler, spreading
    and room colouring follow from the geometry without fractional-delay filtering of the waveform.
    """
    path = flight_trajectory(n, sr, flight_cfg, rng)
    paths = propagation_paths(path, room_cfg, rng)
    c = flight_cfg["speed_of_sound_m_s"]
    pad = int(np.ceil(max(d.max() for d, _ in paths) / c * sr)) + 2
    N = n + pad
    tau_fast = 1 / (2 * np.pi * params.fast_bw_hz)
    f0 = params.f0_hz * np.maximum(1 + smooth_noise(N, sr, params.drift_tau_s, params.drift_cv, rng)
                                   + smooth_noise(N, sr, tau_fast, params.fm_fast_cv, rng), 0.5)
    phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(f0) / sr
    common = (smooth_noise(N, sr, params.am_tau_s, params.am_db, rng)
              + 20 * np.log10(np.maximum(1 + smooth_noise(N, sr, tau_fast, params.am_fast_cv, rng), 0.05)))
    harm = []
    for k, (level_db, psi) in enumerate(zip(params.harmonic_db, params.harmonic_phase), start=1):
        if k * params.f0_hz > max_harmonic_hz:
            break
        g = 10 ** ((level_db + common + smooth_noise(N, sr, params.am_tau_s, params.harmonic_jitter_db, rng)) / 20)
        harm.append((k, g, psi))

    t = np.arange(n) + pad
    expo = rng.uniform(*flight_cfg["spreading_exponent"])
    out = np.zeros(n)
    grid = np.arange(N)
    for dist, gain in paths:
        te = t - dist / c * sr                                # emission time (samples) of what arrives now
        ph = np.interp(te, grid, phase)
        amp = gain * (flight_cfg["min_distance_m"] / np.maximum(dist, flight_cfg["min_distance_m"])) ** expo
        for k, g, psi in harm:
            out += amp * np.interp(te, grid, g) * np.cos(k * ph + psi)
    return out, path, f0[pad:]


def mix_at_tnr(tone, noise, tnr_db):
    """Scale the tone to a tone-to-noise ENERGY ratio (dB); the real residual keeps its own level."""
    g = np.sqrt(np.mean(noise ** 2) * 10 ** (tnr_db / 10) / (np.mean(tone ** 2) + 1e-20))
    return g * tone + noise


def mix_at_snr(tone, noise, target_snr_db, measure, tol_db=0.5, span_db=60.0, coarse_db=3.0, refine=12):
    """Scale the tone so that measure(tone + noise) is as close as possible to the target.

    The noise (a real recording residual) keeps its own level; only the tone gain is searched: a coarse
    scan over +-span/2 dB, then bisection inside the bracket that crosses the target. The measure is
    not perfectly monotonic (leftover residual peaks dominate at low gain; the tone's own skirts cap it
    at high gain), so the closest value seen is returned. `measure` must match the calibration estimator.
    Returns (mix, achieved SNR).
    """
    g0 = np.std(noise) / (np.std(tone) + 1e-12) * 10 ** (target_snr_db / 20)
    m = lambda d: measure(g0 * 10 ** (d / 20) * tone + noise)
    grid = np.arange(-span_db / 2, span_db / 2 + 1e-9, coarse_db)
    vals = np.array([m(d) for d in grid])
    best = int(np.nanargmin(np.abs(vals - target_snr_db)))
    best_d, best_v = grid[best], vals[best]
    above = np.flatnonzero(vals >= target_snr_db)
    if len(above) and above[0] > 0:
        lo, hi = grid[above[0] - 1], grid[above[0]]
        for _ in range(refine):
            if abs(best_v - target_snr_db) <= tol_db:
                break
            mid = (lo + hi) / 2
            v = m(mid)
            if abs(v - target_snr_db) < abs(best_v - target_snr_db):
                best_d, best_v = mid, v
            lo, hi = (mid, hi) if v < target_snr_db else (lo, mid)
    return g0 * 10 ** (best_d / 20) * tone + noise, best_v


def device_output(x, peak_dbfs, orig_sr):
    """Match the real device chain: measured peak level; 16-bit grid for native 8 kHz sources.

    Clips originally recorded at 44.1/48 kHz were resampled to 8 kHz as float, so they keep
    fine amplitude resolution; native 8 kHz clips are 16-bit at 8 kHz and at low levels use
    only a few hundred distinct values, which we reproduce.
    """
    x = x / (np.max(np.abs(x)) + 1e-12) * 10 ** (peak_dbfs / 20)
    if int(orig_sr) == 8000:
        x = np.round(x * 32767) / 32767
    return np.clip(x, -1, 1).astype(np.float32)


def augment_channel(x, sr, cfg, rng):
    """Training-time device randomisation (off for realism tests): filters, EQ, self-noise, clipping."""
    nyq = sr / 2
    x = signal.sosfilt(signal.butter(2, rng.uniform(*cfg["highpass_hz"]) / nyq, "highpass", output="sos"), x)
    x = signal.sosfilt(signal.butter(4, min(rng.uniform(*cfg["lowpass_hz"]), 0.998 * nyq) / nyq, "lowpass", output="sos"), x)
    for _ in range(rng.integers(cfg["n_peaking_eq"][0], cfg["n_peaking_eq"][1] + 1)):
        x = peaking_eq(x, sr, rng.uniform(150, 3500), rng.uniform(*cfg["peaking_gain_db"]), rng.uniform(0.7, 3.0))
    x = x + rng.standard_normal(len(x)) * np.max(np.abs(x)) * 10 ** (rng.uniform(*cfg["self_noise_db"]) / 20)
    if rng.random() < cfg["clip_probability"]:
        x = np.clip(x / np.max(np.abs(x)) * rng.uniform(1.5, 4), -1, 1)
    return x


def peaking_eq(x, sr, f0, gain_db, q):
    """RBJ audio-EQ-cookbook peaking biquad."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return signal.lfilter(b, den, x)
