# Mosquito Radar — Experiment Log

Dated record of every experiment's settings and metrics. Newest entries at the bottom.
Confidential — invention development record. Do not publish.

---

## 2026-09-28 — M1: Data acquisition and metadata inspection

**Source.** HumBugDB v0.0.1, Zenodo record 4904800 (DOI 10.5281/zenodo.4904800), licence CC-BY-4.0.
Metadata `neurips_2021_zenodo_0_0_1.csv` (md5 b93f185c437fd07eb29185e3539c7fc0, verified).
Audio: `humbugdb_neurips_2021_{1..4}.zip` (~4.05 GB total). Reference code: github.com/HumBug-Mosquito/HumBugDB.

**Totals (from metadata, 9,295 clips).**

| sound_type | clips | hours |
|---|---|---|
| mosquito | 6,795 | 18.32 |
| background | 1,900 | 14.63 |
| audio (non-mosquito events, Oxford) | 600 | 0.51 |

**Sample rates.** 44.1 kHz (Telinga/Tascam/Olympus field recorders): 16.2 h mosquito, 3.8 h background.
8 kHz (phones: Alcatel 4009X/4015X, itel A16): 2.1 h mosquito, 10.9 h background.

**Sex labels.** Female 12.97 h, Male 0.21 h (50 clips), unlabelled 5.13 h. Sex classification is **not feasible** with this data.

**Key finding: species are confounded with site/device.** Each species except those from Ifakara (Tanzania)
comes from a single site and device, e.g. *Cx. quinquefasciatus* = Oxford lab only, *An. gambiae s.s.* = Kenya
(USAMRU-K, Alcatel phone) only, *An. dirus/harrisoni/maculatus* = Thailand only. Ifakara has 10+ species recorded
with the same Tascam/Telinga equipment and 41 of 54 sessions contain more than one species — the only
subset where species can be learned without the model shortcutting via site or device.

**Ifakara species (minutes / sessions).** An. arabiensis 315.6/52, An. funestus (s.s.+s.l.) 149.3/36,
Cx. pipiens complex 136.0/36, An. squamosus 34.9/13, An. maculipalpis 33.5/6, Ma. uniformis 27.6/14,
Ae. aegypti 22.0/4, An. coustani 19.0/12, Ma. africanus 13.1/8. Imbalance ~24:1.

**Session key.** `place | device | date` → 80 sessions; no parent recording spans two sessions.

Script: `src/inspect_metadata.py`.

---

## 2026-09-28 — M1b: Additional datasets surveyed and acquired

Motivation: HumBugDB has little or no usable data for the Mumbai-relevant species (Ae. aegypti 22 min, An. stephensi 1.2 min, Cx. quinquefasciatus Oxford-lab only).

| Dataset | Source | Licence | Content | Decision |
|---|---|---|---|---|
| Abuzz (Mukundarajan et al., eLife 2017) | Zenodo 4964774 | CC0 | 1,285 phone recordings, 20 species incl. Ae. aegypti, Ae. albopictus, An. stephensi, Cx. quinquefasciatus, Cx. pipiens, An. gambiae, An. arabiensis; mixed formats (WAV/MP4/M4A/AMR); ~1.2 GB RAR | Add (training + cross-dataset test). Check for duplicates with HumBugDB "CDC insect cultures" clips. |
| Funchal bioacoustic set (Scientific Data 2020) | figshare 11902125 | CC BY 4.0 | Ae. aegypti (8/44.1/48 kHz, ~1.9k files), Cx. quinquefasciatus/pipiens (~1k files), Culiseta (73 files) | Add. |
| UFRGS Aedes smartphone (Paim et al., arXiv 2306.10091) | GitHub ComputerNetworks-UFRGS | Apache-2.0 | Public sample only: 5 female, 5 male Ae. aegypti, 5 non-Aedes clips, 8 kHz, ~25–37 s each | External sanity test only. Sex is confounded with recording date (all females 2021-02-13, all males 2021-02-15). |
| Kaggle Wingbeats (Fanioudakis & Potamitis 2018) | Kaggle | — | 279k optical-sensor (infrared) wingbeat recordings, 6 species | Skip: optical sensor, not a microphone. |

---

## 2026-09-28 — M1c: Physics-informed synthetic flight-tone generator (v0.1)

**Idea.** Generate additional training audio from a physical model of mosquito flight tone whose parameters are
*measured on real recordings* (not hand-set), rather than remixing real clips (cf. arXiv 2512.12365, which
chunks and overlays real Abuzz recordings with white noise and has no species/sex-specific acoustic model).

**Model** (`src/synth/wingbeat.py`, `configs/synth.yaml`):
- Source: harmonic series of f0(t); f0 drawn from an empirical bank of real clips (per class/sex) x per-clip
  jitter; within-flight f0 drift = Ornstein-Uhlenbeck process; per-harmonic level jitter; overall
  "directivity" level modulation (flight tone is directional — Arthur et al. 2014, PMC3985972).
- Flight: random 3-D trajectory around the mic, amplitude ~ 1/r^n (n 1.0–1.3), Doppler from radial speed.
  Trajectory is logged so the same model can drive a multi-mic array for Stage 2 localisation.
- Recording noise: mixed at the tone SNR measured on the real source clip (+-3 dB); noise = real background
  (when available) else white noise shaped to the source clip's measured noise-floor profile.
- Channel: random high/low-pass, 0–3 peaking EQs, self-noise, level, occasional clipping.
- Males: if a class has no real male clips, f0 = female f0 x U(1.29, 1.41) (literature M/F ratio for
  Ae. aegypti, Ae. albopictus, Cx. quinquefasciatus); tagged `source=prior` in the manifest.

**Literature priors used for sanity-checking** (mean ± SD, Hz): Ae. aegypti F 511±46 / M 711±78 (Arthur 2014, ~23 °C);
Ae. albopictus F 499±18 / M 704±50; Cx. quinquefasciatus F 415±18 / M 534±34 (PMC12049413, 27 °C).
Note overlap: Ae. aegypti vs Ae. albopictus females differ by only ~12 Hz, and blood-feeding/age shift f0 by
100+ Hz — f0 alone cannot separate species.

**Calibration on Funchal 8–48 kHz clips** (`src/synth/calibrate.py`): Ae. aegypti 399/400 clips kept, f0 median 552.5 Hz
(IQR 507–608); Cx. quinquefasciatus 385/400, f0 452.5 Hz (IQR 433–476). Consistent with literature ordering.

**Validation** (`src/synth/validate.py`, 200 real vs 200 synthetic, 0.3 s crops):

| Iteration | Change | f0 round-trip (real → synth) | Env. CV real/synth | f0 peak width real/synth | Real-vs-synth AUC |
|---|---|---|---|---|---|
| v0.0 | clean tone + channel | 552→560 / 452→458 | — | — | 1.00 |
| v0.0b | + SNR-matched coloured noise | 552→544 / 452→454 | 0.31 / 0.12 | 30 / 13 Hz | 1.00 |
| v0.1 | + measured noise profile, fast f0 drift, directivity modulation | 552→571 / 452→455 | 0.31 / 0.34 | 30 / 32 Hz | 1.00 |

**Status: synthetic clips are still trivially distinguishable from real (AUC 1.0).** f0 distribution and temporal
modulation now match; the remaining tell is the recording background — real Funchal clips carry low-frequency
transients (150–320 Hz mel bands ~20 dB higher than synthetic) that a stationary noise model does not reproduce.
Next: use real HumBugDB background recordings as the noise bed, recalibrate on all datasets, and — decisively —
measure whether adding synthetic data improves accuracy on REAL held-out test sets (synthetic never used for val/test).

---

## 2026-09-28 — M1d: Synthesizer v0.2 — realism fixes from external review

An external review measured that v0.1 was trivially separable even on fair features (loudness-matched,
<150 Hz removed): AUC 0.93 Aedes / 0.99 Culex, and proposed fixes in priority order. Implemented:

1. **Noise from the Funchal clips themselves.** Calibration now tracks f0 frame-by-frame and notches the
   harmonic comb (±(32 Hz + 1 %·k·f0) per harmonic) out of each real clip, refilling notched STFT bins
   with neighbouring-noise magnitude and random phase. The residual (median harmonic suppression 19.5 dB)
   is that recording's own background, including rumble, DC and handling noise. Residuals are pooled by
   (dataset, original sample rate, species batch). HumBugDB backgrounds are reserved for robustness (M5).
2. **Per-source device chain instead of randomisation.** Each synthetic clip takes its source clip's peak
   level (Aedes median −39.8 dBFS); native-8 kHz sources are quantised to 16-bit, resampled
   44.1/48 kHz sources stay float. Random EQ, filters and self-noise are moved to `augment_channel`
   (off for realism tests).
3. **Mixing level.** The peak-SNR estimator was found to be unstable at low SNR (the same audio reads 21.9
   or 32.9 dB depending on the f0 search range), so a gain search against it could not converge. Replaced
   by the tone-to-noise ENERGY ratio measured on the source clip (original − residual vs residual). Check:
   blind SNR percentiles 10/50/90 on synthetic Aedes 20.1/32.7/47.4 dB vs held-out real 20.1/31.7/46.7 dB.
4. **Tone model.** Measured per-clip relative harmonic phases (fixed waveform shape); harmonics measured and
   synthesised up to Nyquist (the 7th Aedes harmonic at ~3.85 kHz was previously dropped); pitch drift set
   per clip so the synthetic fundamental matches the source clip's peak sharpness (lookup table built by
   simulation, capped at CV 0.08); f0 scaling and harmonic-profile jitter set to 0.
   **Key bug found:** OU (Lorentzian) modulation and linear interpolation of control signals spread
   broadband skirts around every harmonic, capping a synthetic tone's own SNR at ~40 dB (real tone
   component: 57–77 dB). The mixer then made the tone 10–15 dB too loud relative to the noise. Replaced
   with Gaussian-spectrum modulation and a smoothed flight path → tone-only SNR 51–99 dB.
5. **Evaluation.** Realism check now scores held-out real clips only (Funchal split into contiguous index
   blocks per class/sample rate: 50 % calib / 25 % tune / 25 % test), no zero-padding, and prints per-feature
   AUCs. Real-vs-real baseline (calib vs tune, tune vs test): fair AUC 0.59–0.67 — the practical floor.

| Stage | Aedes fair AUC | Culex fair AUC |
|---|---|---|
| v0.1 (external measurement) | 0.93 | 0.99 |
| + own residual noise, device chain | 0.90 | 0.96 |
| + sharpness-matched drift, Nyquist harmonics | 0.87 | 0.97 |
| + energy-ratio mixing, residual pool per batch, smooth modulation | 0.73 | 0.76 |
| real vs real baseline | 0.59–0.67 | 0.60–0.64 |

**Tuning (fix 5).** `src/synth/tune.py`, coordinate descent over drift scale/time-constant, AM scale/time-constant,
per-harmonic jitter, harmonic tilt; score = mean fair AUC on the tune split. Kept: drift_scale 0.85, drift_tau 0.02 s,
am_scale 0.7, am_tau 0.06 s, harmonic_jitter 2.0 dB (tilt made it worse). Single-seed tune score 0.645.

**Honest evaluation (n = 200 per class, mean of seeds 7 and 11).** Seed-to-seed spread is ±0.05–0.10 AUC, so the
single-seed 0.645 was optimistic. The meaningful quantity is the gap above the real-vs-real floor:

| Split | Class | Synthetic vs real (fair AUC) | Real calib vs real split (floor) | Gap |
|---|---|---|---|---|
| tune | Ae. aegypti | 0.65 | 0.56 | +0.09 |
| tune | Cx. quinquefasciatus | 0.71 | 0.60 | +0.11 |
| **test** | Ae. aegypti | **0.77** | 0.66 | +0.11 |
| **test** | Cx. quinquefasciatus | **0.79** | 0.66 | +0.13 |

Remaining giveaways on test are diffuse (no single feature > 0.69): mean level around 0.9–1.3 kHz and 3.5–3.7 kHz.
Next: re-tune averaging 3 seeds (a single-seed search partly fits noise); then the decisive downstream test —
does adding synthetic clips improve accuracy on real held-out recordings.

**Multi-seed re-tune (seeds 7/11/13, n = 200, tune split).** Starting from the single-seed settings: mean fair AUC
0.696 (Aedes 0.669, Culex 0.723). No single-parameter change improved it by > 0.01 (best alternatives: drift_scale 1.0 →
0.706, harmonic_tilt 0.4 → 0.708; drift_tau 0.1 → 0.806). The single-seed choices are therefore a local optimum under
the less noisy score; config unchanged. (Run crashed once on a NaN in the SNR *diagnostic*, not in the audio; guarded.)

**Synthetic set v0.2 generated:** 12,000 clips (0.3 s each, ~60 min audio, 96 MB, 53 s to generate):
Ae. aegypti 3,000 + Cx. quinquefasciatus 3,000 (calibrated, sex unknown), plus 3,000 male clips per species
(`source=prior`: female parameters × U(1.29, 1.41) f0 ratio). Manifest: data/synthetic/manifest_synthetic.csv.

---

## 2026-09-28 — M1e: Abuzz acquired and inventoried

All 20 Abuzz archives verified (size + MD5), extracted to data/raw/abuzz/audio (2.4 GB). ffmpeg 7.1 installed into the
`mosquito` conda env (not system-wide) to decode AMR/M4A/MP4. Probe: data/processed/abuzz_probe.csv.

1,296 audio files (888 WAV, 168 M4A, 123 AMR, 40 MP4), 803.5 min total file duration, 23 species (incl. sub-strains),
all readable; sample rates 8 / 44.1 / 48 kHz. Folder structure = species / lab_phone, so site and device are known per file.

| Species | Minutes (file duration) | Labs / phones |
|---|---|---|
| An. stephensi | 85.9 | 8 phones (Nexus, Huawei, 2× Nokia, Samsung T209, Xperia, iPhone 4S/6) |
| Ae. aegypti | 28.9 | Auburn, CDC (3 phones), UC Davis LA + PR |
| Ae. albopictus | 16.1 | CDC, 3 phones |
| Cx. quinquefasciatus | 12.6 | San Mateo VCD, 2 phones |
| Cx. pipiens | 11.8 | San Mateo VCD, 2 phones |

Durations include non-flight portions; usable wingbeat time needs activity detection (next step).
Cx. quinquefasciatus and Cx. pipiens come from a single lab each, so species ⟂ site cannot be tested for Culex within Abuzz.

---

## 2026-09-28 — M1f: HumBugDB audio acquired and verified

All 4 zips verified (size + MD5 against Zenodo) and extracted to data/raw/humbugdb/audio (10 GB, 9,295 mono WAV named `<id>.wav`).
Audio vs metadata: 9,295/9,295 ids matched, 0 missing, 0 extra; 0 sample-rate mismatches; 0 duration mismatches.
Hours: mosquito 18.32, background 14.63, audio 0.51 — identical to the metadata. Probe: data/processed/humbugdb_probe.csv.

**All four datasets are now local:** HumBugDB (33.5 h), Abuzz (13.4 h file duration), Funchal (~14 min), UFRGS (~7 min sample).

---

## 2026-09-28 — M1g: Synthesizer v0.3 — second review's six fixes (negative result for fix 1)

Goal: implement a six-point review (clean the noise bank, per-species tuning with negative tilt, measured EQ,
bank cleanup, more realistic change over time, trustworthy tuning); expected fair AUC ~0.72–0.75 on test.
**Outcome: not reached.** Removing the leftover tone from the noise bank worked technically but made the
synthetic clips *easier* to spot; the default config keeps the v0.2 residuals. Details below.

**Fix 1 — tone removal (`src/synth/calibrate.py`).** Coarse pitch track at 4 ms hop (was 16 ms, 64 ms window →
32 ms), refined per sample from the phase drift of every demodulated harmonic (bandwidth 200 Hz); per-harmonic
complex envelopes (bandwidth 60 + 20·k Hz) → tone resynthesis → STFT subtraction: bins where the tone dominates
(|T|² > local noise) are refilled with this recording's noise, weaker bins are Wiener-subtracted. (A first version
refilled the whole ±band; those bands cover ~85 % of the Culex spectrum above 2 kHz, which doubled the noise's
level variation over time there and gave AUC 0.95–0.99.) Leftover tone per harmonic, median dB above the local
floor between harmonics (pure noise reads 1.4–3 dB on this measure):

| | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| Ae. aegypti original clip | 27.4 | 27.4 | 19.0 | 15.0 | 10.6 | 9.1 |
| Ae. aegypti v0.3 residual | −1.0 | 0.9 | −0.1 | 0.4 | 0.8 | 2.2 |
| Cx. quinquefasciatus original clip | 33.9 | 29.8 | 17.7 | 14.6 | 10.8 | 9.8 |
| Cx. quinquefasciatus v0.3 residual | 4.8 | 7.8 | 3.3 | 3.1 | 3.0 | 3.8 |
| Cx. quinquefasciatus v0.2 residual (notch, 25 clips) | 10.4 | 11.8 | 4.6 | 5.1 | 2.9 | — |

Aedes meets the ~3 dB target; Culex h1–h2 remain ~3–6 dB above noise (open). tnr_db rises by a median
+1.5 dB (Aedes) / +2.1 dB (Culex) on the same clips, as the review predicted.

**Fix 4 — bank cleanup.** f0 searched only inside the prior range (mean ± 3 SD; unsexed clips accept the female OR
male range, because Funchal Aedes are unsexed and many real clips sit at 650–800 Hz; the female-only range
would also drop 38 % of Culex, whose Funchal median 455 Hz is above the 27 °C literature mean). Octave /
subharmonic re-estimates are accepted (17 Aedes); clips whose comb is weak or whose restricted f0 is not an octave
of the free estimate (another tone dominates) are dropped. Kept: Aedes 885/937, Culex 421/441. The 200–370 Hz
Aedes and 650+ Hz Culex bank entries are gone.

**Fix 5 — change over time.** Flight: fly-by mode (straight pass with closest approach 2–30 cm before, during or
after the clip), turn rate. Room / cage: image-source early reflections of a box around the mic; every path reads
the source at t − r_i(t)/c, so Doppler and position-dependent colouring follow from geometry. Refill from the same
recording (`borrow`) vs random phase: no measurable difference (tune AUC within ±0.01). Also added fast
(> 20 Hz, wingbeat-to-wingbeat) pitch and level wobble, measured per clip (fast FM CV median 2.2 %, fast AM CV
20–27 %). **Bug fixed:** `smooth_noise` kept the DC bin; with long time constants the normalisation blew it up
(up to 58,000×) — found when the search tried am_tau 0.12 s.

**Fixes 2 + 6 — tuning (`src/synth/tune.py`).** Optuna TPE per species, all knobs continuous and free (drift,
AM, per-harmonic jitter, fast wobble, tilt −1.5…+1.5 dB/harmonic, SNR jitter, TNR offset, flight ranges, fly-by
share, room size / reflectivity, energy vs peak harmonic levels), 160 trials × 3 seeds, 4 workers. The winner is
re-scored against the current settings on 5 fresh seeds and kept only if it gains > 0.02.

| Search | Aedes current → best (fresh seeds) | Culex current → best | Verdict |
|---|---|---|---|
| v0.3 clean residuals | 0.877 → 0.813 | 0.980 → 0.873 | kept (in configs/synth_v03_clean.yaml) |
| v0.3 bank + v0.2 residuals, from v0.2 settings | 0.784 → 0.769 | 0.767 → 0.758 | rejected (< 0.02) |

The searches chose no room reflections, few fly-bys, near-zero tilt and a 2–3 dB louder tone for Culex.

**Fix 3 — measured EQ (`src/synth/fit_eq.py`).** Median real / median synthetic spectrum on tune, smoothed
150 Hz, ±6 dB, fitted iteratively on the tone (the noise is real) and, separately, on the whole mix. Both match the
median spectra (remaining gap 0.65–0.9 dB) but make the tune AUC worse (0.820/0.863 → 0.835/0.903 tone,
0.835/0.901 mix). Off.

**Test split (fair AUC, mean of seeds 7/11/13, n = 297 / 199; floor = real calib vs real test):**

| Config | Ae. aegypti | Cx. quinquefasciatus |
|---|---|---|
| real vs real floor | 0.697 | 0.671 |
| v0.2 as shipped (re-run with current code) | 0.834 | 0.784 |
| v0.3 clean residuals, tuned (configs/synth_v03_clean.yaml) | 0.864 | 0.891 |
| v0.3 tuned tone on v0.2 residuals | 0.832 | 0.897 |
| **default: v0.3 bank + v0.2 residuals + v0.2 settings (configs/synth.yaml)** | **0.844** (gap +0.15) | **0.810** (gap +0.14) |

Default chosen on the tune split; test used only for this table. Figure: results/figures/synth_vs_real.png.

**Why fix 1 backfired.** The v0.2 residuals still carried the real tone's broad skirts (energy ±50–150 Hz around
each harmonic) and its steady upper harmonics. With them removed, the synthetic tone must supply both, and it
cannot yet: clean-residual synthetic clips vary 30–50 % more over time above 2 kHz (mel std 3.3–3.65 vs 2.3–2.6 dB)
and have too-sharp fundamentals. Measured fast wobble at its measured size makes this worse, not better.

**Next.** Stop polishing the realism score: the gap to the floor is ~0.14, and fixes 2–6 did not move it by more
than seed noise. Two directions: (a) the decisive M4/M5 test — does adding synthetic clips improve accuracy on
real held-out recordings; (b) if realism must improve, model the tone from measured per-clip modulation
(replay/perturb the measured envelopes) instead of parametric drift. The synthetic training set was NOT
regenerated (data/synthetic/clips is still v0.2).

---

## 2026-09-28 — M1h: Independent check of synthesizer v0.3 (M1g)

Re-verified M1g against the data: calibration banks (885/937 Aedes, 421/441 Culex kept; 72 dropped, all octave-error
or implausible f0), residual-check table (identical to 0.1 dB), all 1,306 residual files present, no NaN. Test-split
headline reproduced exactly with the current code: Ae. aegypti 0.844 vs floor 0.697 (+0.147), Cx. quinquefasciatus
0.810 vs 0.671 (+0.139). Notes: (1) `src/synth/report.py` hard-codes v0.2 bars 0.80/0.82, which do not match the
same-code re-run of v0.2 (0.834/0.784) — on equal terms v0.3 default ≈ v0.2 (within seed noise). (2) My earlier v0.2
test numbers (0.77/0.79, M1d) used n = 200 and 2 seeds; n = 300 / 3 seeds gives 0.834/0.784. (3) The default
"v0.3 bank + v0.2 residuals" was built ad hoc (v0.3 fields incl. tnr_db from the clean residual; residual path from
v0.2); it is now reproducible via `noise.residual: notch` in src/synth/calibrate.py.

---

## 2026-09-28 — M1i: Final synthetic set v1.0 — Mumbai species

**Scope (user decision):** Ae. aegypti, Ae. albopictus, An. stephensi, Cx. quinquefasciatus; 3,000 female + 3,000 male
clips each; 1 s at 8 kHz. Config: `configs/synth_mumbai.yaml` (v0.3 default synthesizer and v0.2 tuned settings).

**Real calibration data** (`src/data/windows.py`): HumBugDB mosquito clips + Abuzz recordings cut into non-overlapping
1 s windows (12,734 windows; ≤ 40 per recording, raised for species with little audio). Funchal (0.3 s) not used.
Realism split: whole sessions for Ae. aegypti and Cx. quinquefasciatus (≥ 5 sessions); for Ae. albopictus and
An. stephensi (one dominant session each) contiguous time blocks within each recording (60/20/20). Bug found and fixed:
Abuzz reuses file names across phone folders, so 25 An. stephensi windows had overwritten each other — window names now
include a hash of the source path (uniqueness asserted).

**Calibration** (8 workers, 2.5 min): quality gate = SNR ≥ 8 dB, f0 in the search range, consistent harmonics.
Search ranges combine literature and data (Abuzz CDC Ae. albopictus females sit at 548–641 Hz vs 499 ± 18 Hz in the
27 °C literature; An. stephensi has no reliable female literature value — range from Abuzz labelled windows).
Kept: Ae. aegypti 396, Ae. albopictus 386, An. stephensi 352 (only ~25 % of stephensi windows contain a clear tone),
Cx. quinquefasciatus 1,067 (capped at 1,500 analysed per group). Noise = v0.2-style notch residuals.

**Sex** (`src/synth/assign_sex.py`): filename / metadata labels where present; otherwise from measured f0
(Ae. aegypti ≤ 640 F / ≥ 680 M; Ae. albopictus ≤ 650 / ≥ 700; Cx. quinquefasciatus ≤ 470 / ≥ 510 — Oxford Culex f0 is
clearly bimodal, ~400–450 and ~575–625 Hz, ratio ≈ 1.43 ≈ literature F/M); An. stephensi labels only (Abuzz F/M medians
only 1.13× apart; LSTMH London F 502 / M 652). Males: real entries used with probability min(1, n_real / 50), else a
female entry × U(1.29, 1.41) f0 (`source=prior`).

| Species | Female bank (label + f0) | Male bank | Male clips from real / prior |
|---|---|---|---|
| Ae. aegypti | 192 + 156 | 44 (f0) | 2,652 / 348 |
| Ae. albopictus | 316 + 68 | 1 (f0) | 46 / 2,954 |
| An. stephensi | 52 | 25 (label) | 1,536 / 1,464 |
| Cx. quinquefasciatus | 356 (f0) | 634 (f0) | 3,000 / 0 |

**Realism check, test split** (gated real windows, fair AUC, mean of 3 seeds):

| Species | Split | Synthetic vs real | Real vs real floor | Gap |
|---|---|---|---|---|
| Ae. aegypti | held-out session | 0.989 | 0.980 | +0.01 |
| Cx. quinquefasciatus | held-out sessions | 0.885 | 0.824 | +0.06 |
| An. stephensi | time block, same session | 0.719 | 0.509 | +0.21 |
| Ae. albopictus | time block, same session | 0.823 | 0.572 | +0.25 |

For Aedes/Culex the held-out sessions differ so much from the calibration sessions that even real recordings are
easy to tell apart, so the small gap says little. Within one session (stephensi, albopictus) synthetic clips are more
distinguishable than on Funchal (gap 0.21–0.25 vs 0.14); remaining tells are diffuse (≤ 0.69 per feature): level
variation above 2.8 kHz and crest factor. Settings were tuned on 0.3 s Funchal clips; not re-tuned for 1 s windows.

**Output:** data/synthetic/clips_mumbai_v1 — 24,000 clips, 653 MB; manifest data/synthetic/manifest_clips_mumbai_v1.csv
(per-clip f0, drift, SNR, flight distance, source window, noise file, `source`, `sex_source`). Sanity: all 8 kHz, 1 s,
finite, no clipping, no silence; measured f0 medians match targets (e.g. Culex F 398 vs 400 Hz).
Mistake fixed: the generator wrote every set's manifest to one file name and overwrote the v0.2 manifest; manifests are
now per output folder, and data/synthetic/manifest_clips.csv was rebuilt from folder names (per-clip metadata lost).
Synthetic clips are for training only; all validation / test data stay real.

---

## 2026-09-29 — M2: Labelled segments, session splits, synthetic v1.1, log-mel features

**Labels** (`src/data/segments.py` → data/processed/segments.csv, 14,613 segments). presence (mosquito / background or
other sound); species ∈ {Ae. aegypti, Ae. albopictus, An. stephensi, Cx. quinquefasciatus, other_mosquito} (user decision:
include "other mosquito"); Cx. pipiens and "pipiens complex" ignored for species (the complex contains quinquefasciatus);
genus; sex (labels only). HumBugDB clips = one segment each. Abuzz recordings: only seconds passing the calibration tone
gate count as mosquito (232 of 591 min, 39 %); the rest is excluded, not called background. Funchal not used (0.3 s).

**Splits** (`configs/splits.yaml`, `src/data/splits.py`): unit = session; forced assignments, the rest at random to
~70/15/15 % of minutes. test_A = Kenya (USAMRU-K), test_B = Thailand field site (unseen sites). Held-out test sites:
LSTMH London (An. stephensi, different lab + recorder), CDC Atlanta HumBugDB (same insectary as Abuzz-CDC but other phone
and date), San Mateo VCD (Cx. quinquefasciatus from a lab other than Oxford). A first random draw left 0.1 min of
Ae. aegypti in train → its training sessions are pinned. Checks: no session or source file in two splits.

| Species (min) | train | val | test | test_A | test_B | external |
|---|---|---|---|---|---|---|
| Ae. aegypti | 25.4 | 7.3 | 7.7 | – | – | 4.6 (UFRGS) |
| Ae. albopictus | 10.9 | 0 | 0.7 | – | – | – |
| An. stephensi | 13.8 | 0 | 1.2 | – | – | – |
| Cx. quinquefasciatus | 67.4 | 28.2 | 32.6 | – | – | – |
| other mosquito | 564 | 128 | 102 | 41 | 48 | – |

Limitation: Ae. albopictus and An. stephensi have no validation data and ~1 min of test data each (public data limit).

**Synthetic v1.1 (leakage fix).** v1.0 had been calibrated on realism-split windows, some of which lie in sessions that are
now in the model's val/test sets (e.g. 2,092 Culex windows). Recalibrated on the 7,906 training-session windows only and
regenerated 24,000 clips (data/synthetic/clips_mumbai_v1_1; v1.0 kept but superseded). Verified: all 24,000 clips take
their tone parameters AND their noise residual from training-session windows. An. stephensi real males now come only from
Abuzz (median 848 Hz vs females 604 Hz, ratio 1.40).

**Features** (`configs/features.yaml`, `src/features.py`): 8 kHz, 1 s windows, 50 % overlap, log-mel 64 bands 100–4000 Hz,
n_fft 512, hop 128 → 64 × 63, float16. Mosquito segments capped at 60 windows each in train/val (background uncapped — a
first run capped both and left only 9,647 background windows in train); 0.4–1 s segments wrap-padded (flagged).
norm.json = per-band mean/std of real training windows.

| Split | windows | mosquito | background |
|---|---|---|---|
| train | 100,997 | 78,766 | 22,231 |
| val | 23,288 | 20,245 | 3,043 |
| test | 22,866 | 17,703 | 5,163 |
| test_A (Kenya) | 66,959 | 3,988 | 62,971 |
| test_B (Thailand) | 31,101 | 16,970 | 14,131 |
| external (UFRGS) | 820 | 535 | 285 |
| synthetic (train only) | 24,000 | 24,000 | – |

**Visual check** (results/figures/m2_features.png): band profiles of real vs synthetic match closely for Culex and
An. stephensi; synthetic Aedes shows shallower valleys between harmonics and some strongly wobbling examples (10–25 % of clips
sit at the pitch-drift cap, mirroring real 1 s windows with broad peaks). A 6-setting check on 1 s windows (fair AUC vs
training-session calibration windows, 2 seeds) found no setting clearly better than the current one (mean 0.778; per-species
alternatives within ±0.05 ≈ seed noise), so settings were not changed.

**Noise sets for M5 (downloading):** DEMAND 16 kHz (CC BY 4.0 — correction: earlier notes said CC BY-SA), MUSAN (11 GB).
ESC-50 remains evaluation-only (CC BY-NC).

---

## 2026-09-29 — M3: Baseline detector (mosquito vs not), real data only

**Setup** (`configs/m3_detector.yaml`, `src/models.py`, `src/train.py`, `src/evaluate.py`): TensorFlow 2.21 / Keras, CPU.
CNN: 4 conv blocks (16/32/64/96 ch, 3×3, BN, ReLU, 2×2 max-pool) → global average pool → dropout 0.3 → sigmoid;
per-band normalisation built into the model. **79,409 parameters.** Real training windows only (100,997), balanced
class weights, augmentation: ±10 dB gain, SpecAugment (≤ 8 bands, ≤ 10 frames, p = 0.5). Adam 1e-3, batch 256, early
stopping on validation ROC-AUC (patience 5). Best epoch 5 of 10 (val AUC 0.988), 17.7 min. Seed 1, deterministic ops.
Two bugs of mine fixed before the recorded run: output piped through `head` (would kill training) and a wrong early-stopping
metric name (`val_presence_auc` vs Keras' `val_auc` for single-output models).

**Results** (results/m3/metrics.json, detection.png, operating_points.csv). ROC-AUC: val 0.989, test (held-out sessions /
labs) 0.995, **test_B Thailand 0.970, test_A Kenya 0.594**, external UFRGS 0.666. The max-F1-on-validation threshold (0.05) is
unusable for a device (190–840 false alarms/h) because validation is 87 % mosquito. With 2-of-3 smoothing:

| Threshold | test recall / FA per h | Thailand recall / FA per h | Kenya recall / FA per h |
|---|---|---|---|
| 0.5 | 0.92 / 1.4 | 0.85 / 53.5 | 0.02 / 22.1 |
| 0.9 | 0.83 / 0.0 | 0.80 / 32.6 | 0.01 / 0.5 |
| 0.98 | 0.71 / 0.0 | 0.76 / 26.5 | 0.00 / 0.0 |

Recall per species on test (threshold 0.05): Ae. aegypti 0.99, Ae. albopictus 0.95, An. stephensi 1.00, Cx. quinquefasciatus 0.99.

**Diagnosis of the Kenya failure.** Not a device confound (training has 1.7 h background vs 2.0 h mosquito from native 8 kHz
phones). The model scores Kenya background low (median 0.023) but also Kenya *mosquito* windows (0.035). An objective
tone test (calibration gate: SNR ≥ 8 dB, consistent harmonics, f0 300–1000 Hz) finds a clear wingbeat in only **10 %** of
Kenya mosquito windows — the same rate as Kenya background (11 %) — vs ~70 % of mosquito windows in train / test. Kenya
therefore mostly measures detection of very faint mosquitoes (or labels covering inaudible stretches); on the 10 % with a
tone the model is still weak (median score 0.16). Thailand background has a clear tone in 44 % of windows (field site —
probably unlabelled mosquitoes / insects), so its false-alarm rate is likely inflated.

**Flags for next milestones.** (1) Low-SNR detection is the weak point → M5 noise augmentation must include low SNRs.
(2) Report detection at fixed operating points (e.g. threshold 0.9 + 2-of-3), not max-F1. (3) Validation has only 0.42 h of
background, too little to set a false-alarm threshold reliably.

**M3 runner for manual runs** (`src/run_m3.py`, added 2026-09-29 at the user's request): one command runs data summary →
model build → training → evaluation on real recordings, with timestamped progress in the terminal and a copy in
`<out>/run.log`. Options: `--train-data real | real+synth | synth` (synth = synthetic mosquitoes + real background only;
validation and all tests always real), `--quick` (smoke test, meaningless numbers), `--epochs`, `--out`. Each run writes to
results/m3_runs/<time>_<train-data>/ so the recorded M3 run in results/m3/ is never overwritten. `src/evaluate.py` now exposes
`run(run_dir, log)` and also writes operating_points.csv (thresholds 0.5 / 0.9 / 0.98, 2-of-3 smoothing).

**M4 runner for manual runs** (`src/run_m4.py`, added 2026-09-29 at the user's request — M4 is the main model): same
layout as the M3 runner, one CNN with two outputs (presence + species), `--train-data real | real+synth | synth`, early
stopping on validation loss, then presence evaluation (all real splits, operating points) and species evaluation
(per-class precision / recall / F1 at window and recording level, macro-F1, genus accuracy, recall per recording site,
confusion matrices) on real val and test. Outputs in results/m4_runs/<time>_<train-data>/. My background real+synthetic
run (configs/m4_species_synth.yaml, epoch 6) was stopped so the user's own run gets the whole CPU; the finished real-only
run is in results/m4_real (trained before the runner existed; the Mac slept during it, 430 min wall time).

---

## 2026-09-29 — M4 first results: species classification is dominated by the recording site

Run by the user: `python -m src.run_m4 --train-data real+synth` (results/m4_runs/20260929_094352_real_plus_synth). Early stopping
on validation loss picked epoch 2 and stopped at 7; validation loss swings ±0.3 between epochs (1.02, 0.47, 0.88, 0.55, 1.11,
0.52, 0.61) while training loss falls steadily, so the stop is premature and the choice of epoch 2 is partly luck.

Detection (ROC-AUC): val 0.981, test 0.992, Kenya 0.604, Thailand 0.948, UFRGS 0.840 — like M3.

Species, TEST (real recordings), recall per class — real-only model (results/m4_real) vs real + synthetic:

| Training | Ae. aegypti | Ae. albopictus | An. stephensi | Cx. quinquefasciatus | other | macro-F1 |
|---|---|---|---|---|---|---|
| real only | 0.09 | 0.00 | 0.00 | 0.92 | 0.81 | 0.352 |
| real + synthetic | 0.10 | 0.00 | 0.00 | 0.92 | 0.82 | 0.348 |

By site: Cx. quinquefasciatus 0.998 at Oxford (lab seen in training) but 0.15 at San Mateo (unseen lab); Ae. albopictus
(CDC Atlanta, other phone) and An. stephensi (London) 0.00; Ae. aegypti at Ifakara 0.06, 81 % of its windows predicted
"other mosquito" (92 % of Ifakara training mosquitoes are other species). Only notable synthetic effect: Ae. aegypti at UC Davis LA
(unseen lab) 0.19 → 0.56.

**Interpretation.** The classifier learns the recording site / device, not the species — the confound flagged in M1 (most species
come from one or two labs). The synthetic clips carry background noise from the same lab as their species (noise pooled per
dataset × sample rate × species), so they reinforce the shortcut instead of breaking it.

**Proposed fixes (next):** (1) synthetic v2 for training: pair each synthetic tone with background noise from OTHER labs /
devices and household noise (DEMAND, MUSAN), so species and site are decoupled; (2) remove fixed device colouring from the input
(per-recording spectral mean normalisation); (3) balance training by site as well as by class; (4) steadier training: decaying
learning rate, patience 8–10, model selection on a smoother metric.

---

## 2026-09-29 — M4 v2: four fixes against the recording-site shortcut (user approved)

1. **Synthetic v2** (`configs/synth_v2.yaml`, `src/synth/generate_v2.py`, data/synthetic/clips_v2, 843 MB): 24,000 mosquito
   clips (4 species × F/M × 3,000; tone = v1.1 bank from training sessions) + **12,000 noise-only clips**. Noise bed = 50 %
   real training background (site chosen uniformly: Abuzz multiphone, Ifakara, Oxford — the only training sites with
   background audio; a first version sampled segments uniformly and over-used Oxford) and 50 % DEMAND household noise from 12
   TRAINING environments (5 held out for M5 tests: DLIVING, STRAFFIC, OOFFICE, TCAR, PCAFETER; custom Mumbai recordings test-only).
   30 % get a second, quieter noise layer. Tone-to-noise ratio 250–3900 Hz uniform −5…25 dB (includes faint mosquitoes). Device
   randomised on the whole mix (filters, EQ, self-noise, clipping, level −45…−3 dBFS, 16-bit for 50 %). The v1 notch residuals are
   NOT used (they carry traces of the real tone and only occur in mosquito clips). Check: noise source shares, second-layer and
   16-bit rates are identical for mosquito and noise-only clips (±1 %); 0 clips use a test household environment. MUSAN not yet
   used (download at 1.5 / 11 GB).
2. **Spectral whitening in the model** (`SpectralWhitening`, `src/models.py`): subtract each window's time-averaged band
   envelope smoothed over 15 mel bands; normalisation statistics computed after whitening. A synthetic 24 dB tilt + 7 dB gain
   changes the input by 19 dB before and 1.3 dB after whitening. Saves / reloads identically; converts to TFLite.
3. **Site-balanced weights** (`balanced_weights`, `src/train.py`): equal weight per class and, within a class, per recording
   site (capped at 10× the class median).
4. **Steadier training**: cosine learning-rate decay over 40 epochs, patience 10, epoch chosen by val_score = mean(val presence
   AUC, val species balanced accuracy) instead of the volatile validation loss.

`src/run_m4.py` now defaults to `configs/m4_v2.yaml` (old behaviour: `--config configs/m4_species_real.yaml`). Quick-mode checks
pass for `--train-data real` and `real+synth`.

---

## 2026-09-29 — M4 v2 results (user run results/m4_runs/20260929_103659_real_plus_synth)

Training: 18 epochs (patience 10), best epoch 8 by val_score (0.879); validation loss swung 0.48–2.20 while val presence AUC
stayed 0.97–0.99 — the species part of the loss on a validation set that is 78 % "other mosquito" while training is class- and
site-balanced; balanced accuracy 0.55–0.80. 32 min.

| Test (real) | before fixes (v1 run) | after fixes (v2) |
|---|---|---|
| Ae. aegypti recall (all / Ifakara / UC Davis LA) | 0.10 / 0.06 / 0.56 | **0.75 / 0.75 / 0.79** |
| Ae. albopictus recall (CDC Atlanta, 55 windows) | 0.00 | 0.15 |
| An. stephensi recall (London) | 0.00 | 0.00 |
| Cx. quinquefasciatus recall (Oxford / San Mateo) | 0.998 / 0.15 | 0.99 / 0.18 |
| other mosquito recall | 0.82 | 0.47 |
| species macro-F1, windows / recordings | 0.348 / 0.340 | 0.337 / 0.391 |
| genus accuracy on target species | 0.73 | **0.85** |
| presence ROC-AUC test / Kenya / Thailand / UFRGS | 0.992 / 0.604 / 0.948 / 0.840 | 0.989 / 0.564 / 0.965 / 0.784 |
| presence at threshold 0.9 (2-of-3): test recall, FA/h; Thailand FA/h | 0.81, 1.4; 66.7 | 0.60, 0.0; 18.9 |

**Where errors go (test):** An. stephensi from London → Cx. quinquefasciatus 83 %; Cx. quinquefasciatus from San Mateo → other 74 %;
"other mosquito" at Abuzz-CDC → Ae. aegypti 90 % (lab shortcut: Ae. aegypti was trained on the same CDC phones); other at Ifakara →
Ae. aegypti 26 % / Culex 18 %.

**Interpretation.** The fixes broke the Ifakara shortcut (Ae. aegypti 0.10 → 0.75, also on an unseen lab) at the cost of "other
mosquito" precision. The remaining failures are largely pitch shifts between labs: London An. stephensi fly at ~502 / 652 Hz (F / M)
vs ~604 / 848 Hz in the Abuzz training lab — inside the Culex range the model learned (~400 / 594 Hz); San Mateo Culex ~380 Hz vs
Oxford ~420 / 600 Hz. Wingbeat frequency depends on temperature, age and rearing conditions by as much as it differs between species,
so with one training lab per species the model cannot learn which part of the pitch is species and which is lab. Not fixable by
model tweaks alone. Candidate next steps: temperature-aware classification (device thermometer), pitch-shift augmentation to push
the model toward harmonic / modulation cues, genus-level output as the primary claim, recording-level decisions with an "uncertain"
option, and — most valuable — real recordings from more labs / from Mumbai homes.

---

## 2026-09-29 — Synthetic v2 for all species + optional 4-species head (user decisions)

**User decisions:** (1) test a species head with only the four Mumbai species (`configs/m4_mumbai4.yaml`); other species
stay in the data for the detector (≈ 80 % of all mosquito audio, many sites) but get no species target, and the report checks
whether low confidence can flag unknown species as "uncertain" (Mumbai also has e.g. Armigeres subalbatus, Cx. tritaeniorhynchus);
(2) since all species are needed, generate synthetic data for the other species too (previously only the four targets).

**Other-species synthetic set** (`configs/synth_other.yaml`, `configs/synth_v2_other.yaml`): 16 species with ≥ 4 min of audio in
TRAINING sessions (An. arabiensis, funestus, maculipalpis, squamosus, atroparvus, minimus, coustani, albimanus, gambiae,
quadrimaculatus, dirus, farauti; Ae. sierrensis; Cs. incidens; Ma. uniformis, africanus). 38,292 training windows
(`src/data/windows.py --train-only`), calibrated (f0 search 300–1000 Hz, ≤ 400 windows per species × sex label): 2,995 kept, e.g.
An. minimus F 597 / M 780 Hz, An. farauti F 603 / M 860 Hz. Generated 1,000 clips per species (sex not modelled: one pool per species)
+ 8,000 noise-only clips with the same noise pool / SNR / device randomisation as v2 (data/synthetic/clips_v2_other). Labels:
species = other_mosquito, species_raw = the real species. Checks: all 50 tone-source sessions are training sessions; no held-out
household environment; noise type 50 / 50 for mosquito and noise-only clips.

**Training configs:** `configs/m4_v2.yaml` (5 classes) and `configs/m4_mumbai4.yaml` (4 classes) both train on
synthetic_set = [synth2_train, synth2other_train] = 40,000 synthetic mosquito + 20,000 noise-only clips. Species list is now
configurable (`species_classes`, `set_species` in src/models.py). Quick-mode checks pass for both.

---

## 2026-09-29 — M4 v2 with synthetic data for all species: results (5 classes)

Run `results/m4_runs/20260929_142950_real_plus_synth` (`configs/m4_v2.yaml`, real + 60,000 synthetic clips, 53.5 min, best epoch 3).
Compared with the previous v2 run (`20260929_103659`, synthetic data for the 4 target species only):

| | previous | + other-species synth |
|---|---|---|
| presence ROC-AUC test / Kenya / Thailand / UFRGS | 0.989 / 0.564 / 0.965 / 0.784 | 0.985 / 0.578 / 0.960 / 0.829 |
| species macro-F1 val (windows / recordings) | 0.576 / 0.598 | 0.657 / 0.730 |
| species macro-F1 test (windows / recordings) | 0.337 / 0.391 | 0.363 / 0.397 |
| test recall Ae. aegypti Ifakara / UCD_LA | 0.75 / 0.79 | 0.66 / 0.65 |
| test recall Ae. albopictus / An. stephensi | 0.15 / 0 | 0 / 0 |
| test recall Culex Oxford / San Mateo | 0.99 / 0.18 | 0.92 / 0.02 |
| test recall other mosquito (Ifakara / CDC / London) | 0.56 / 0.07 / 0.13 | 0.74 / 0.19 / 0.40 |

**Interpretation.** Synthetic data for the other species mainly helps the "other mosquito" class (fewer foreign species mistaken for
a target) and the unseen UFRGS phones; the four target species did not improve (Aedes and San Mateo Culex slightly worse). The
cross-lab pitch shift remains the main limit. Validation species accuracy fluctuates strongly between epochs (few validation sessions).

The 4-species run (`configs/m4_mumbai4.yaml`, `results/m4_runs/20260929_152400_real_plus_synth`) stopped at epoch 10 of 40 at 15:40
(process ended, likely the Mac slept / restarted); it has no results and must be re-run.

---

## 2026-09-29 — M4 4-species head (Mumbai-4) vs 5-class head

Run `results/m4_runs/20260929_200213_real_plus_synth` (`configs/m4_mumbai4.yaml`, same data and synthetic sets as the 5-class
run, 40.2 min, best epoch 7). Species classes: Ae. aegypti, Ae. albopictus, An. stephensi, Cx. quinquefasciatus; other species
train the detector only.

| test (real, held-out labs) | 5 classes (`20260929_142950`) | 4 classes (`20260929_200213`) |
|---|---|---|
| Ae. aegypti recall (Ifakara / UCD_LA) | 0.66 / 0.65 | 0.96 / 0.95 |
| Ae. albopictus recall | 0.00 | 0.16 |
| An. stephensi recall | 0.00 | 0.00 |
| Cx. quinquefasciatus recall (Oxford / San Mateo) | 0.92 / 0.02 | 0.94 / 0.50 |
| genus accuracy, 4 target species | 0.77 | 0.88 |
| other species handled | "other" class, recall 0.65 | forced guess: 76 % Ae. aegypti, 22 % Culex; at confidence ≥ 0.9, 43 % "uncertain" |
| presence ROC-AUC test / Kenya / Thailand / UFRGS | 0.985 / 0.578 / 0.960 / 0.829 | 0.988 / 0.553 / 0.967 / 0.797 |

Confidence-threshold trade-off (4 classes, test): threshold 0.8 → 81 % of target windows answered, 96 % correct, 28 % of other-species
windows flagged "uncertain"; 0.9 → 72 % / 98 % / 43 %. (Macro-F1 values of the two runs are not comparable: different class sets.)

**Interpretation.** Removing the "other mosquito" class makes the four target species much easier to learn (aegypti and Culex
recall up by 0.3 and 0.5 on unseen labs), but the model has no way to say "not one of these": most unknown species are called
Ae. aegypti, which would raise false dengue-vector alerts. The confidence threshold only partly fixes that. An. stephensi (London,
pitch-shifted) is still never recognised. Candidate next steps: pitch-shift augmentation; a combined decision (4-class head for the
species name + the 5-class model's "other" probability, or an "other" class with reduced loss weight) to reject unknown species.


---

## 2026-09-30 — Target device changed to Android; combined species decision (no training)

**User decision:** the model will run on an **Android phone** instead of an ESP32-S3 / Raspberry Pi. Size / compute limits are
much looser (a phone runs models of several MB in milliseconds); export remains TensorFlow Lite; input from the phone microphone
is resampled to the existing 8 kHz pipeline.

**Combined decision** (`src/combine_eval.py`, output `results/m4_combined/`): 4-species model names the species, the 5-class
model's "other mosquito" probability vetoes it. Veto threshold chosen on validation recordings (0.50), reported on test.

| test, recording level | Ae. aegypti | Ae. albopictus | An. stephensi | Culex | other rejected | balanced | other called Ae. aegypti |
|---|---|---|---|---|---|---|---|
| 4-class alone | 78 % | 17 % | 0 % | 96 % | 0 % | 38 % | 70 % |
| 5-class alone | 58 % | 0 % | 0 % | 89 % | 78 % | 45 % | 18 % |
| product rule | 58 % | 10 % | 0 % | 88 % | 80 % | 47 % | 16 % |
| veto @ 0.50 | 64 % | 13 % | 0 % | 90 % | 75 % | 49 % | 19 % |

**Interpretation.** Combining gives only a small gain (balanced recall 45 → 49 %) and no "best of both": whenever the veto rejects
other species it also rejects real Ae. aegypti, because the 5-class model itself cannot tell many real aegypti from other species
across labs. The threshold is a dial (test windows: 0.1 → 89 % of other species rejected but only 54 % of targets answered; 0.9 →
23 % / 96 %). Conclusion: the limit is what the acoustic features separate across labs, not how the two outputs are combined;
merging the two into one three-head model would reproduce the same trade-off. Next: attack the cause (pitch-shift augmentation;
a larger model now that the target is a phone) and expose the veto threshold as a sensitivity setting.

---

## 2026-09-30 — M4 final run prepared (user runs it)

`configs/m4_final.yaml`, run with `caffeinate -i python -m src.run_m4 --config configs/m4_final.yaml`. Changes vs m4_v2:
(1) **pitch-shift augmentation** (`pitch_shifter` in src/train.py): 70 % of training windows shifted by a log-uniform ratio in
[1/1.15, 1.15], done on the log-mel window (output band at f reads the input at f / ratio, Slaney mel scale, linear interpolation).
Check against re-synthesised harmonic tones (f0 420–600 Hz, ratios 0.87–1.15): spectral-profile correlation with the truly
shifted tone 0.91–0.97 (unshifted −0.13 to −0.03); peak band matches in all cases. (2) **Larger CNN** for Android: channels
32/64/128/192, 316,454 parameters (~309 KB int8), ~2.3× slower to train. (3) 30 epochs (cosine), patience 8. (4) The report
adds the "P(other)" threshold table = the app's sensitivity setting. Quick-mode run passes. Note: model size and pitch shift change
together, so any gain cannot be attributed to one of them without a further ablation run.

---

## 2026-09-30 — M4 final run: results (bigger CNN + pitch shift, 5 classes)

Run `results/m4_runs/20260930_114325_real_plus_synth` (`configs/m4_final.yaml`, 316k parameters, 18 epochs / 107 min, best epoch 10).
Compared with the previous 5-class run `20260929_142950` (80k parameters, no pitch shift):

| | previous 5-class | final |
|---|---|---|
| presence ROC-AUC test / Kenya / Thailand / UFRGS | 0.985 / 0.578 / 0.960 / 0.829 | 0.985 / 0.555 / 0.980 / 0.850 |
| false alarms / h at threshold 0.9 (test / Kenya / Thailand) | 18.1 / 65.6 / 99.4 (recall test 0.90) | 1.4 / 46.1 / 40.8 (recall test 0.88) |
| species macro-F1 test, windows / recordings | 0.363 / 0.397 | 0.386 / 0.497 |
| test recording recall Ae. aegypti / albopictus / stephensi / Culex | 0.58 / 0 / 0 / 0.89 | 0.56 / 0.56 / 0.14 / 0.91 |
| test window recall San Mateo Culex / London stephensi / UCD_LA aegypti | 0.02 / 0 / 0.65 | 0.05 / 0.13 / 0.44 |

Sensitivity setting (test recordings, answer "other / uncertain" when P(other) ≥ threshold):

| | threshold | targets answered | correct when answered | other rejected | other called Ae. aegypti |
|---|---|---|---|---|---|
| previous | 0.3 | 66 % | 89 % | 88 % | 9 % |
| previous | 0.4 | 76 % | 88 % | 83 % | 14 % |
| final | 0.2 | 83 % | 87 % | 85 % | 6 % |
| final | 0.3 | 88 % | 86 % | 75 % | 11 % |

**Interpretation.** Clear improvement in the detector's false alarms (Thailand −59 %, test −92 % at 0.9) and in the
species trade-off (at the same rejection of unknown species, ~10 points more target recordings answered with fewer false
Ae. aegypti). Ae. albopictus and An. stephensi are recognised for the first time (weak evidence: 55 and 121 test windows, one
lab each). Unchanged: Kenya (faint recordings), San Mateo Culex. Caveats: validation species accuracy swings 0.56–0.78 between
epochs, so run-to-run (seed) variance is likely large; model size and pitch shift changed together; the threshold table is read
on TEST — the default sensitivity must be fixed on validation or Mumbai recordings, not chosen from this table.

---

## 2026-09-30 — M5: robustness to noise never heard in training

`src/eval_noise.py` on the final M4 model (`results/m4_runs/20260930_114325_real_plus_synth/m5_noise/`). Test-only noise:
DEMAND held-out environments (DLIVING, OOFFICE, PCAFETER, STRAFFIC, TCAR; 25 min) and ESC-50 (2,000 clips, 50 classes in
5 groups, CC BY-NC, downloaded for evaluation only). MUSAN: download incomplete (only music so far) — skipped; custom Mumbai
folder still empty. Mosquitoes: 797 real TEST windows (≤ 300 per class, ≤ 3 per segment). SNR = band energy 250–3900 Hz.

**A. Accuracy vs SNR** (threshold 0.9; no added noise: detected 88 %, species 64 %, genus 67 % on this window sample):
detection holds to within ~5 points down to 10 dB for every noise type; at 0 dB 52 % (DEMAND: steady broadband ambient) to
74 % (ESC-50 animals / domestic); at −10 dB 7–55 %. Species accuracy degrades earlier: 52–61 % at 10 dB, 34–51 % at 0 dB.

**B. False alarms on noise alone** (threshold 0.9, 2-of-3 smoothing): DEMAND held-out **0 / h**; ESC-50 **168–496 / h**
(animals worst). Worst classes: cat, crying baby, siren, rooster, hen, insects, dog, clock alarm, pouring water, car horn,
laughing — mostly harmonic / pitched sounds, which look like a wingbeat harmonic stack on a spectrogram. The model never saw
such sounds as negatives (training backgrounds: field / lab recordings + DEMAND ambient). ESC-50 is event-dense (every clip is a
5 s sound event), so rates are far above what a home would produce, but the weakness is real: a meowing cat or crying baby
will trigger "mosquito". ESC-50 "insects" (flying insects) may contain actual mosquitoes / flies.

**Problem flagged:** false alarms on pitched household sounds. Candidate fixes: (1) hard-negative training with pitched
non-mosquito sounds from a commercially usable source (FSD50K, CC0 / CC-BY per clip; MUSAN noise), keeping ESC-50 strictly as
the test set; (2) on Android, a pretrained general sound classifier (YAMNet, Apache-2.0) as a veto for speech / animals /
alarms; (3) longer alarm smoothing (a mosquito pass lasts seconds).

---

## 2026-10-01 — M4 ResNet prepared (user request: ResNet backbone with custom input layers)

`configs/m4_resnet.yaml`, `model.arch: resnet` in `src/models.py` (`resnet_backbone`). Same data, synthetic sets, augmentation
(incl. pitch shift), site balancing, epoch selection and evaluation as m4_final, so the comparison isolates the backbone. Model:
input 64 × 63 log-mel dB → spectral whitening → band normalisation (unchanged) → **adapter layers**: bilinear resize to 128 × 128
(a ResNet stem divides the input by 4, the whole network by 32; on 64 × 63 that merges ~4 mel bands before the first residual block
and ends on a 2 × 2 map) and a learnable 1 × 1 conv 1 → 3 channels (initialised to copy the window into all three) → **ImageNet-
pretrained ResNet50V2 without its classifier** (4 × 4 × 2048) → global average pool → dropout 0.3 → presence (sigmoid) + species
(softmax, 5 classes). 23,577,100 parameters (75× m4_final). Learning rate 3e-4 (fine-tuning) instead of 1e-3; everything else as m4_final.

Checks: Keras save / reload works; TFLite conversion with dynamic-range int8 quantisation works (24.1 MB; TFLite vs Keras outputs
within 0.01). Cost on the M4 Mac (CPU only, TensorFlow 2.21, no GPU plugin): 10.1 s per training batch of 256 at 128 × 128 ≈ 105 min
per epoch (m4_final: ~6 min), i.e. roughly 1–2 days for an 18–30 epoch run; at `input_size: [64, 64]` 3.1 s per batch ≈ 32 min per
epoch. Phone latency not yet measured (M6).
Quick-mode check (`--quick`, results/m4_runs/resnet_quickcheck): train → save → reload → presence and species evaluation all run;
~6 min per epoch on 8,000 windows (≈ 11 s per batch incl. validation), consistent with the estimate. Numbers not meaningful.
