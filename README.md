# Mosquito Radar — Stage 1: hear a mosquito, name the species

> **Confidential — patent application in preparation.** Keep this repository **private**. Do not make it public,
> fork it to a public account or upload its code, data or results to any public service: public disclosure before
> filing can destroy patentability.

Mosquito Radar is a low-cost home device that listens for mosquitoes and identifies them by the sound of their wings.
Stage 1 (this repository) is the acoustic model: from a 1-second microphone window it answers

1. **Is there a mosquito?** (detection, with false alarms per hour as the key cost)
2. **Which species?** The four main mosquitoes of Mumbai homes — *Aedes aegypti*, *Aedes albopictus* (dengue,
   chikungunya, Zika), *Anopheles stephensi* (urban malaria), *Culex quinquefasciatus* (filariasis) — or "other mosquito".
3. **Male or female?** Not trained yet (too few sex labels); the model has a hook for a sex output.

Later stages localise mosquitoes in a room with a microphone array. The target device is an **Android phone**
(TensorFlow Lite); the model is a few hundred KB.

## How it works

```
microphone ──► 8 kHz mono ──► 1 s windows (hop 0.5 s) ──► log-mel spectrogram (64 bands × 63 frames, 100–4000 Hz)
          ──► CNN ──► P(mosquito)  +  P(species: aegypti / albopictus / stephensi / quinquefasciatus / other)
```

A mosquito's wingbeat is a buzz at 300–1000 Hz plus evenly spaced harmonics; on the spectrogram it is a stack of
parallel lines. The CNN learns the spacing (wingbeat pitch), relative brightness and wobble of those lines. Inside the
model, **spectral whitening** removes the microphone's and room's colouring so the model cannot identify a species by
the lab it was recorded in.

**Key ideas**

- **Physics-informed synthetic mosquitoes** (`src/synth/`). Harmonic flight tones with measured pitch, harmonic levels,
  pitch drift and amplitude modulation, a flight path with distance attenuation and Doppler, and a randomised
  microphone chain. Every parameter is calibrated on real **training** recordings, and a discriminator checks realism
  against held-out real clips. The tones are mixed onto random backgrounds (other labs' recordings, household noise) so
  that background never gives away the species.
- **Splits by recording session, not by clip.** Whole labs and countries are held out, so the test measures how well
  the model works somewhere it has never heard.
- **Training against shortcuts**: per-class *and* per-site balanced weights, pitch-shift augmentation (±15 %), and
  SpecAugment-style masking.

## Results (real recordings only; synthetic data is used for training only)

Final M4 model: `configs/m4_final.yaml`, CNN with 316k parameters (~310 KB int8).

| Detection, ROC-AUC | Held-out labs | Thailand (unseen site) | UFRGS phones | Kenya (unseen site) |
|---|---|---|---|---|
| Final model | 0.985 | 0.980 | 0.850 | 0.555 |

- At threshold 0.9: 88 % of mosquito windows detected in held-out labs, with 1.4 false alarms per hour of their
  background.
- In unseen household noise (M5, DEMAND living room / office / café / street / car), detection is within ~5 points of
  clean down to 10 dB SNR, with 0 false alarms per hour.

| Species (test recordings) | Ae. aegypti | Ae. albopictus | An. stephensi | Cx. quinquefasciatus | Macro-F1 |
|---|---|---|---|---|---|
| Recording-level recall | 56 % | 56 % | 14 % | 91 % | 0.50 |

**Known limitations** (details and every experiment in [`results/experiments.md`](results/experiments.md)):

- **Lab-to-lab pitch differences.** Wingbeat pitch changes with temperature and rearing conditions as much as it does
  between species. Most species were recorded in only 1–3 labs, so species accuracy on new labs is limited.
  An. stephensi and Ae. albopictus have only ~1 minute of independent test audio each.
- **Pitched household sounds.** Cats, crying babies, sirens and alarms trigger false alarms (M5, ESC-50). Fixes are
  planned: hard-negative training, a YAMNet veto on Android, and longer alarm confirmation.
- **Very faint recordings.** In the Kenya field recordings (most windows have no measurable wingbeat tone), detection
  is near chance.
- **Sex.** Not yet classified.

## Repository layout

```
configs/            all settings (YAML): splits, features, synthesizer, training runs
src/
  download.py       dataset download with resume + MD5 check (HumBugDB, Abuzz, DEMAND)
  data/             manifests, labelled segments, session splits, calibration windows
  synth/            physics-informed synthesizer: calibrate → assign sex → generate → validate / tune
  features.py       log-mel windows for real splits and synthetic sets
  models.py         CNN (and optional ResNet) with spectral whitening and band normalisation
  train.py          shared training utilities (weights, augmentation incl. pitch shift, schedules)
  run_m3.py         M3 runner: detector only
  run_m4.py         M4 runner: detector + species (main model) — logs to terminal and <run>/run.log
  evaluate.py       detection metrics, false alarms / hour, operating points, figures
  evaluate_species.py, combine_eval.py   species metrics; 4-class / 5-class combination study
  eval_noise.py     M5: accuracy vs SNR and false alarms on unseen noise
data/               datasets and generated data (NOT in git; see below)
  noise/custom/     put your own household recordings here (test only; see its README)
results/
  experiments.md    dated log of every experiment, decision and result
  m4_runs/<run>/    run.log, config.yaml, metrics.json, history.csv, figures (models are not in git)
notebooks/
```

## Setup

Tested on macOS (Apple M4, CPU), Python 3.11.

```bash
conda create -n mosquito python=3.11
conda activate mosquito
pip install "tensorflow==2.21.*" librosa==0.11.* numpy pandas scipy scikit-learn soundfile pyyaml matplotlib optuna
conda install -c conda-forge ffmpeg      # decoding of compressed audio
```

Run every command from the repository root as a module (`python -m src....`).

## Data

Nothing in `data/` is version-controlled. Datasets and their licences:

| Dataset | Used for | Licence | How to get it |
|---|---|---|---|
| HumBugDB (Zenodo 4904800) | training, validation, test | CC BY 4.0 | `python -m src.download humbugdb` |
| Abuzz (Zenodo 4964774) | training, test | CC0 | `python -m src.download abuzz` |
| Funchal (figshare 11902125) | synthesizer realism checks | CC BY 4.0 | download, unpack to `data/raw/funchal/` |
| UFRGS Aedes smartphone sample | external test only | Apache-2.0 | clone the ComputerNetworks-UFRGS repository into `data/raw/ufrgs/repo/` |
| DEMAND (Zenodo 1227121) | household noise: 12 environments train, 5 test | CC BY 4.0 | `python -m src.download demand` |
| ESC-50 | noise **test only** | CC BY-NC 3.0 | unzip to `data/noise/public/esc50/ESC-50-master/` |
| MUSAN (OpenSLR 17) | noise test (optional) | per-file CC / public domain | extract to `data/noise/public/musan/musan/` |

ESC-50 is non-commercial. It is used only to measure, never to train.

## Reproducing the pipeline

Each step's exact history is in `results/experiments.md` (M1–M5).

```bash
# 1. data: manifests, labelled segments, session-level splits
python -m src.data.manifest
python -m src.data.manifest_funchal
python -m src.data.segments
python -m src.data.splits                      # configs/splits.yaml

# 2. features for the real splits
python -m src.features

# 3. synthetic training data (calibrated on TRAINING sessions only) -- see experiments.md, M1i, for calibration
python -m src.synth.generate_v2                                          # 4 Mumbai species + noise-only clips
python -m src.synth.generate_v2 --config configs/synth_v2_other.yaml     # 16 other species
python -m src.features --skip-real --synthetic data/synthetic/manifest_clips_v2.csv       --synthetic-name synth2_train
python -m src.features --skip-real --synthetic data/synthetic/manifest_clips_v2_other.csv --synthetic-name synth2other_train

# 4. train + evaluate the main model (~1.5-2 h on CPU; caffeinate keeps a Mac awake)
caffeinate -i python -m src.run_m4 --config configs/m4_final.yaml
python -m src.run_m4 --config configs/m4_final.yaml --quick              # 2-minute pipeline check

# 5. noise robustness (M5)
python -m src.eval_noise --run results/m4_runs/<run>
```

Every run writes to its own timestamped folder in `results/m4_runs/`, so nothing is overwritten. Seeds are fixed in the
configs.

## Milestones

| | Milestone | Status |
|---|---|---|
| M1 | Data exploration, plan, synthesizer | done |
| M2 | Session splits, log-mel features | done |
| M3 | Detector baseline | done |
| M4 | Detector + species model | done (`m4_final`); ResNet comparison prepared (`configs/m4_resnet.yaml`) |
| M5 | Accuracy vs noise, false alarms | first results; fixes for pitched household sounds pending |
| M6 | TensorFlow Lite export for Android: size, accuracy drop, latency | to do |
| M7 | `predict.py clip.wav` → "Aedes aegypti — 94 % confidence", final documentation | to do |

## Licence

Proprietary — all rights reserved. Third-party datasets keep their own licences (table above).
