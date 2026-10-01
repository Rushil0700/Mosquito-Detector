"""Run M4 (the main model: mosquito detection + species) end to end with readable logs in the terminal
(and a copy in <out>/run.log).

    conda activate mosquito
    cd "/Users/rushil/Documents/Mosquito Detector"
    caffeinate -i python -m src.run_m4 --config configs/m4_final.yaml   # FINAL M4 run (Android: bigger CNN + pitch shift)
    caffeinate -i python -m src.run_m4 --train-data real+synth   # earlier main run (configs/m4_v2.yaml: synthetic v2 + all fixes)
    python -m src.run_m4 --train-data real                        # comparison: real training data only
    python -m src.run_m4 --train-data synth                       # synthetic mosquitoes + REAL background only
    python -m src.run_m4 --quick                                  # ~2 min smoke test (numbers not meaningful)
(`caffeinate -i` keeps the Mac awake; a sleeping Mac pauses training for hours.)

One model, two outputs: "is there a mosquito?" (presence) and "which species?" (Ae. aegypti, Ae. albopictus,
An. stephensi, Cx. quinquefasciatus, other mosquito). Training data options - validation and every test set are ALWAYS
real recordings:
  real        real windows only
  real+synth  real windows + the config's synthetic set (m4_v2: synthetic v2 = 24,000 mosquito clips on randomised
              backgrounds + 12,000 noise-only clips)
  synth       synthetic clips + real BACKGROUND windows only (no real mosquito audio; the synthetic set has no
              "other mosquito" clips, so that class cannot be learned in this mode)
The config sets the fixes against the recording-site shortcut: spectral whitening (model.whiten_bands), site-balanced
weights (train.site_balance), cosine learning-rate decay and model selection on val_score (train.monitor).
Older behaviour: --config configs/m4_species_real.yaml (synthetic v1, no whitening, no site balance, val-loss stopping).
Each run writes to its own folder (default results/m4_runs/<time>_<train-data>) so nothing is overwritten.
"""
import argparse
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src import evaluate
from src.evaluate_species import NAMES, predict_species, scores
from src.evaluate_species import plot_cm
from src.models import SPECIES, build_cnn, set_species
from src.run_m3 import describe, log, setup_logging, step
from src.train import ValScore, balanced_weights, load_split, lr_schedule, make_ds, targets

GENUS_OF = {"ae_aegypti": "Aedes", "ae_albopictus": "Aedes", "an_stephensi": "Anopheles", "cx_quinquefasciatus": "Culex"}


class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, n_batches, every=100, monitor="val_loss"):
        super().__init__()
        self.n, self.every, self.monitor = n_batches, every, monitor
        self.sign = 1 if monitor == "val_loss" else -1          # lower is better for loss, higher for scores
        self.best = np.inf

    def on_epoch_begin(self, epoch, logs=None):
        self.t0, self.epoch = time.time(), epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.every == 0:
            log.info(f"    epoch {self.epoch}: batch {batch + 1}/{self.n}  loss {logs['loss']:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        mark, v = "", self.sign * logs[self.monitor]
        if v < self.best:
            self.best, mark = v, "  <- best so far (weights kept)"
        log.info(f"  epoch {epoch + 1:2d}  train: loss {logs['loss']:.4f}, presence AUC {logs['presence_auc']:.4f}, "
                 f"species acc {logs['species_sparse_categorical_accuracy']:.3f}  |  val: loss {logs['val_loss']:.4f}, "
                 f"presence AUC {logs['val_presence_auc']:.4f}, species balanced acc {logs.get('val_species_bal_acc', float('nan')):.3f}"
                 f", score {logs.get('val_score', float('nan')):.4f}  ({time.time() - self.t0:.0f} s){mark}")


def species_report(out, split):
    """Per-class precision / recall / F1 (windows and recordings), confusion matrix, genus, per-source recall."""
    M, P = predict_species(out, split)
    if not len(M):
        log.info(f"  {split}: no species-labelled windows")
        return None
    y = M.species.map({s: i for i, s in enumerate(SPECIES)}).values
    yhat = P.argmax(1)
    win = scores(y, yhat)
    M["_i"] = range(len(M))
    rec = M.groupby("src").agg(true=("species", "first"), idx=("_i", list))
    ry = rec.true.map({s: i for i, s in enumerate(SPECIES)}).values
    rec_scores = scores(ry, np.array([P[i].mean(0).argmax() for i in rec.idx]))
    rows = []
    for s in SPECIES:
        w, r = win["per_class"][s], rec_scores["per_class"][s]
        if w["n"] or r["n"]:
            rows.append(dict(species=NAMES[s], windows=w["n"], precision=round(w["precision"], 3), recall=round(w["recall"], 3),
                             f1=round(w["f1"], 3), recordings=r["n"], recording_recall=round(r["recall"], 3)))
    log.info(f"\n  {split.upper()}: per species (window level; recording level = average over a recording's windows)")
    log.info("\n" + pd.DataFrame(rows).to_string(index=False))
    log.info(f"  macro-F1: windows {win['macro_f1']:.3f} | recordings {rec_scores['macro_f1']:.3f}   "
             f"(accuracy: windows {win['accuracy']:.3f}, recordings {rec_scores['accuracy']:.3f})")
    tgt = M.species.isin(GENUS_OF).values
    if tgt.any():
        true_g = M.species[tgt].map(GENUS_OF).values
        pred_g = np.array([GENUS_OF.get(SPECIES[k], "other") for k in yhat[tgt]])
        log.info(f"  genus accuracy on the 4 target species (Aedes / Anopheles / Culex): {np.mean(true_g == pred_g):.3f}")
    by_src = M.assign(correct=(y == yhat)).groupby(["species", "site"]).correct.agg(["mean", "size"]).round(3)
    log.info("  recall by recording site (where does it work, where not):")
    log.info("\n" + by_src.rename(columns={"mean": "recall", "size": "windows"}).to_string())
    plot_cm(y, yhat, f"species, {split} windows (macro-F1 {win['macro_f1']:.2f})", out / f"species_cm_{split}.png")
    log.info(f"  confusion matrix: {out / f'species_cm_{split}.png'}")
    return dict(window=win, recording=rec_scores)


def unknown_species_check(out, split):
    """Mosquitoes of species the model was NOT trained to name: does low confidence separate them from the target species?

    For each confidence threshold: share of target-species windows answered (confident) and correct among those, and share
    of other-species windows answered "uncertain" (max probability below the threshold)."""
    cfg = yaml.safe_load(open(Path(out) / "config.yaml"))
    fdir = Path(cfg["features_dir"])
    M = pd.read_csv(fdir / f"{split}_meta.csv", keep_default_na=False)
    X = np.load(fdir / f"{split}.npy", mmap_mode="r")
    model = tf.keras.models.load_model(Path(out) / "model.keras")
    tgt = np.flatnonzero(((M.presence == 1) & M.species.isin(SPECIES)).values)
    oth = np.flatnonzero(((M.presence == 1) & (M.species == "other_mosquito")).values)
    Pt = model.predict(X[tgt].astype(np.float32), batch_size=1024, verbose=0)["species"]
    Po = model.predict(X[oth].astype(np.float32), batch_size=1024, verbose=0)["species"]
    yt = M.species.values[tgt]
    forced = pd.Series(np.array(SPECIES)[Po.argmax(1)]).map(NAMES).value_counts(normalize=True).round(3)
    log.info(f"\n  {split.upper()}: {len(oth):,} windows of OTHER mosquito species (not trained to name) -- forced guesses:")
    log.info("  " + ", ".join(f"{k} {v:.0%}" for k, v in forced.items()))
    rows = []
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        conf_t, conf_o = Pt.max(1) >= thr, Po.max(1) >= thr
        correct = (np.array(SPECIES)[Pt.argmax(1)] == yt)
        rows.append(dict(confidence_threshold=thr, target_answered=f"{conf_t.mean():.0%}",
                         target_correct_when_answered=f"{correct[conf_t].mean() if conf_t.any() else float('nan'):.0%}",
                         other_species_said_uncertain=f"{1 - conf_o.mean():.0%}"))
    log.info("  'uncertain' option: answer a species only if the model's confidence >= threshold")
    log.info("\n" + pd.DataFrame(rows).to_string(index=False))


def other_threshold_table(out, split):
    """The app's sensitivity setting: answer "other mosquito / species uncertain" when P(other) >= threshold, otherwise
    the most likely of the 4 target species. Recording level (probabilities averaged over a recording's windows)."""
    M, P = predict_species(out, split)
    oi = SPECIES.index("other_mosquito")
    tgt_idx = [i for i in range(len(SPECIES)) if i != oi]
    M["_i"] = range(len(M))
    rec = M.groupby("src").agg(true=("species", "first"), idx=("_i", list))
    y = rec.true.map({s: i for i, s in enumerate(SPECIES)}).values
    Pr = np.array([P[i].mean(0) for i in rec.idx])
    is_t, is_o = y != oi, y == oi
    rows = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        ans = np.where(Pr[:, oi] >= thr, oi, np.array(tgt_idx)[Pr[:, tgt_idx].argmax(1)])
        rows.append({"P(other) threshold": thr, "target species answered": f"{(ans[is_t] != oi).mean():.0%}",
                     "correct when answered": f"{(ans[is_t] == y[is_t])[ans[is_t] != oi].mean():.0%}",
                     "other species rejected": f"{(ans[is_o] == oi).mean():.0%}",
                     "other called Ae. aegypti": f"{(ans[is_o] == SPECIES.index('ae_aegypti')).mean():.0%}"})
    log.info(f"\n  {split.upper()} recordings: sensitivity setting -- 'other / uncertain' if P(other) >= threshold, else the "
             f"best target species ({int(is_t.sum())} target, {int(is_o.sum())} other-species recordings)")
    log.info("\n" + pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m4_v2.yaml")
    ap.add_argument("--train-data", choices=["real", "real+synth", "synth"], default="real+synth")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="small subset, 2 epochs, val + test only (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_species(cfg.get("species_classes"))
    tag = args.train_data.replace("+", "_plus_") + ("_quick" if args.quick else "")
    out = Path(args.out or f"results/m4_runs/{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.quick:
        cfg["train"]["epochs"] = 2
        cfg["evaluate"]["splits"] = ["val", "test"]
    cfg.update(name=f"m4_{tag}", out_dir=str(out), train_data=args.train_data, tasks=["presence", "species"])
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)

    step(1, "environment and settings")
    log.info(f"  python {platform.python_version()} | TensorFlow {tf.__version__} | devices {[d.device_type for d in tf.config.list_physical_devices()]}")
    log.info(f"  training data: {args.train_data} | seed {cfg['seed']} | output {out}")
    m = cfg["model"]
    arch = (f"{m.get('resnet', 'ResNet50V2')} ({m.get('pretrained') or 'random init'}), input resized to {m.get('input_size', [128, 128])}"
            if m.get("arch") == "resnet" else f"CNN channels {m['channels']}")
    log.info(f"  model {arch}, dropout {m['dropout']} | batch {cfg['train']['batch_size']}, "
             f"lr {cfg['train']['learning_rate']}, max epochs {cfg['train']['epochs']}, early stop patience {cfg['train']['early_stopping_patience']}")
    ps = cfg["augment"].get("pitch_shift")
    log.info(f"  augmentation: gain ±{cfg['augment']['gain_db']} dB, freq mask ≤{cfg['augment']['freq_mask']} bands, "
             f"time mask ≤{cfg['augment']['time_mask']} frames (p={cfg['augment']['mask_prob']})"
             + (f", pitch shift ±{(ps['max_ratio'] - 1) * 100:.0f} % (p={ps['prob']})" if ps else ", no pitch shift"))
    log.info(f"  species classes: {', '.join(NAMES[s] for s in SPECIES)}")
    if "other_mosquito" not in SPECIES:
        log.info("  other mosquito species: NOT a species class -- they still train and test the detector; the species")
        log.info("  report checks whether the model is unsure about them (an 'uncertain' answer for unknown species)")
    tr = cfg["train"]
    log.info(f"  fixes: whitening {cfg['model'].get('whiten_bands') or 'off'} bands | site balance {tr.get('site_balance', False)} | "
             f"lr schedule {tr.get('lr_schedule', 'constant')} | select epoch by {tr.get('monitor', 'val_loss')} | "
             f"synthetic set {cfg.get('synthetic_set', 'synth_train')}")
    tf.keras.utils.set_random_seed(cfg["seed"])
    tf.config.experimental.enable_op_determinism()

    step(2, "load features (1 s log-mel windows, 64 bands x 63 frames)")
    fdir = cfg["features_dir"]
    Xtr, Mtr = load_split(fdir, "train")
    if args.train_data in ("real+synth", "synth"):
        sets = cfg.get("synthetic_set", "synth_train")
        parts = [load_split(fdir, name) for name in ([sets] if isinstance(sets, str) else sets)]
        Xs, Ms = np.concatenate([x for x, _ in parts]), pd.concat([m for _, m in parts], ignore_index=True)
        if args.train_data == "synth":
            bg = np.flatnonzero((Mtr.presence == 0).values)
            Xtr, Mtr = Xtr[bg], Mtr.iloc[bg].reset_index(drop=True)
        Xtr, Mtr = np.concatenate([Xtr, Xs]), pd.concat([Mtr, Ms], ignore_index=True)
    Xva, Mva = load_split(fdir, "val")
    if args.quick:
        rng = np.random.default_rng(0)
        i, j = rng.choice(len(Xtr), 8000, replace=False), rng.choice(len(Xva), 3000, replace=False)
        Xtr, Mtr, Xva, Mva = Xtr[i], Mtr.iloc[i].reset_index(drop=True), Xva[j], Mva.iloc[j].reset_index(drop=True)
        log.info("  QUICK MODE: 8,000 training / 3,000 validation windows, 2 epochs -- pipeline check only; numbers are not meaningful.")
    describe("train", Mtr)
    describe("val", Mva)
    for s in cfg["evaluate"]["splits"][1:]:
        describe(s, pd.read_csv(Path(fdir) / f"{s}_meta.csv", keep_default_na=False))
    lab = Mtr[Mtr.species.isin(SPECIES)]
    syn = lab.get("synthetic", pd.Series(False, index=lab.index)).astype(str).eq("True")
    log.info("  species-labelled TRAINING windows (real / synthetic):")
    for s in SPECIES:
        log.info(f"    {NAMES[s]:22s} {int(((lab.species == s) & ~syn).sum()):>7,} real  {int(((lab.species == s) & syn).sum()):>7,} synthetic")
    log.info("  (validation and all test sets are real recordings; synthetic clips appear in training only)")

    step(3, "build the model (shared CNN, two outputs: presence + species)")
    tasks = ["presence", "species"]
    ytr, _ = targets(Mtr, tasks)
    yva, wva = targets(Mva, tasks)
    wtr = balanced_weights(Mtr, ytr, tasks, by_site=cfg["train"].get("site_balance", False))
    for t in tasks:
        log.info(f"  {t} training weights: min {wtr[t][wtr[t] > 0].min():.2f}, median {np.median(wtr[t][wtr[t] > 0]):.2f}, "
                 f"max {wtr[t].max():.2f}" + ("  (equal weight per class and per recording site)" if cfg['train'].get('site_balance') else ""))
    real_lab = Mtr[Mtr.species.isin(SPECIES) & ~Mtr.get("synthetic", pd.Series(False, index=Mtr.index)).astype(str).eq("True")]
    log.info("  training recording sites per species (real): " + "; ".join(
        f"{NAMES[s]} {real_lab[real_lab.species == s].site.nunique()}" for s in SPECIES))
    sample = Xtr[np.random.default_rng(0).choice(len(Xtr), min(20000, len(Xtr)), replace=False)].astype(np.float32)
    model = build_cnn(cfg, Path(fdir) / "norm.json", Xtr.shape[1:], tasks, norm_sample=sample)
    steps = int(np.ceil(len(Xtr) / cfg["train"]["batch_size"]))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule(cfg, steps)),
                  loss={"presence": "binary_crossentropy", "species": "sparse_categorical_crossentropy"},
                  metrics={"presence": [tf.keras.metrics.AUC(name="auc")], "species": ["sparse_categorical_accuracy"]})
    for layer in model.layers:
        shape = getattr(layer, "output", None)
        shape = tuple(shape.shape) if shape is not None and hasattr(shape, "shape") else ""
        log.info(f"  {layer.name:12s} {layer.__class__.__name__:22s} output {str(shape):22s} params {layer.count_params():>7,}")
    log.info(f"  total parameters: {model.count_params():,} (~{model.count_params() / 1024:.0f} KB after int8 quantisation, "
             f"~{model.count_params() * 4 / 1024:.0f} KB as float32 -- both small for an Android phone)")

    monitor = cfg["train"].get("monitor", "val_loss")
    mode = "min" if monitor == "val_loss" else "max"
    step(4, f"train (keeps the epoch with the best {monitor}; stops after {cfg['train']['early_stopping_patience']} epochs without improvement)")
    t0 = time.time()
    hist = model.fit(make_ds(Xtr, ytr, wtr, cfg, True), validation_data=make_ds(Xva, yva, wva, cfg, False),
                     epochs=cfg["train"]["epochs"], verbose=0,
                     callbacks=[ValScore(Xva, Mva), EpochLogger(steps, monitor=monitor),
                                tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=cfg["train"]["early_stopping_patience"],
                                                                 restore_best_weights=True),
                                tf.keras.callbacks.CSVLogger(out / "history.csv")])
    best = int(np.argmin(hist.history[monitor]) if mode == "min" else np.argmax(hist.history[monitor]))
    model.save(out / "model.keras")
    log.info(f"  done in {(time.time() - t0) / 60:.1f} min: best epoch {best + 1}; model saved to {out / 'model.keras'}")

    step(5, "evaluate 'is there a mosquito?' on REAL recordings")
    log.info("  test = held-out sessions / labs; test_A = Kenya and test_B = Thailand (never heard in training); external = UFRGS phones")
    pres = evaluate.run(out, log=log.info)

    step(6, "evaluate 'which species?' on REAL recordings")
    log.info("  Only windows with a known species label count. Ae. albopictus and An. stephensi have ~1 min of test audio each")
    log.info("  (the public data has no more from independent labs), so their numbers are weak evidence.")
    sp = {s: species_report(out, s) for s in ["val", "test"]}
    if "other_mosquito" not in SPECIES:
        unknown_species_check(out, "test")
    else:
        other_threshold_table(out, "test")

    step(7, "summary" + ("  (QUICK MODE: numbers are not meaningful)" if args.quick else ""))
    for s, r in pres["splits"].items():
        log.info(f"  presence  {s:9s} ROC-AUC {r.get('roc_auc', float('nan')):.3f}")
    for s, r in sp.items():
        if r:
            log.info(f"  species   {s:9s} macro-F1 {r['window']['macro_f1']:.3f} (windows), {r['recording']['macro_f1']:.3f} (recordings)")
    log.info(f"  all outputs in {out}  (run.log, model.keras, metrics.json, operating_points.csv, detection.png, species_cm_*.png)")


if __name__ == "__main__":
    main()
