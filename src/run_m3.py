"""Run M3 end to end with readable logs in the terminal (and a copy in <out>/run.log).

    conda activate mosquito
    cd "/Users/rushil/Documents/Mosquito Detector"
    python -m src.run_m3                         # full run, real training data (the M3 baseline), ~20 min
    python -m src.run_m3 --quick                 # ~3 min smoke test on a small subset
    python -m src.run_m3 --train-data real+synth # real + synthetic mosquitoes
    python -m src.run_m3 --train-data synth      # synthetic mosquitoes + REAL background only

Training data options (validation and every test set are ALWAYS real recordings):
  real        real mosquito + real background windows (M3 baseline)
  real+synth  real windows + 24,000 synthetic v1.1 mosquito clips
  synth       synthetic mosquito clips + real BACKGROUND windows only (no real mosquito audio at all):
              the purest test of whether synthetic data teaches the model what a real mosquito sounds like
Each run writes to its own folder (default results/m3_runs/<time>_<train-data>) so nothing is overwritten.
"""
import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow's internal C++ messages (Python errors still show)

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src import evaluate
from src.models import build_cnn
from src.train import load_split, make_ds, targets

log = logging.getLogger("m3")


def setup_logging(out):
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def step(n, title):
    log.info("")
    log.info("=" * 78)
    log.info(f"STEP {n}: {title}")
    log.info("=" * 78)


class EpochLogger(tf.keras.callbacks.Callback):
    """One readable line per epoch plus a progress line every `every` batches."""

    def __init__(self, n_batches, every=100):
        super().__init__()
        self.n, self.every, self.best = n_batches, every, -np.inf

    def on_epoch_begin(self, epoch, logs=None):
        self.t0, self.epoch = time.time(), epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.every == 0:
            log.info(f"    epoch {self.epoch}: batch {batch + 1}/{self.n}  loss {logs['loss']:.4f}  auc {logs['auc']:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        mark = ""
        if logs["val_auc"] > self.best:
            self.best, mark = logs["val_auc"], "  <- best so far (weights kept)"
        log.info(f"  epoch {epoch + 1:2d}  loss {logs['loss']:.4f}  train AUC {logs['auc']:.4f}  |  "
                 f"val loss {logs['val_loss']:.4f}  val AUC {logs['val_auc']:.4f}  ({time.time() - self.t0:.0f} s){mark}")


def describe(name, M):
    mos, bg = int((M.presence == 1).sum()), int((M.presence == 0).sum())
    syn = int(M.get("synthetic", pd.Series(False, index=M.index)).astype(str).eq("True").sum())
    log.info(f"  {name:10s} {len(M):>8,} windows | mosquito {mos:>7,} | background {bg:>7,}"
             + (f" | synthetic {syn:,}" if syn else "") + f" | ~{len(M) * 0.5 / 3600:.1f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m3_detector.yaml")
    ap.add_argument("--train-data", choices=["real", "real+synth", "synth"], default="real")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override the config's maximum epochs")
    ap.add_argument("--quick", action="store_true", help="small subset, 2 epochs, val + test only (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tag = args.train_data.replace("+", "_plus_") + ("_quick" if args.quick else "")
    out = Path(args.out or f"results/m3_runs/{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.quick:
        cfg["train"]["epochs"] = 2
        cfg["evaluate"]["splits"] = ["val", "test"]
    cfg.update(name=f"m3_{tag}", out_dir=str(out), train_data=args.train_data)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"), sort_keys=False)

    step(1, "environment and settings")
    log.info(f"  python {platform.python_version()} | TensorFlow {tf.__version__} | devices {[d.device_type for d in tf.config.list_physical_devices()]}")
    log.info(f"  config {args.config} | training data: {args.train_data} | seed {cfg['seed']} | output {out}")
    log.info(f"  model channels {cfg['model']['channels']}, dropout {cfg['model']['dropout']} | batch {cfg['train']['batch_size']}, "
             f"lr {cfg['train']['learning_rate']}, max epochs {cfg['train']['epochs']}, early stop patience {cfg['train']['early_stopping_patience']}")
    log.info(f"  augmentation: gain ±{cfg['augment']['gain_db']} dB, freq mask ≤{cfg['augment']['freq_mask']} bands, "
             f"time mask ≤{cfg['augment']['time_mask']} frames (p={cfg['augment']['mask_prob']})")
    tf.keras.utils.set_random_seed(cfg["seed"])
    tf.config.experimental.enable_op_determinism()

    step(2, "load features (1 s log-mel windows, 64 bands x 63 frames)")
    fdir = cfg["features_dir"]
    Xtr, Mtr = load_split(fdir, "train")
    if args.train_data in ("real+synth", "synth"):
        Xs, Ms = load_split(fdir, "synth_train")
        if args.train_data == "synth":  # keep only REAL background; all mosquito examples are synthetic
            bg = np.flatnonzero((Mtr.presence == 0).values)
            Xtr, Mtr = Xtr[bg], Mtr.iloc[bg].reset_index(drop=True)
        Xtr, Mtr = np.concatenate([Xtr, Xs]), pd.concat([Mtr, Ms], ignore_index=True)
    Xva, Mva = load_split(fdir, "val")
    if args.quick:
        rng = np.random.default_rng(0)
        i, j = rng.choice(len(Xtr), 8000, replace=False), rng.choice(len(Xva), 3000, replace=False)
        Xtr, Mtr, Xva, Mva = Xtr[i], Mtr.iloc[i].reset_index(drop=True), Xva[j], Mva.iloc[j].reset_index(drop=True)
        log.info("  QUICK MODE: 8,000 training and 3,000 validation windows, 2 epochs -- a smoke test of the pipeline;")
        log.info("  its accuracy numbers are meaningless (the model is barely trained). Run without --quick for real results.")
    describe("train", Mtr)
    describe("val", Mva)
    for s in cfg["evaluate"]["splits"][1:]:
        describe(s, pd.read_csv(Path(fdir) / f"{s}_meta.csv", keep_default_na=False))
    log.info("  (validation and all test sets are real recordings; synthetic clips appear in training only)")

    step(3, "build the model")
    ytr, wtr = targets(Mtr, ["presence"])
    yva, wva = targets(Mva, ["presence"])
    p = ytr["presence"]
    cw = {1: len(p) / (2 * p.sum()), 0: len(p) / (2 * (len(p) - p.sum()))}
    wtr["presence"] = np.where(p == 1, cw[1], cw[0]).astype(np.float32)
    model = build_cnn(cfg, Path(fdir) / "norm.json", Xtr.shape[1:], ["presence"])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["train"]["learning_rate"]),
                  loss={"presence": "binary_crossentropy"}, metrics={"presence": [tf.keras.metrics.AUC(name="auc")]})
    for layer in model.layers:
        shape = getattr(layer, "output", None)
        shape = tuple(shape.shape) if shape is not None else ""
        log.info(f"  {layer.name:12s} {layer.__class__.__name__:22s} output {str(shape):22s} params {layer.count_params():>7,}")
    log.info(f"  total parameters: {model.count_params():,} (~{model.count_params() * 4 / 1024:.0f} KB as float32, "
             f"~{model.count_params() / 1024:.0f} KB after int8 quantisation)")
    log.info(f"  class weights (balance mosquito vs background): mosquito {cw[1]:.2f}, background {cw[0]:.2f}")

    step(4, "train (early stopping on validation ROC-AUC)")
    ds_tr = make_ds(Xtr, ytr, wtr, cfg, True)
    t0 = time.time()
    hist = model.fit(ds_tr, validation_data=make_ds(Xva, yva, wva, cfg, False), epochs=cfg["train"]["epochs"], verbose=0,
                     callbacks=[EpochLogger(int(np.ceil(len(Xtr) / cfg["train"]["batch_size"]))),
                                tf.keras.callbacks.EarlyStopping("val_auc", mode="max", patience=cfg["train"]["early_stopping_patience"],
                                                                 restore_best_weights=True),
                                tf.keras.callbacks.CSVLogger(out / "history.csv")])
    best = int(np.argmax(hist.history["val_auc"]))
    model.save(out / "model.keras")
    log.info(f"  done in {(time.time() - t0) / 60:.1f} min: best epoch {best + 1} (val AUC {hist.history['val_auc'][best]:.4f}); "
             f"model saved to {out / 'model.keras'}")

    step(5, "evaluate on REAL recordings")
    log.info("  test = held-out sessions / labs; test_A = Kenya and test_B = Thailand (sites never heard in training);")
    log.info("  external = UFRGS phone recordings. False alarms are counted per hour of background audio.")
    res = evaluate.run(out, log=log.info)

    step(6, "summary" + ("  (QUICK MODE: numbers are not meaningful)" if args.quick else ""))
    for s, r in res["splits"].items():
        log.info(f"  {s:9s} ROC-AUC {r.get('roc_auc', float('nan')):.3f}")
    log.info(f"  all outputs in {out}  (run.log, model.keras, metrics.json, operating_points.csv, detection.png)")


if __name__ == "__main__":
    main()
