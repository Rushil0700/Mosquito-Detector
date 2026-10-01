"""Train a model from a config (M3: presence detector; M4 adds the species head).

Usage: python -m src.train configs/m3_detector.yaml
Writes <out_dir>/model.keras, history.csv, config.yaml (copy of the settings used) and train_log.txt.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.models import SPECIES, build_cnn


def load_split(fdir, split, synthetic=False):
    X = np.load(Path(fdir) / f"{split}.npy")
    M = pd.read_csv(Path(fdir) / f"{split}_meta.csv", keep_default_na=False)
    return X, M


def targets(M, tasks):
    y, w = {}, {}
    if "presence" in tasks:
        y["presence"] = M.presence.values.astype(np.float32)
        w["presence"] = np.ones(len(M), np.float32)
    if "species" in tasks:
        idx = M.species.map({s: i for i, s in enumerate(SPECIES)})
        y["species"] = idx.fillna(0).astype(np.int32).values
        w["species"] = idx.notna().astype(np.float32).values  # masked where no species label
    return y, w


def balanced_weights(M, y, tasks, by_site=True, cap=10.0):
    """Sample weights: every class gets equal total weight; with by_site, within a class every recording SITE gets equal
    weight too (so e.g. Ifakara, which holds most "other mosquito" audio, cannot teach "Ifakara = other"). Per-window
    weights are capped at `cap` x the class median so a site with a handful of windows cannot dominate."""
    site = M.site.values if by_site else np.zeros(len(M))
    w = {}
    for t in tasks:
        lab = (np.ones(len(M), bool) if t == "presence" else M.species.isin(SPECIES).values)
        cls = y[t].astype(int)
        wt = np.zeros(len(M), np.float32)
        classes = np.unique(cls[lab])
        for c in classes:
            in_c = lab & (cls == c)
            sites = np.unique(site[in_c])
            for st in sites:
                g = in_c & (site == st)
                wt[g] = 1.0 / (len(classes) * len(sites) * g.sum())
            med = np.median(wt[in_c])
            wt[in_c] = np.minimum(wt[in_c], cap * med)
        wt[lab] *= lab.sum() / wt[lab].sum()   # mean weight 1 over labelled windows
        w[t] = wt
    return w


class ValScore(tf.keras.callbacks.Callback):
    """Adds logs['val_score'] = mean(validation presence AUC, validation species balanced accuracy): steadier than the
    total validation loss for choosing the epoch (the species part of the loss swings by +-0.3 between epochs)."""

    def __init__(self, Xva, Mva, batch=1024):
        super().__init__()
        self.X, self.lab = Xva, Mva.species.isin(SPECIES).values
        self.y = Mva.species.map({s: i for i, s in enumerate(SPECIES)}).values

    def on_epoch_end(self, epoch, logs=None):
        P = self.model.predict(self.X[self.lab].astype(np.float32), batch_size=1024, verbose=0)["species"].argmax(1)
        y = self.y[self.lab].astype(int)
        recalls = [np.mean(P[y == c] == c) for c in np.unique(y)]
        logs["val_species_bal_acc"] = float(np.mean(recalls))
        logs["val_score"] = 0.5 * (logs["val_presence_auc"] + logs["val_species_bal_acc"])


def lr_schedule(cfg, steps_per_epoch):
    t = cfg["train"]
    if t.get("lr_schedule") == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(t["learning_rate"], decay_steps=t["epochs"] * steps_per_epoch, alpha=0.02)
    return t["learning_rate"]


def _mel(f):
    """Slaney mel scale (librosa's default, htk=False), as used by src/features.py."""
    f = tf.convert_to_tensor(f, tf.float32)
    return tf.where(f < 1000.0, f * 3.0 / 200.0, 15.0 + tf.math.log(tf.maximum(f, 1e-3) / 1000.0) * 27.0 / np.log(6.4))


def pitch_shifter(fcfg):
    """Returns shift(x, ratio): the log-mel window of the same sound with every frequency multiplied by `ratio`.
    Output band k (centre frequency f_k) is read from the input at frequency f_k / ratio (linear interpolation between
    bands; bands whose source lies outside fmin..fmax repeat the edge band). Moves the fundamental and all harmonics
    together, like a mosquito flying warmer / colder or reared in another lab."""
    import librosa
    n = fcfg["n_mels"]
    centres = tf.constant(librosa.mel_frequencies(n + 2, fmin=fcfg["fmin"], fmax=fcfg["fmax"])[1:-1], tf.float32)
    lo, hi = _mel(float(fcfg["fmin"])), _mel(float(fcfg["fmax"]))

    def shift(x, ratio):
        src = tf.clip_by_value((_mel(centres / ratio) - lo) / (hi - lo) * (n + 1) - 1, 0.0, n - 1.0)
        i0 = tf.cast(tf.floor(src), tf.int32)
        i1 = tf.minimum(i0 + 1, n - 1)
        w = (src - tf.floor(src))[:, None]
        return tf.gather(x, i0) * (1 - w) + tf.gather(x, i1) * w
    return shift


def augment(cfg):
    a = cfg["augment"]
    ps = a.get("pitch_shift")
    if ps:
        shift = pitch_shifter(yaml.safe_load(open(cfg.get("features_config", "configs/features.yaml"))))
        log_max = float(np.log(ps["max_ratio"]))

    def fn(x, y, w):
        x = tf.cast(x, tf.float32)
        if ps and tf.random.uniform([]) < ps["prob"]:  # ratio log-uniform in [1/max_ratio, max_ratio]
            x = shift(x, tf.exp(tf.random.uniform([], -log_max, log_max)))
        x = x + tf.random.uniform([], -a["gain_db"], a["gain_db"])  # level shift in dB
        n_mels, n_t = x.shape[0], x.shape[1]
        if tf.random.uniform([]) < a["mask_prob"]:
            f = tf.random.uniform([], 0, a["freq_mask"] + 1, tf.int32)
            f0 = tf.random.uniform([], 0, n_mels - f + 1, tf.int32)
            m = tf.concat([tf.ones([f0]), tf.zeros([f]), tf.ones([n_mels - f0 - f])], 0)[:, None]
            x = x * m + (1 - m) * tf.reduce_mean(x)
        if tf.random.uniform([]) < a["mask_prob"]:
            t = tf.random.uniform([], 0, a["time_mask"] + 1, tf.int32)
            t0 = tf.random.uniform([], 0, n_t - t + 1, tf.int32)
            m = tf.concat([tf.ones([t0]), tf.zeros([t]), tf.ones([n_t - t0 - t])], 0)[None, :]
            x = x * m + (1 - m) * tf.reduce_mean(x)
        return x, y, w
    return fn


def make_ds(X, y, w, cfg, train):
    ds = tf.data.Dataset.from_tensor_slices((X, y, w))
    if train:
        ds = ds.shuffle(len(X), seed=cfg["seed"], reshuffle_each_iteration=True).map(augment(cfg), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y, w: (tf.cast(x, tf.float32), y, w))
    return ds.batch(cfg["train"]["batch_size"]).prefetch(tf.data.AUTOTUNE)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tf.keras.utils.set_random_seed(cfg["seed"])
    tf.config.experimental.enable_op_determinism()
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(sys.argv[1], out / "config.yaml")
    fdir, tasks = cfg["features_dir"], cfg["tasks"]

    Xtr, Mtr = load_split(fdir, "train")
    if cfg.get("use_synthetic"):
        Xs, Ms = load_split(fdir, "synth_train")
        Xtr, Mtr = np.concatenate([Xtr, Xs]), pd.concat([Mtr, Ms], ignore_index=True)
    Xva, Mva = load_split(fdir, "val")
    ytr, wtr = targets(Mtr, tasks)
    yva, wva = targets(Mva, tasks)
    if "presence" in tasks and cfg["train"]["class_weight"] == "balanced":
        p = ytr["presence"]
        cw = {1: len(p) / (2 * p.sum()), 0: len(p) / (2 * (len(p) - p.sum()))}
        wtr["presence"] = np.where(p == 1, cw[1], cw[0]).astype(np.float32)
    if "species" in tasks:  # balance species classes among labelled windows
        lab = wtr["species"] > 0
        counts = np.bincount(ytr["species"][lab], minlength=len(SPECIES)).astype(np.float32)
        cls_w = lab.sum() / (len(SPECIES) * np.maximum(counts, 1))
        wtr["species"] = np.where(lab, cls_w[ytr["species"]], 0).astype(np.float32)

    model = build_cnn(cfg, Path(fdir) / "norm.json", Xtr.shape[1:], tasks)
    losses = {"presence": "binary_crossentropy", "species": "sparse_categorical_crossentropy"}
    metrics = {"presence": [tf.keras.metrics.AUC(name="auc")], "species": ["sparse_categorical_accuracy"]}
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["train"]["learning_rate"]),
                  loss={t: losses[t] for t in tasks}, metrics={t: metrics[t] for t in tasks})
    params = model.count_params()
    print(f"{cfg['name']}: {params:,} parameters | train {len(Xtr):,} windows | val {len(Xva):,}", flush=True)
    # Keras prefixes metric names with the output name only when the model has several outputs
    # single task: val AUC; several heads: total validation loss (species has no val data for 2 classes, so AUC alone
    # would ignore the species head)
    monitor = "val_auc" if tasks == ["presence"] else "val_loss"
    mode = "max" if "auc" in monitor else "min"
    cb = [tf.keras.callbacks.EarlyStopping(monitor, mode=mode, patience=cfg["train"]["early_stopping_patience"], restore_best_weights=True),
          tf.keras.callbacks.CSVLogger(out / "history.csv")]
    t0 = time.time()
    hist = model.fit(make_ds(Xtr, ytr, wtr, cfg, True), validation_data=make_ds(Xva, yva, wva, cfg, False),
                     epochs=cfg["train"]["epochs"], callbacks=cb, verbose=2)
    model.save(out / "model.keras")
    best = int(np.argmax(hist.history[monitor]) if mode == "max" else np.argmin(hist.history[monitor]))
    info = dict(params=params, epochs_run=len(hist.history[monitor]), best_epoch=best + 1,
                best_val=float(hist.history[monitor][best]), monitor=monitor, train_minutes=round((time.time() - t0) / 60, 1),
                n_train=len(Xtr), n_val=len(Xva))
    json.dump(info, open(out / "train_info.json", "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()
