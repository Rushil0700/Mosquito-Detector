"""M4: species evaluation of one or more runs on REAL data (mosquito windows with a species label).

Usage: python -m src.evaluate_species results/m4_real results/m4_synth [--split test]
Per run: per-class precision / recall / F1 and macro-F1 (window level and recording level = majority vote over a source
file's windows), confusion matrix figure, per-source recall. Writes <run>/species_<split>.json and a comparison table
results/m4_comparison_<split>.csv.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.models import SPECIES

NAMES = {"ae_aegypti": "Ae. aegypti", "ae_albopictus": "Ae. albopictus", "an_stephensi": "An. stephensi",
         "cx_quinquefasciatus": "Cx. quinquefasciatus", "other_mosquito": "other mosquito"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def predict_species(run, split):
    cfg = yaml.safe_load(open(Path(run) / "config.yaml"))
    fdir = Path(cfg["features_dir"])
    M = pd.read_csv(fdir / f"{split}_meta.csv", keep_default_na=False)
    keep = (M.presence == 1) & M.species.isin(SPECIES)
    X = np.load(fdir / f"{split}.npy", mmap_mode="r")[np.flatnonzero(keep)].astype(np.float32)
    model = tf.keras.models.load_model(Path(run) / "model.keras")
    P = model.predict(X, batch_size=1024, verbose=0)["species"]
    return M[keep].reset_index(drop=True), P


def scores(y, yhat):
    labels = list(range(len(SPECIES)))
    p, r, f, n = precision_recall_fscore_support(y, yhat, labels=labels, zero_division=0)
    present = n > 0
    return dict(per_class={SPECIES[i]: dict(precision=p[i], recall=r[i], f1=f[i], n=int(n[i])) for i in labels},
                macro_f1=float(f[present].mean()), accuracy=float((y == yhat).mean()))


def plot_cm(y, yhat, title, path):
    cm = confusion_matrix(y, yhat, labels=range(len(SPECIES)))
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURF)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(SPECIES)):
        for j in range(len(SPECIES)):
            if rows[i, 0]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else INK)
    names = [NAMES[s] for s in SPECIES]
    ax.set_xticks(range(len(SPECIES)), names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(SPECIES)), [f"{n}  (n={int(r)})" for n, r in zip(names, rows[:, 0])], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8, color=MUTED)
    ax.set_ylabel("True (row-normalised; counts in brackets)", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    table = []
    for run in args.runs:
        M, P = predict_species(run, args.split)
        y = M.species.map({s: i for i, s in enumerate(SPECIES)}).values
        yhat = P.argmax(1)
        win = scores(y, yhat)
        # recording level: mean probability over a source file's windows
        M["_i"] = range(len(M))
        rec = M.groupby("src").agg(true=("species", "first"), idx=("_i", list))
        ry = rec.true.map({s: i for i, s in enumerate(SPECIES)}).values
        ryhat = np.array([P[i].mean(0).argmax() for i in rec.idx])
        recs = scores(ry, ryhat)
        by_source = (M.assign(ok=(y == yhat)).groupby(["species", "dataset", "site"]).ok.agg(["mean", "size"]).round(3)
                     .reset_index().rename(columns={"mean": "recall", "size": "windows"}))
        name = yaml.safe_load(open(Path(run) / "config.yaml"))["name"]
        json.dump(dict(window=win, recording=recs, by_source=by_source.to_dict("records")),
                  open(Path(run) / f"species_{args.split}.json", "w"), indent=1, default=float)
        plot_cm(y, yhat, f"{name}: species, {args.split} windows (macro-F1 {win['macro_f1']:.2f})", Path(run) / f"species_cm_{args.split}.png")
        for s in SPECIES:
            table.append(dict(run=name, species=NAMES[s], n_windows=win["per_class"][s]["n"],
                              precision=win["per_class"][s]["precision"], recall=win["per_class"][s]["recall"], f1=win["per_class"][s]["f1"],
                              recording_recall=recs["per_class"][s]["recall"], n_recordings=recs["per_class"][s]["n"]))
        table.append(dict(run=name, species="MACRO-F1 (window / recording)", f1=win["macro_f1"], recording_recall=recs["macro_f1"]))
        print(f"\n== {name} ({args.split}) ==")
        print(by_source.to_string(index=False))
    T = pd.DataFrame(table)
    T.to_csv(f"results/m4_comparison_{args.split}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
