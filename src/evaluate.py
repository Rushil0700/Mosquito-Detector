"""Evaluate a trained detector (M3) on every split.

Usage: python -m src.evaluate results/m3
Reads <run>/config.yaml and model.keras; writes <run>/predictions/<split>.npy, metrics.json, per_species.csv and
figures (ROC per split; recall vs false alarms per hour).

False alarms: background windows (1 s, hop 0.5 s) are ordered per source file; a run of consecutive positive windows
is ONE alarm event. False alarms per hour = events / background hours (windows x hop). "k of n" smoothing = a window
fires only if at least k of the last n windows are positive (a simple on-device debounce).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

HOP_S = 0.5
COLORS = {"val": "#2a78d6", "test": "#eb6834", "test_A": "#1baf7a", "test_B": "#eda100", "external": "#e87ba4"}
LABELS = {"val": "validation", "test": "test (held-out sessions / labs)", "test_A": "test A: Kenya (unseen site)",
          "test_B": "test B: Thailand (unseen site)", "external": "external: UFRGS phones"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"


def smooth(pos, k, n):
    """k-of-n debounce within one ordered sequence of window decisions."""
    if k <= 1:
        return pos
    c = np.convolve(pos.astype(int), np.ones(n, int))[: len(pos)]
    return c >= k


def ordered_groups(M):
    order = M.sort_values(["src", "win_start_s"]).index.values
    return order, M.loc[order, "src"].values


def decisions(p, M, thr, k=1, n=1):
    """Window decisions after optional k-of-n smoothing along each source file."""
    order, src = ordered_groups(M)
    out = np.zeros(len(p), bool)
    starts = np.r_[0, np.flatnonzero(src[1:] != src[:-1]) + 1, len(src)]
    for a, b in zip(starts[:-1], starts[1:]):
        idx = order[a:b]
        out[idx] = smooth(p[idx] >= thr, k, n)
    return out


def false_alarms_per_hour(dec, M):
    bg = (M.presence == 0).values
    if not bg.any():
        return np.nan, 0.0
    Mb = M[bg]
    order, src = ordered_groups(Mb)
    d = dec[bg][np.argsort(np.argsort(Mb.index.values))] if False else pd.Series(dec[bg], index=Mb.index).loc[order].values
    new_src = np.r_[True, src[1:] != src[:-1]]
    rising = d & (new_src | ~np.r_[False, d[:-1]])
    hours = bg.sum() * HOP_S / 3600
    return float(rising.sum() / hours), hours


def split_metrics(p, M, thr, smoothing):
    y = M.presence.values
    out = dict(windows=len(y), mosquito=int(y.sum()), background=int((1 - y).sum()))
    if 0 < y.sum() < len(y):
        out.update(roc_auc=roc_auc_score(y, p), avg_precision=average_precision_score(y, p))
    for tag, (k, n) in {"raw": (1, 1), f"{smoothing[0]}of{smoothing[1]}": tuple(smoothing)}.items():
        d = decisions(p, M, thr, k, n)
        pr, rc, f1, _ = precision_recall_fscore_support(y, d, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
        fa, hours = false_alarms_per_hour(d, M)
        out[tag] = dict(precision=pr, recall=rc, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                        false_alarms_per_hour=fa, background_hours=round(hours, 2))
    return out


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)


def run(run, log=print, splits=None):
    """Evaluate the model in `run`; `log` receives every line of output (print or a logger)."""
    run = Path(run)
    cfg = yaml.safe_load(open(run / "config.yaml"))
    ev, fdir = cfg["evaluate"], Path(cfg["features_dir"])
    if splits:
        ev["splits"] = [s for s in ev["splits"] if s in splits]
    model = tf.keras.models.load_model(run / "model.keras")
    (run / "predictions").mkdir(exist_ok=True)
    P, M = {}, {}
    for split in ev["splits"]:
        log(f"  predicting {split} ...")
        X = np.load(fdir / f"{split}.npy")
        M[split] = pd.read_csv(fdir / f"{split}_meta.csv", keep_default_na=False)
        out = model.predict(X.astype(np.float32), batch_size=1024, verbose=0)
        P[split] = np.asarray(out["presence"] if isinstance(out, dict) else out).ravel()
        np.save(run / "predictions" / f"{split}.npy", P[split])

    yv = M["val"].presence.values
    thr_grid = np.linspace(0.02, 0.98, 97)
    f1s = [precision_recall_fscore_support(yv, P["val"] >= t, average="binary", zero_division=0)[2] for t in thr_grid]
    thr = float(thr_grid[int(np.argmax(f1s))])
    res = {"threshold": thr, "threshold_from": "max F1 on validation", "splits": {}}
    for split in ev["splits"]:
        res["splits"][split] = split_metrics(P[split], M[split], thr, ev["smoothing"])
    json.dump(res, open(run / "metrics.json", "w"), indent=1, default=float)

    rows = []
    for split in ev["splits"]:
        m = M[split]
        dec = P[split] >= thr
        for key, g in m[m.presence == 1].groupby(m.species.replace("", "(no species label)")):
            rows.append(dict(split=split, species=key, windows=len(g), recall=round(float(dec[g.index].mean()), 3)))
    pd.DataFrame(rows).to_csv(run / "per_species.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
    ax = axes[0]
    for split in ev["splits"]:
        y = M[split].presence.values
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, P[split])
            ax.plot(fpr, tpr, color=COLORS[split], lw=2, label=f"{LABELS[split]}  AUC {res['splits'][split]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("False positive rate (background windows)", fontsize=8, color=MUTED)
    ax.set_ylabel("True positive rate (mosquito windows)", fontsize=8, color=MUTED)
    style(ax, "ROC per split")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[1]
    sweep = np.r_[np.linspace(0.05, 0.95, 37), 0.97, 0.98, 0.99, 0.995, 0.998]
    for split in ["test", "test_A", "test_B"]:
        if split not in P:
            continue
        m = M[split]
        pts = []
        for t in sweep:
            d = decisions(P[split], m, t, *ev["smoothing"])
            pts.append((false_alarms_per_hour(d, m)[0], d[(m.presence == 1).values].mean()))
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS[split], lw=2, marker="o", ms=3, label=LABELS[split])
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(f"False alarms per hour of background ({ev['smoothing'][0]}-of-{ev['smoothing'][1]} smoothing)", fontsize=8, color=MUTED)
    ax.set_ylabel("Mosquito recall (share of mosquito windows detected)", fontsize=8, color=MUTED)
    style(ax, "Operating curve: threshold sweep")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle(f"{cfg['name']}: mosquito detection (threshold {thr:.2f} from validation)", x=0.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(run / "detection.png", dpi=120, facecolor=SURF)

    log(f"threshold (max F1 on validation): {thr:.2f}")
    tab = []
    for split, r in res["splits"].items():
        s = r[f"{ev['smoothing'][0]}of{ev['smoothing'][1]}"]
        tab.append(dict(split=split, roc_auc=r.get("roc_auc", np.nan), precision=r["raw"]["precision"], recall=r["raw"]["recall"],
                        f1=r["raw"]["f1"], fa_per_h_raw=r["raw"]["false_alarms_per_hour"], fa_per_h_smoothed=s["false_alarms_per_hour"],
                        recall_smoothed=s["recall"], bg_hours=r["raw"]["background_hours"]))
    log("\n" + pd.DataFrame(tab).round(3).to_string(index=False))
    log("\nrecall per species (share of mosquito windows detected, at the validation threshold):")
    log("\n" + pd.DataFrame(rows).pivot(index="species", columns="split", values="recall").to_string())

    # fixed operating points with k-of-n smoothing: what a device would actually use
    ops = []
    for split in [s for s in ("val", "test", "test_A", "test_B") if s in P]:
        m = M[split]
        for t in (0.5, 0.9, 0.98):
            d = decisions(P[split], m, t, *ev["smoothing"])
            ops.append(dict(split=split, threshold=t, recall=round(float(d[(m.presence == 1).values].mean()), 3),
                            false_alarms_per_h=round(false_alarms_per_hour(d, m)[0], 1)))
    ops = pd.DataFrame(ops)
    ops.to_csv(run / "operating_points.csv", index=False)
    log(f"\noperating points ({ev['smoothing'][0]}-of-{ev['smoothing'][1]} smoothing): recall / false alarms per hour")
    log("\n" + ops.pivot(index="threshold", columns="split", values=["recall", "false_alarms_per_h"]).to_string())
    log(f"\nfigure: {run / 'detection.png'}")
    return res


def main():
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
