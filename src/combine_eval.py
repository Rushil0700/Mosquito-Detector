"""Combined species decision from two trained M4 models (no training).

  4-species model  -> names the species (Ae. aegypti / Ae. albopictus / An. stephensi / Cx. quinquefasciatus)
  5-class model    -> its "other mosquito" probability vetoes species it was not trained to name

Rules compared on REAL mosquito windows (val and test):
  4-class alone    forced answer from the 4-species model
  5-class alone    the 5-class model on its own ("other" is one of its answers)
  product          P = [(1 - p_other) * P4, p_other], answer = argmax (no tuned parameter)
  veto @ t         "other / uncertain" if p_other >= t, else argmax P4; t chosen on VAL, then reported on TEST

    python -m src.combine_eval
    python -m src.combine_eval --four results/m4_runs/<4-class run> --five results/m4_runs/<5-class run>
Writes <out>/run.log and <out>/combined.csv.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.evaluate_species import NAMES
from src.models import DEFAULT_SPECIES  # noqa: F401  (registers SpectralWhitening for model loading)

TARGETS = ["ae_aegypti", "ae_albopictus", "an_stephensi", "cx_quinquefasciatus"]
OTHER = len(TARGETS)  # answer index for "other mosquito / uncertain"
log = logging.getLogger("combine")


def setup_logging(out):
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log")):
        h.setFormatter(fmt)
        log.addHandler(h)


def check_classes(run, n):
    cls = yaml.safe_load(open(Path(run) / "config.yaml")).get("species_classes") or DEFAULT_SPECIES
    assert cls[:4] == TARGETS and len(cls) == n, f"{run}: unexpected species classes {cls}"


def predict(run, X):
    model = tf.keras.models.load_model(Path(run) / "model.keras")
    return model.predict(X, batch_size=1024, verbose=0)["species"]


def load(split, fdir, four, five):
    M = pd.read_csv(Path(fdir) / f"{split}_meta.csv", keep_default_na=False)
    keep = np.flatnonzero(((M.presence == 1) & M.species.isin(TARGETS + ["other_mosquito"])).values)
    X = np.load(Path(fdir) / f"{split}.npy", mmap_mode="r")[keep].astype(np.float32)
    M = M.iloc[keep].reset_index(drop=True)
    y = M.species.map({s: i for i, s in enumerate(TARGETS)}).fillna(OTHER).astype(int).values
    return M, y, predict(four, X), predict(five, X)


def rules(P4, P5, t):
    po = P5[:, OTHER]
    comb = np.column_stack([(1 - po)[:, None] * P4, po])
    return {"4-class alone": P4.argmax(1),
            "5-class alone": P5.argmax(1),
            "product": comb.argmax(1),
            f"veto @ {t:.2f}": np.where(po >= t, OTHER, P4.argmax(1))}


def score(y, yhat):
    """Recall per group (4 targets + other); 'balanced' = mean over groups present; false-aegypti = other called Ae. aegypti."""
    r = {NAMES[s]: (yhat[y == i] == i).mean() if (y == i).any() else np.nan for i, s in enumerate(TARGETS)}
    o = y == OTHER
    r["other → rejected"] = (yhat[o] == OTHER).mean()
    r["balanced"] = np.nanmean(list(r.values()))
    t = y != OTHER
    r["targets answered"] = (yhat[t] != OTHER).mean()
    r["correct when answered"] = (yhat[t] == y[t])[yhat[t] != OTHER].mean()
    r["genus correct"] = np.mean([TARGETS[a][:2] == TARGETS[b][:2] for a, b in zip(y[t], yhat[t]) if b != OTHER])
    r["other → 'Ae. aegypti'"] = (yhat[o] == 0).mean()
    return r


def by_recording(M, y, P4, P5):
    g = M.groupby("src").indices
    idx = list(g.values())
    return (np.array([y[i[0]] for i in idx]), np.array([P4[i].mean(0) for i in idx]), np.array([P5[i].mean(0) for i in idx]))


def table(y, P4, P5, t):
    return pd.DataFrame({k: score(y, v) for k, v in rules(P4, P5, t).items()}).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--four", default="results/m4_runs/20260929_200213_real_plus_synth")
    ap.add_argument("--five", default="results/m4_runs/20260929_142950_real_plus_synth")
    ap.add_argument("--out", default="results/m4_combined")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out)
    check_classes(args.four, 4)
    check_classes(args.five, 5)
    fdir = yaml.safe_load(open(Path(args.four) / "config.yaml"))["features_dir"]
    pd.set_option("display.width", 220)
    pct = lambda d: d.map(lambda v: "—" if pd.isna(v) else f"{v:.0%}")
    log.info(f"4-species model: {args.four}\n5-class model:   {args.five}")

    Mv, yv, P4v, P5v = load("val", fdir, args.four, args.five)
    Mt, yt, P4t, P5t = load("test", fdir, args.four, args.five)

    # choose the veto threshold on VALIDATION (recording level: the device decides per recording / per event)
    rv = by_recording(Mv, yv, P4v, P5v)
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    bal = [score(rv[0], rules(rv[1], rv[2], t)[f"veto @ {t:.2f}"])["balanced"] for t in grid]
    t = float(grid[int(np.argmax(bal))])
    log.info(f"\nveto threshold chosen on VALIDATION recordings (max balanced recall): p_other >= {t:.2f}")
    log.info("  (validation has only Ae. aegypti and Culex among the targets; albopictus / stephensi are test-only)")

    res = []
    for split, (M, y, P4, P5) in {"val": (Mv, yv, P4v, P5v), "test": (Mt, yt, P4t, P5t)}.items():
        for level, args_ in (("windows", (y, P4, P5)), ("recordings", by_recording(M, y, P4, P5))):
            tab = table(*args_, t)
            n = np.bincount(args_[0], minlength=OTHER + 1)
            log.info(f"\n{split.upper()} — {level} (n: " + ", ".join(f"{NAMES[s]} {n[i]}" for i, s in enumerate(TARGETS))
                     + f", other {n[OTHER]})")
            log.info("\n" + pct(tab).to_string())
            res.append(tab.assign(split=split, level=level))

    # sensitivity: the whole threshold range on TEST windows (for transparency; not used to choose t)
    rows = [dict(threshold=g, **{k: v for k, v in score(yt, rules(P4t, P5t, g)[f"veto @ {g:.2f}"]).items()
                                 if k in ("balanced", "targets answered", "correct when answered", "other → rejected",
                                          "other → 'Ae. aegypti'")}) for g in grid[1::2]]
    log.info("\nTEST windows, veto threshold sweep (sensitivity only):")
    log.info("\n" + pct(pd.DataFrame(rows).set_index("threshold")).to_string())
    pd.concat(res).to_csv(out / "combined.csv")
    log.info(f"\nwrote {out / 'run.log'} and {out / 'combined.csv'}")


if __name__ == "__main__":
    main()
