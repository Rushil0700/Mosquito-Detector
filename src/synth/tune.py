"""Tune synthesizer knobs per class against the realism check (Optuna TPE, all knobs free at once).

Score = fair AUC (synthetic vs real) on the `tune` split only, averaged over >= 3 seeds (real-clip sample
+ synthesis RNG); the `test` split is reserved for the final report. With one seed the AUC moves by about
+-0.025, which a search mistakes for improvements, so:
  * every trial is scored on --seeds (default 3);
  * the best trial and the current config are re-scored on --check-seeds (fresh seeds), and the new
    settings are kept only if they beat the current ones by more than --min-gain (default 0.02).
Workers share one SQLite study, so several processes can search in parallel.

Usage: python -m src.synth.tune --real data/processed/manifest_funchal.csv --cls ae_aegypti [--trials 120 --workers 4 --write]
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def suggest(trial):
    """Search space: one per-class override block (tuning / flight / room / source)."""
    t = {
        "drift_from": trial.suggest_categorical("drift_from", ["sharpness", "track_cv"]),
        "drift_scale": trial.suggest_float("drift_scale", 0.3, 3.0, log=True),
        "drift_tau_s": trial.suggest_float("drift_tau_s", 0.005, 0.15, log=True),
        "am_scale": trial.suggest_float("am_scale", 0.1, 2.5, log=True),
        "am_tau_s": trial.suggest_float("am_tau_s", 0.01, 0.2, log=True),
        "harmonic_jitter_db": trial.suggest_float("harmonic_jitter_db", 0.0, 4.0),
        "fast_fm_scale": trial.suggest_float("fast_fm_scale", 0.0, 1.5),
        "fast_am_scale": trial.suggest_float("fast_am_scale", 0.0, 1.5),
        "fast_bw_hz": trial.suggest_float("fast_bw_hz", 10.0, 150.0, log=True),
        "harmonic_tilt_db": trial.suggest_float("harmonic_tilt_db", -1.5, 1.5),
        "snr_jitter_db": trial.suggest_float("snr_jitter_db", 0.0, 5.0),
        "tnr_offset_db": trial.suggest_float("tnr_offset_db", -4.0, 4.0),
    }
    f = {
        "start_distance_m": [0.03, trial.suggest_float("start_distance_max_m", 0.08, 1.0, log=True)],
        "speed_m_s": [0.1, trial.suggest_float("speed_max_m_s", 0.3, 2.0, log=True)],
        "turn_rate": trial.suggest_float("turn_rate", 0.5, 6.0, log=True),
        "flyby_prob": trial.suggest_float("flyby_prob", 0.0, 1.0),
        "flyby_miss_m": trial.suggest_float("flyby_miss_m", 0.02, 0.3, log=True),
    }
    room_on = trial.suggest_categorical("room", [False, True])
    r = {"enabled": room_on}
    if room_on:
        r.update(dims_m=[0.2, trial.suggest_float("room_dims_max_m", 0.3, 4.0, log=True)],
                 reflection_coef=[0.1, trial.suggest_float("reflection_max", 0.2, 0.95)], order=1)
    s = {"harmonic_levels": trial.suggest_categorical("harmonic_levels", ["energy", "peak"])}
    return {"tuning": t, "flight": f, "room": r, "source": s}


def params_from_block(block):
    """Optuna params equivalent to a per-class override block (to enqueue the current config)."""
    t, f, r, s = block["tuning"], block["flight"], block["room"], block["source"]
    p = {k: t[k] for k in ("drift_from", "drift_scale", "drift_tau_s", "am_scale", "am_tau_s",
                           "harmonic_jitter_db", "fast_fm_scale", "fast_am_scale", "fast_bw_hz",
                           "harmonic_tilt_db", "snr_jitter_db", "tnr_offset_db")}
    p.update(start_distance_max_m=float(np.clip(f["start_distance_m"][1], 0.08, 1.0)),
             speed_max_m_s=float(np.clip(f["speed_m_s"][1], 0.3, 2.0)),
             turn_rate=f.get("turn_rate", 2.0), flyby_prob=f.get("flyby_prob", 0.0), flyby_miss_m=f.get("flyby_miss_m", 0.1),
             room=bool(r.get("enabled")), harmonic_levels=s.get("harmonic_levels", "energy"))
    p["drift_scale"] = float(np.clip(p["drift_scale"], 0.3, 3.0))
    p["harmonic_tilt_db"] = float(np.clip(p["harmonic_tilt_db"], -1.5, 1.5))
    if p["room"]:
        p.update(room_dims_max_m=r["dims_m"][1], reflection_max=r["reflection_coef"][1])
    return p


def current_block(cfg, cls):
    from src.synth.generate import class_cfg
    c = class_cfg(cfg, cls)
    return {k: copy.deepcopy(c[k]) for k in ("tuning", "flight", "room", "source")}


def with_block(cfg, cls, block):
    out = copy.deepcopy(cfg)
    sp = out.setdefault("species", {}).setdefault(cls, {}) or {}
    for k, v in block.items():
        sp[k] = {**sp.get(k, {}), **v}
    out["species"][cls] = sp
    return out


class Scorer:
    def __init__(self, args):
        from src.synth.generate import NoiseBank
        self.args = args
        self.cfg = yaml.safe_load(open(args.config))
        self.cfg.setdefault("eq", {})["enabled"] = False  # EQ is fitted after tuning
        self.bank = json.load(open(self.cfg["calibration_file"]))
        self.nb = NoiseBank(self.bank, self.cfg["sample_rate"], self.cfg["noise"]["pool_by"])
        man = pd.read_csv(args.real)
        self.real = man[man.split == "tune"]

    def __call__(self, cfg, seeds):
        from src.synth.validate import compare
        aucs = []
        for seed in seeds:
            res, _ = compare(self.real, cfg, self.bank, self.nb, self.args.n, np.random.default_rng(seed), seed=seed,
                             classes=[self.args.cls])
            aucs.append(res[0]["auc_fair"])
        return float(np.mean(aucs)), aucs


def worker(args, n_trials, wid):
    import optuna
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scorer = Scorer(args)
    study = optuna.load_study(study_name=args.name, storage=args.storage,
                              sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=1000 + wid,
                                                                 n_startup_trials=20))

    def objective(trial):
        block = suggest(trial)
        try:
            sc, per = scorer(with_block(scorer.cfg, args.cls, block), args.seeds)
        except Exception as e:  # a pathological setting must not kill the search
            print(f"[w{wid}] trial {trial.number} failed: {e!r}", flush=True)
            return 1.0
        trial.set_user_attr("per_seed", per)
        print(f"[w{wid}] trial {trial.number}: {sc:.3f} {np.round(per, 3).tolist()}", flush=True)
        return sc

    study.optimize(objective, n_trials=n_trials)


def main():
    import optuna
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--cls", required=True)
    ap.add_argument("--config", default="configs/synth.yaml")
    ap.add_argument("--n", type=int, default=300, help="real clips per evaluation (synthetic: same number)")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 13])
    ap.add_argument("--check-seeds", type=int, nargs="+", default=[17, 19, 23, 29, 31])
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--write", action="store_true", help="write the accepted block into the config")
    ap.add_argument("--tag", default="", help="suffix for the study / output files (separate searches)")
    args = ap.parse_args()
    Path("results/tuning").mkdir(parents=True, exist_ok=True)
    args.name = args.cls + (f"_{args.tag}" if args.tag else "")
    args.storage = f"sqlite:///results/tuning/optuna_{args.name}.db"

    cfg = yaml.safe_load(open(args.config))
    base_block = current_block(cfg, args.cls)
    study = optuna.create_study(study_name=args.name, storage=args.storage, direction="minimize", load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(params_from_block(base_block))  # trial 0 = current settings
    per_worker = int(np.ceil(args.trials / args.workers))
    procs = [mp.get_context("spawn").Process(target=worker, args=(args, per_worker, w)) for w in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    study = optuna.load_study(study_name=args.name, storage=args.storage)
    df = study.trials_dataframe()
    df.to_csv(f"results/tuning/trials_{args.name}.csv", index=False)
    best = study.best_trial
    print(f"\nbest trial {best.number}: {best.value:.3f} on seeds {args.seeds}")

    # Re-check on fresh seeds: best vs current (both scored identically), keep only a clear gain.
    scorer = Scorer(args)
    best_block = suggest(optuna.trial.FixedTrial(best.params))
    base_sc, base_per = scorer(with_block(scorer.cfg, args.cls, base_block), args.check_seeds)
    best_sc, best_per = scorer(with_block(scorer.cfg, args.cls, best_block), args.check_seeds)
    gain = base_sc - best_sc
    keep = gain > args.min_gain
    print(f"check seeds {args.check_seeds}: current {base_sc:.3f} {np.round(base_per, 3).tolist()}  "
          f"best {best_sc:.3f} {np.round(best_per, 3).tolist()}  gain {gain:+.3f} -> {'KEEP' if keep else 'reject'}")
    summary = dict(cls=args.cls, n_trials=len(df), best_trial=best.number, search_score=best.value,
                   check_current=base_sc, check_best=best_sc, gain=gain, kept=keep, block=best_block)
    json.dump(summary, open(f"results/tuning/summary_{args.name}.json", "w"), indent=1, default=str)
    print(yaml.safe_dump({"species": {args.cls: best_block}}, sort_keys=False))
    if keep and args.write:
        cfg = yaml.safe_load(open(args.config))
        cfg["species"][args.cls] = best_block
        write_species_block(args.config, args.cls, best_block)
        print(f"wrote species.{args.cls} to {args.config}")


def write_species_block(path, cls, block):
    """Replace `species.<cls>` in the YAML file, keeping the rest of the file (and its comments) intact."""
    lines = open(path).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.rstrip() == "species:")
    end = start + 1
    while end < len(lines) and (lines[end].startswith("  ") or not lines[end].strip()):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    cur = yaml.safe_load("\n".join(lines[start:end]))["species"] or {}
    rnd = lambda v: {k: rnd(x) for k, x in v.items()} if isinstance(v, dict) else (
        [rnd(x) for x in v] if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v))
    cur[cls] = rnd(block)
    dumped = yaml.safe_dump({"species": cur}, sort_keys=False, default_flow_style=None, width=120).rstrip().split("\n")
    open(path, "w").write("\n".join(lines[:start] + dumped + lines[end:]))


if __name__ == "__main__":
    main()
