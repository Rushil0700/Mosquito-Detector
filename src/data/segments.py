"""M2: master list of labelled REAL audio segments -> data/processed/segments.csv

One row per segment (start_s, end_s inside a file) with labels:
  presence  1 mosquito / 0 background or other sound
  species   ae_aegypti | ae_albopictus | an_stephensi | cx_quinquefasciatus | other_mosquito | '' (ignored)
            Cx. pipiens / "pipiens complex" are ignored for species: the complex includes Cx. quinquefasciatus,
            so they can't be used as either class. Mosquito segments without a species label are ignored too.
  genus     Aedes | Anopheles | Culex | Culiseta | Mansonia | ''
  sex       female | male | '' (labels only; no f0 inference on real data)
Sources:
  HumBugDB  every clip as one segment (expert-labelled segments)
  Abuzz     mosquito recordings: 1 s windows that pass the calibration tone gate, merged into runs (the rest of the
            file is excluded - neither mosquito nor background); background recordings as whole segments
  UFRGS     external test only (split fixed to `external`)
  Funchal   not used (0.3 s clips are shorter than the 1 s model window)

Usage: python -m src.data.segments [--workers 8]
"""
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import pandas as pd
import yaml

from src.synth.calibrate import analyse_clip, plausible_range

TARGET = ["ae_aegypti", "ae_albopictus", "an_stephensi", "cx_quinquefasciatus"]
IGNORE_SPECIES = {"cx_pipiens", "cx_pipiens_complex", "unknown", "none", ""}
SR = 8000


def species_label(cls):
    if cls in TARGET:
        return cls
    if cls in IGNORE_SPECIES or pd.isna(cls):
        return ""
    return "other_mosquito"


def _gate_file(job):
    """Seconds of an Abuzz recording that contain a clear flight tone -> list of (start, end) runs."""
    warnings.filterwarnings("ignore")
    path, lo, hi = job
    x, _ = librosa.load(path, sr=SR)
    ok = []
    for k in range(len(x) // SR):
        w = x[k * SR:(k + 1) * SR]
        ok.append(np.std(w) > 0 and analyse_clip(w, SR, f0_range=(lo, hi), check_harmonics=True) is not None)
    runs, start = [], None
    for k, v in enumerate(ok + [False]):
        if v and start is None:
            start = k
        elif not v and start is not None:
            runs.append((float(start), float(k)))
            start = None
    return runs, len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_mumbai.yaml")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    priors = yaml.safe_load(open(args.config))["priors"]
    m = pd.read_csv("data/processed/manifest_files.csv")
    m["cls"] = m.cls.fillna("")
    base = lambda r, s, e, presence: dict(
        path=r.path, start_s=s, end_s=e, dataset=r.dataset, site=r.site, device=r.device, session=r.session,
        orig_sr=r.orig_sr, presence=presence, species=species_label(r.cls) if presence else "",
        species_raw=r.cls if presence else "", genus=r.genus if presence and r.genus != "none" else "",
        sex=r.sex if presence and r.sex in ("female", "male") else "")

    rows = []
    h = m[m.dataset == "humbugdb"]
    for r in h.itertuples():
        rows.append(base(r, 0.0, float(r.duration_s), int(r.sound_type == "mosquito")))

    a = m[m.dataset == "abuzz"]
    for r in a[a.sound_type == "background"].itertuples():
        rows.append(base(r, 0.0, float(r.duration_s), 0))
    am = a[a.sound_type == "mosquito"]
    jobs = [(p, *plausible_range(priors, c, "unknown")) if c in priors else (p, 300.0, 1000.0) for p, c in zip(am.path, am.cls)]
    with ProcessPoolExecutor(args.workers) as ex:
        gated = list(ex.map(_gate_file, jobs))
    kept = tot = 0
    for r, (runs, n) in zip(am.itertuples(), gated):
        tot += n
        for s, e in runs:
            kept += e - s
            rows.append(base(r, s, e, 1))

    u = m[m.dataset == "ufrgs"]
    for r in u.itertuples():
        rows.append(base(r, 0.0, float(r.duration_s), int(r.sound_type == "mosquito")))

    seg = pd.DataFrame(rows)
    seg["dur"] = seg.end_s - seg.start_s
    seg.to_csv("data/processed/segments.csv", index=False)
    print(f"Abuzz mosquito recordings: {kept / 60:.0f} of {tot / 60:.0f} min pass the tone gate ({100 * kept / max(tot, 1):.0f} %)")
    pd.set_option("display.width", 200)
    print((seg.groupby(["dataset", "presence"]).dur.sum() / 3600).round(2).unstack())
    mos = seg[seg.presence == 1]
    print("\nspecies label, minutes:")
    print((mos.pivot_table(index="species", columns="dataset", values="dur", aggfunc="sum").fillna(0) / 60).round(1))
    print(f"\nwrote data/processed/segments.csv ({len(seg)} segments)")


if __name__ == "__main__":
    main()
