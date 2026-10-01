"""Regroup a calibration bank by sex: labelled entries keep their label; unlabelled / mixed-cage entries get a sex
from their measured f0 (`sex_from_f0` in the config) or stay `unassigned` (still used as noise, never as a tone
source). Every entry gets `sex_source`: label | f0 | none.

Usage: python -m src.synth.assign_sex --config configs/synth_mumbai.yaml
"""
import argparse
import json

import pandas as pd
import yaml


def assign(entry, sex_label, rule):
    if sex_label in ("female", "male"):
        return sex_label, "label"
    if rule == "labels_only" or rule is None:
        return "unassigned", "none"
    a, b = rule
    if entry["f0"] <= a:
        return "female", "f0"
    if entry["f0"] >= b:
        return "male", "f0"
    return "unassigned", "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_mumbai.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    path = cfg["calibration_file"]
    bank = json.load(open(path))
    out, rows = {}, []
    for cls, by_sex in bank.items():
        rule = cfg.get("sex_from_f0", {}).get(cls)
        for sex_label, entries in by_sex.items():
            for e in entries:
                sex, src = assign(e, sex_label, rule)
                e = dict(e, sex_label=sex_label, sex_source=src)
                out.setdefault(cls, {}).setdefault(sex, []).append(e)
                rows.append(dict(cls=cls, sex=sex, source=src, f0=e["f0"]))
    json.dump(out, open(path, "w"), indent=1)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index="cls", columns=["sex", "source"], values="f0", aggfunc="size", fill_value=0))
    print(df.groupby(["cls", "sex"]).f0.median().unstack().round(0))
    print(f"rewrote {path}")


if __name__ == "__main__":
    main()
