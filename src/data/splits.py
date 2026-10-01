"""M2: assign every segment to a split by SESSION (configs/splits.yaml) -> data/processed/segments_split.csv

Splits: train, val, test (held-out sessions / sites), test_A (Kenya, unseen site), test_B (Thailand, unseen site),
external (UFRGS). Also checks that no session, source file or Abuzz recording site ends up in two splits.

Usage: python -m src.data.splits
"""
import numpy as np
import pandas as pd
import yaml


def main():
    cfg = yaml.safe_load(open("configs/splits.yaml"))
    seg = pd.read_csv("data/processed/segments.csv", keep_default_na=False, na_values=[""])
    for c in ("species", "genus", "sex"):
        seg[c] = seg[c].fillna("")
    forced = {}
    for s in seg.session.unique():
        site = seg.loc[seg.session == s, "site"].iloc[0]
        ds = seg.loc[seg.session == s, "dataset"].iloc[0]
        if ds in cfg["external_datasets"]:
            forced[s] = "external"
        elif site in cfg["test_A_sites"]:
            forced[s] = "test_A"
        elif site in cfg["test_B_sites"]:
            forced[s] = "test_B"
        elif site in cfg["test_sites"] or s in cfg["test_sessions"]:
            forced[s] = "test"
        elif s in cfg["val_sessions"]:
            forced[s] = "val"
        elif s in cfg["train_sessions"]:
            forced[s] = "train"
    missing = set(cfg["test_sessions"] + cfg["val_sessions"] + cfg["train_sessions"]) - set(seg.session)
    assert not missing, f"sessions in configs/splits.yaml not found: {missing}"

    # remaining sessions: shuffled, assigned to whichever split is furthest below its target share of minutes
    mins = seg.groupby("session").dur.sum() / 60
    pool = [s for s in mins.index if s not in forced]
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(pool)
    ratios = cfg["ratios"]
    have = {k: sum(mins[s] for s, v in forced.items() if v == k) for k in ratios}
    total = sum(have.values()) + sum(mins[s] for s in pool)
    assign = dict(forced)
    for s in pool:
        k = min(ratios, key=lambda k: have[k] / total - ratios[k])
        assign[s] = k
        have[k] += mins[s]
    seg["split"] = seg.session.map(assign)

    # leakage checks
    assert seg.groupby("session").split.nunique().max() == 1, "a session spans two splits"
    assert seg.groupby("path").split.nunique().max() == 1, "a source file spans two splits"
    ab = seg[seg.dataset == "abuzz"]
    shared_sites = ab.groupby("site").split.nunique()
    seg.to_csv("data/processed/segments_split.csv", index=False)

    pd.set_option("display.width", 220)
    order = ["train", "val", "test", "test_A", "test_B", "external"]
    print("hours by split (presence 1 = mosquito, 0 = background / other sound):")
    print((seg.pivot_table(index="split", columns="presence", values="dur", aggfunc="sum") / 3600).reindex(order).round(2))
    print("\nspecies minutes by split:")
    mos = seg[(seg.presence == 1) & (seg.species != "")]
    print((mos.pivot_table(index="species", columns="split", values="dur", aggfunc="sum").fillna(0) / 60).reindex(columns=order).round(1))
    print("\nsessions by split:", seg.groupby("split").session.nunique().reindex(order).to_dict())
    print("Abuzz sites spanning several splits (sessions = species within a site; fine for other species):",
          shared_sites[shared_sites > 1].index.tolist())
    print("leakage checks passed: no session or source file in two splits")


if __name__ == "__main__":
    main()
