"""M1: summarise HumBugDB metadata (hours per class, species x site, sessions).

Usage: python src/inspect_metadata.py [path/to/neurips_2021_zenodo_0_0_1.csv]
"""
import sys

import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/metadata/neurips_2021_zenodo_0_0_1.csv"


def hours(s):
    return round(s.sum() / 3600, 2)


def add_session(df):
    """Session = site + recording device + calendar date (leakage-safe grouping key)."""
    date = pd.to_datetime(df.record_datetime, format="%d-%m-%y %H:%M").dt.date.astype(str)
    df["session"] = df.place + "|" + df.device_type.fillna(df.mic_type) + "|" + date
    return df


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    df = add_session(pd.read_csv(CSV))
    m = df[df.sound_type == "mosquito"]

    print("== hours by sound_type\n", df.groupby("sound_type").length.agg(["count", hours]))
    print("\n== hours by sample_rate x sound_type\n", df.groupby(["sample_rate", "sound_type"]).length.agg(hours))
    print("\n== hours by device\n", df.groupby(["mic_type", "device_type", "sound_type"], dropna=False).length.agg(hours))
    print("\n== mosquito hours by species\n", m.groupby("species", dropna=False).length.agg(["count", hours]).sort_values("hours", ascending=False))
    print("\n== mosquito hours by sex\n", m.groupby("gender", dropna=False).length.agg(["count", hours]))
    print("\n== species x place (minutes)\n", (m.pivot_table(index="species", columns="place", values="length", aggfunc="sum").fillna(0) / 60).round(1))
    print("\n== sessions:", df.session.nunique(), "| parent recordings:", df.name.nunique())


if __name__ == "__main__":
    main()
