"""Build the Funchal manifest with original sample rate and a contiguous-block split.

Funchal files are numbered segments (e.g. aedes_aegypti_0..1697), likely cut from longer recordings,
so neighbouring indices may share a recording. Splitting by contiguous index blocks within each
(class, sample-rate folder) keeps neighbours together: first 50 % calib, next 25 % tune, last 25 % test.

Usage: python -m src.data.manifest_funchal
"""
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path("data/raw/funchal/Mosquitos Audio Samples")
FOLDERS = {"Aedes Aegypti": "ae_aegypti", "Culex Pipiens & Quinquefasctiatus": "cx_quinquefasciatus"}
SPLITS = [("calib", 0.50), ("tune", 0.75), ("test", 1.0)]


def main():
    rows = []
    for top, cls in FOLDERS.items():
        for sub in sorted((ROOT / top).iterdir()):
            if not sub.is_dir() or sub.name.startswith("Mix"):  # Mix = pipiens+quinquefasciatus, ambiguous label
                continue
            files = sorted(sub.glob("*.wav"), key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
            for i, p in enumerate(files):
                frac = (i + 0.5) / len(files)
                split = next(name for name, edge in SPLITS if frac < edge)
                rows.append(dict(path=str(p), cls=cls, sex="unknown", dataset="funchal",
                                 orig_sr=sf.info(p).samplerate, group=sub.name, split=split))
    df = pd.DataFrame(rows)
    out = Path("data/processed/manifest_funchal.csv")
    df.to_csv(out, index=False)
    print(df.groupby(["cls", "orig_sr", "split"]).size().unstack())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
