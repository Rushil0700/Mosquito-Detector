"""Build the unified file-level manifest over all datasets -> data/processed/manifest_files.csv

One row per audio FILE (Abuzz files are later cut into flight segments by src/data/activity.py).
Columns: path, dataset, sound_type (mosquito/background/other_audio), species_raw, cls, genus, sex
(female/male/mixed/unknown), site, device, session, orig_sr, duration_s.

Session = the unit that must never be split across train/val/test:
  HumBugDB  place | device | date
  Abuzz     site | species-recording (all phones in a folder set recorded the SAME cages at the same time,
            so the phone is NOT an independent session; background files join their species' session)
  Funchal   species | sample-rate folder | contiguous index block (numbered segments of longer recordings)
  UFRGS     recording date (from filename)

Usage: python -m src.data.manifest
"""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import soundfile as sf

RAW = Path("data/raw")
OUT = Path("data/processed/manifest_files.csv")

# species label -> (cls, genus). cls groups strains / sensu-lato labels; species_raw keeps the original.
SPECIES = {
    "ae aegypti": "ae_aegypti", "aedes aegypti": "ae_aegypti",
    "ae albopictus": "ae_albopictus", "aedes albopictus": "ae_albopictus",
    "aedes mediovittatus": "ae_mediovittatus", "aedes sierrensis": "ae_sierrensis",
    "an arabiensis": "an_arabiensis", "anopheles arabiensis dongola": "an_arabiensis", "anopheles arabiensis rufisque": "an_arabiensis",
    "an gambiae ss": "an_gambiae", "an gambiae sl": "an_gambiae", "an gambiae": "an_gambiae", "an coluzzii": "an_gambiae",
    "anopheles gambiae kisumu": "an_gambiae", "anopheles gambiae akron": "an_gambiae", "anopheles gambiae rsp": "an_gambiae",
    "an funestus ss": "an_funestus", "an funestus sl": "an_funestus", "an funestus": "an_funestus",
    "an stephensi": "an_stephensi", "anopheles stephensi": "an_stephensi",
    "an dirus": "an_dirus", "anopheles dirus": "an_dirus",
    "an minimus": "an_minimus", "anopheles minimus": "an_minimus",
    "an farauti": "an_farauti", "anopheles farauti": "an_farauti",
    "an freeborni": "an_freeborni", "anopheles freeborni": "an_freeborni",
    "an albimanus": "an_albimanus", "anopheles albimanus": "an_albimanus",
    "an atroparvus": "an_atroparvus", "anopheles atroparvus": "an_atroparvus",
    "an quadriannulatus": "an_quadriannulatus", "anopheles quadriannulatus": "an_quadriannulatus",
    "anopheles quadrimaculatus": "an_quadrimaculatus", "an merus": "an_merus", "anopheles merus": "an_merus",
    "culex quinquefasciatus": "cx_quinquefasciatus", "culex pipiens": "cx_pipiens",
    "culex pipiens complex": "cx_pipiens_complex", "culex tarsalis": "cx_tarsalis", "culex tigripes": "cx_tigripes",
    "culiseta incidens": "cs_incidens", "ma uniformis": "ma_uniformis", "ma africanus": "ma_africanus",
}
GENUS = {"ae": "Aedes", "an": "Anopheles", "cx": "Culex", "cs": "Culiseta", "ma": "Mansonia"}


def cls_of(species_raw):
    s = str(species_raw).strip().lower()
    if s in SPECIES:
        return SPECIES[s]
    if s.startswith("an "):
        return "an_" + s.split()[1]
    return "unknown" if s in ("", "nan") else s.replace(" ", "_")


def genus_of(cls):
    return GENUS.get(cls.split("_")[0], "unknown")


# ---------------- Abuzz filename parsing ----------------
ABBREV = sorted(["as", "aste", "ad", "ag", "agam", "aalb", "afar", "afree", "af", "am", "amin", "aegy", "albo", "ebro",
                 "dongola", "rufisque", "gambiae", "akron", "sanqua", "cxt", "cuincidens", "ma", "quad", "quadri", "cxq",
                 "cxp", "aela", "aepr", "rsp", "albopictus", "medio", "medi", "ae"], key=len, reverse=True)
F_SUFFIX, M_SUFFIX, MIX_SUFFIX = {"f", "ft"}, {"m", "mt"}, {"mf", "fm"}


def abuzz_sex(stem):
    """female / male / mixed / unknown from Abuzz filename conventions (e.g. AsteF, AdM, _F, asm1, ag1ft, 'ebro f')."""
    s = stem.lower()
    if "mix" in s or "swarm" in s:
        return "mixed"
    found = set()
    for tok in re.findall(r"[a-z]+", s):
        if tok in F_SUFFIX or tok.startswith("fthet"):
            found.add("female")
            continue
        if tok in M_SUFFIX:
            found.add("male")
            continue
        if tok in MIX_SUFFIX:
            found.add("mixed")
            continue
        for ab in ABBREV:
            if tok.startswith(ab) and len(tok) > len(ab):
                rest = tok[len(ab):]
                if rest in F_SUFFIX or rest.startswith("fthet"):
                    found.add("female")
                elif rest in M_SUFFIX:
                    found.add("male")
                elif rest in MIX_SUFFIX:
                    found.add("mixed")
                break
    if "mixed" in found or {"female", "male"} <= found:
        return "mixed"
    return found.pop() if found else "unknown"


PHONE_ONLY = {"GoogleNexus", "Huawei866C", "Nokia5235", "Nokia6555B", "SamsungT209", "SonyXperia", "iPhone4S", "iPhone4C", "iPhone6"}


def abuzz_site_device(species_dir, folder):
    if folder in PHONE_ONLY:
        return "Abuzz_multiphone_2016-03", folder
    if folder == "-":
        if species_dir == "Anopheles merus":        # Voice0151-0155 continue the CDC_T209 numbering (Nov 2016)
            return "CDC", "T209"
        return "field_citizen", "unknown"
    if folder == "Snipped":
        return "field_citizen", "iPhone4S"
    site, _, device = folder.rpartition("_")
    return site, device


def probe(path):
    try:
        i = sf.info(path)
        return i.samplerate, i.duration
    except Exception:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                           capture_output=True, text=True)
        d = json.loads(r.stdout)
        a = [s for s in d["streams"] if s["codec_type"] == "audio"][0]
        return int(a["sample_rate"]), float(d["format"]["duration"])


def build_abuzz():
    rows = []
    root = RAW / "abuzz/audio"
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".wav", ".m4a", ".amr", ".mp4"}:
            continue
        rel = p.relative_to(root).parts
        species_dir, folder = rel[0], rel[1] if len(rel) > 2 else "-"
        site, device = abuzz_site_device(species_dir, folder)
        is_bg = re.search(r"background|\bbg\b", p.stem.lower()) is not None
        cls = cls_of(species_dir)
        sr, dur = probe(p)
        rows.append(dict(path=str(p), dataset="abuzz", sound_type="background" if is_bg else "mosquito",
                         species_raw=species_dir, cls="none" if is_bg else cls, genus="none" if is_bg else genus_of(cls),
                         sex="none" if is_bg else abuzz_sex(p.stem), site=site, device=device,
                         session=f"abuzz|{site}|{species_dir}", orig_sr=sr, duration_s=round(dur, 3),
                         stem_key=f"{species_dir}/{folder}/{p.stem.strip().lower()}", ext=p.suffix.lower()))
    df = pd.DataFrame(rows)
    # The same recording often exists twice (original phone codec + a WAV conversion): keep the original codec.
    pref = {".amr": 0, ".m4a": 0, ".mp4": 0, ".wav": 1}
    df["_p"] = df.ext.map(pref)
    before = len(df)
    df = df.sort_values(["stem_key", "_p"]).drop_duplicates("stem_key", keep="first").drop(columns=["_p", "ext", "stem_key"])
    print(f"abuzz: {before} files, {before - len(df)} same-recording duplicates (codec + wav) removed -> {len(df)}")
    return df


def build_humbugdb():
    md = pd.read_csv("data/metadata/neurips_2021_zenodo_0_0_1.csv")
    date = pd.to_datetime(md.record_datetime, format="%d-%m-%y %H:%M").dt.date.astype(str)
    device = md.device_type.fillna(md.mic_type)
    st = md.sound_type.map({"mosquito": "mosquito", "background": "background", "audio": "other_audio"})
    cls = [cls_of(s) if t == "mosquito" else "none" for s, t in zip(md.species, st)]
    sex = [("none" if t != "mosquito" else {"Female": "female", "Male": "male"}.get(g, "unknown")) for g, t in zip(md.gender, st)]
    return pd.DataFrame(dict(
        path=[f"data/raw/humbugdb/audio/{i}.wav" for i in md.id], dataset="humbugdb", sound_type=st,
        species_raw=md.species.fillna(""), cls=cls, genus=[genus_of(c) if c != "none" else "none" for c in cls], sex=sex,
        site=md.place, device=device, session="humbugdb|" + md.place + "|" + device + "|" + date,
        orig_sr=md.sample_rate, duration_s=md.length.round(3)))


def build_funchal():
    f = pd.read_csv("data/processed/manifest_funchal.csv")  # from src.data.manifest_funchal (block splits)
    blocks = {"calib": 0, "tune": 1, "test": 2}
    return pd.DataFrame(dict(
        path=f.path, dataset="funchal", sound_type="mosquito", species_raw=f.cls, cls=f.cls, genus=f.cls.map(genus_of),
        sex="unknown", site="Funchal_NHM", device=f.group, session="funchal|" + f.cls + "|" + f.group + "|block" + f.split.map(blocks).astype(str),
        orig_sr=f.orig_sr, duration_s=[round(sf.info(p).duration, 3) for p in f.path]))


def build_ufrgs():
    rows = []
    for p in sorted((RAW / "ufrgs/repo/dataset").rglob("*.wav")):
        folder = p.parent.name
        mosq = folder.startswith("Aedes")
        i = sf.info(p)
        date = re.search(r"_(\d{6})_", p.name).group(1)
        rows.append(dict(path=str(p), dataset="ufrgs", sound_type="mosquito" if mosq else "other_audio",
                         species_raw=folder, cls="ae_aegypti" if mosq else "none", genus="Aedes" if mosq else "none",
                         sex=("female" if folder.endswith("female") else "male") if mosq else "none",
                         site="UFRGS", device="smartphone", session=f"ufrgs|{date}", orig_sr=i.samplerate, duration_s=round(i.duration, 3)))
    return pd.DataFrame(rows)


def main():
    df = pd.concat([build_humbugdb(), build_abuzz(), build_funchal(), build_ufrgs()], ignore_index=True)
    df = df[df.duration_s > 0.05]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["dataset", "sound_type"]).agg(files=("path", "size"), hours=("duration_s", lambda x: round(x.sum() / 3600, 2)),
                                                      sessions=("session", "nunique")))
    ab = df[(df.dataset == "abuzz") & (df.sound_type == "mosquito")]
    print("\nAbuzz sex from filenames (minutes):")
    print((ab.pivot_table(index="cls", columns="sex", values="duration_s", aggfunc="sum").fillna(0) / 60).round(1))
    print(f"\nwrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
