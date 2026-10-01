"""Download public datasets from Zenodo with resume + size/MD5 verification.

Usage: python -m src.download [humbugdb] [abuzz] [demand]
Retries until every file matches the size and checksum published by Zenodo.
"""
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RECORDS = {
    "humbugdb": ("4904800", Path("data/raw/humbugdb/zips")),
    "abuzz": ("4964774", Path("data/raw/abuzz/rar")),
    "demand": ("1227121", Path("data/noise/public/demand/zips")),   # household / street noise, CC BY 4.0
}
FILE_FILTER = {"demand": lambda k: k.endswith("_16k.zip")}  # 16 kHz versions are enough (resampled to 8 kHz)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, size, checksum, max_rounds=200):
    for _ in range(max_rounds):
        if dest.exists() and dest.stat().st_size == size:
            break
        if dest.exists() and dest.stat().st_size > size:
            dest.unlink()
        subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "10", "--retry-all-errors", "-o", str(dest), url])
        time.sleep(2)
    ok = dest.exists() and dest.stat().st_size == size and md5(dest) == checksum
    print(f"{'OK  ' if ok else 'FAIL'} {dest.name}", flush=True)
    return ok


def main():
    names = sys.argv[1:] or list(RECORDS)
    all_ok = True
    for name in names:
        rec, out = RECORDS[name]
        out.mkdir(parents=True, exist_ok=True)
        meta = None
        while meta is None:  # tolerate transient network / TLS interruptions
            r = subprocess.run(["curl", "-sSfL", "--retry", "10", "--retry-all-errors", f"https://zenodo.org/api/records/{rec}"],
                               capture_output=True, text=True)
            try:
                meta = json.loads(r.stdout)
            except json.JSONDecodeError:
                time.sleep(30)
        keep = FILE_FILTER.get(name, lambda k: True)
        files = [f for f in meta["files"] if not f["key"].endswith(".csv") and keep(f["key"])]
        with ThreadPoolExecutor(4) as pool:  # Zenodo throttles per connection
            oks = pool.map(lambda f: fetch(f["links"]["self"], out / f["key"], f["size"], f["checksum"].split(":")[1]), files)
            all_ok &= all(oks)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
