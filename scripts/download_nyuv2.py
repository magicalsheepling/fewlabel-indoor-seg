"""Download the labelled NYUv2 file into data/ (skips if present)."""

from pathlib import Path
from urllib.request import urlretrieve

URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
DEST = Path("data/nyu_depth_v2_labeled.mat")

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"{DEST} already there ({DEST.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"downloading {URL} (~2.8 GB) ...")
        urlretrieve(URL, DEST)
        print("done")
