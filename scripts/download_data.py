"""
Download the raw MLIT data this project uses into data/raw/.

All files come from MLIT's National Land Numerical Information service
(国土数値情報, https://nlftp.mlit.go.jp/ksj/) and are published under the
Japan Public Data License 1.0, which permits reuse with attribution.

  L01  Chika-Kouji benchmark land prices (地価公示), Kumamoto, 2019-2026
  A29  Urban planning zones (用途地域), FY2019
  A31  Flood inundation risk zones (洪水浸水想定区域), FY2012
  A33  Landslide hazard warning zones (土砂災害警戒区域), FY2018
  N03  Administrative boundaries (行政区域), 2023-01-01

Each archive is unpacked flat into its target folder, keeping only the files
the pipeline reads plus the KS-META metadata (which documents the column
codes). Folders that already contain files are skipped, so the script is
safe to re-run.

Usage:
  python scripts/download_data.py
"""

import io
import os
import re
import sys
import zipfile

import requests

BASE_URL = "https://nlftp.mlit.go.jp/ksj/gml/data"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

_GEOJSON = r"\.geojson$|^KS-META-.*\.xml$"
_SHAPEFILE = r"\.(shp|shx|dbf|prj)$|^KS-META-.*\.xml$"

# (url path under BASE_URL, target folder under data/raw, basename regex to keep)
DATASETS = [
    *(
        (f"L01/L01-{yy}/L01-{yy}_43_GML.zip", f"chika_kouji/L01-{yy}_43_GML", _GEOJSON)
        for yy in range(19, 27)
    ),
    ("A29/A29-19/A29-19_43_GML.zip", "hazards/A29_43_GML", _GEOJSON),
    ("A31/A31-12/A31-12_43_GML.zip", "hazards/A31_43_GML", _SHAPEFILE),
    ("A33/A33-18/A33-18_43_GML.zip", "hazards/A33_43_GML", _SHAPEFILE),
    ("N03/N03-2023/N03-20230101_43_GML.zip", "hazards/N03_43_GML", _GEOJSON),
]


def _fetch(url_path: str, target: str, keep: str) -> int:
    resp = requests.get(f"{BASE_URL}/{url_path}", timeout=300)
    resp.raise_for_status()
    os.makedirs(target, exist_ok=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for member in zf.infolist():
            name = os.path.basename(member.filename)
            if member.is_dir() or not re.search(keep, name):
                continue
            with zf.open(member) as src, open(os.path.join(target, name), "wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def main() -> int:
    print(f"Source: {BASE_URL} (MLIT, Japan Public Data License 1.0)")
    for url_path, folder, keep in DATASETS:
        target = os.path.join(RAW_DIR, folder)
        if os.path.isdir(target) and os.listdir(target):
            print(f"  skip  {folder} (already present)")
            continue
        n = _fetch(url_path, target, keep)
        print(f"  saved {folder} ({n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
