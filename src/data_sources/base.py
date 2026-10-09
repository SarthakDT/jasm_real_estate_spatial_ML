"""Shared helpers for data_sources modules: bbox validation and cached HTTP fetches."""

import json
import os
import time

import pandas as pd
import requests

from src import config


def validate_kumamoto_bbox(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.Series:
    """Return a boolean mask of rows falling outside Kumamoto Prefecture's bounding box.

    A coarse sanity check for swapped or garbled coordinates. It is far too
    loose to detect a datum shift.
    """
    return (
        ~df[lat_col].between(config.LAT_MIN, config.LAT_MAX, inclusive="neither")
        | ~df[lon_col].between(config.LON_MIN, config.LON_MAX, inclusive="neither")
    )


def retry_request(fn, max_attempts: int = 3, backoff: float = 2.0):
    """Call fn() with up to max_attempts tries, sleeping `backoff * attempt` seconds between failures."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(backoff * attempt)
    raise last_exc


def cached_get(url: str, cache_path: str, params: dict | None = None, headers: dict | None = None, force_refresh: bool = False) -> dict:
    """GET a JSON endpoint, caching the raw response at cache_path.

    If cache_path already exists and force_refresh is False, the network call
    is skipped entirely and the cached JSON is returned — the same
    idempotent, re-runnable pattern the manually-downloaded MLIT GML folders
    already follow.
    """
    if os.path.isfile(cache_path) and not force_refresh:
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    def _do_request():
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    data = retry_request(_do_request)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)

    return data
