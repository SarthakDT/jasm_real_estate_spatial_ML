"""
Stage 1 entrypoint — thin compatibility wrapper.

The actual ingestion logic lives in src/data_sources/chika_kouji.py (the
first of several per-source modules under src/data_sources/). This wrapper
exists so main.py's `from src import load_data; load_data.run()` keeps
working unchanged.
"""

from src.data_sources import chika_kouji


def run():
    return chika_kouji.run()


if __name__ == "__main__":
    run()
