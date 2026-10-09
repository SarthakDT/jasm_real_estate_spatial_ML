"""
Run the JASM land-price pipeline.

Usage:
  python main.py --all          # run every stage in order
  python main.py --stage 1      # run a single stage (1-4)
"""

import argparse
import sys
import time

from src import evaluate, features, load_data, train

STAGES = {
    1: ("Load MLIT land-price data", load_data.run),
    2: ("Build features", features.run),
    3: ("Train and compare models", train.run),
    4: ("Evaluate the selected model", evaluate.run),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="JASM land-price pipeline")
    parser.add_argument("--all", action="store_true", help="run every stage in order")
    parser.add_argument("--stage", type=int, choices=sorted(STAGES), help="run a single stage")
    args = parser.parse_args()

    if not (args.all or args.stage):
        parser.print_help()
        sys.exit(1)

    start = time.time()
    for number, (name, run) in STAGES.items():
        if args.all or args.stage == number:
            print(f"\n=== Stage {number}: {name} ===")
            run()
    print(f"\nDone in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
