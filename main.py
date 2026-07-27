"""
JASM Spatial ML — Production Master Orchestrator

Usage:
  python main.py --all          # Execute entire pipeline end-to-end
  python main.py --stage 1      # Execute a single targeted stage (1-4)
"""

import argparse
import sys
import time

from src import evaluate, features, load_data, train


def run_pipeline() -> None:
    parser = argparse.ArgumentParser(description="JASM Spatial ML Production Pipeline")
    parser.add_argument(
        "--all", action="store_true", help="Execute entire pipeline end-to-end"
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2, 3, 4], help="Execute a single targeted stage"
    )
    args = parser.parse_args()

    if not (args.all or args.stage):
        parser.print_help()
        sys.exit(1)

    start_time = time.time()
    print("=========================================================")
    print("STARTING JASM SPATIAL MACHINE LEARNING PRODUCTION PIPELINE")
    print("=========================================================\n")

    try:
        if args.all or args.stage == 1:
            print("[PIPELINE EVENT] Launching Stage 1: Data Ingestion...")
            load_data.run()

        if args.all or args.stage == 2:
            print("\n[PIPELINE EVENT] Launching Stage 2: Feature Engineering...")
            features.run()

        if args.all or args.stage == 3:
            print("\n[PIPELINE EVENT] Launching Stage 3: Architecture Training...")
            train.run()

        if args.all or args.stage == 4:
            print("\n[PIPELINE EVENT] Launching Stage 4: Validation Diagnostics...")
            evaluate.run()

        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Pipeline completed successfully in {elapsed:.2f} seconds.")

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] Pipeline execution halted due to error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
