"""
Master script to prepare all data for training.

Runs all data preparation steps in sequence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def run_script(script_name: str) -> None:
    """Run a Python script with the current interpreter."""
    script_path = ROOT_DIR / script_name

    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print(f"{'=' * 60}\n")

    subprocess.run([sys.executable, str(script_path)], check=True, cwd=ROOT_DIR)


def main() -> None:
    print("=" * 60)
    print("ZERO-SHOT EMOTION DETECTION: DATA PREPARATION")
    print("=" * 60)

    print("\n[1/4] Downloading GoEmotions Dataset...")
    run_script("scripts/download_datasets.py")

    print("\n[2/4] Simplifying to 7 core emotions...")
    run_script("scripts/simplify_emotions.py")

    print("\n[3/4] Creating Hindi emotion dataset...")
    run_script("scripts/create_hindi_emotions.py")

    print("\n[4/4] Creating Bhojpuri emotion test set...")
    run_script("scripts/create_bhojpuri_emotions.py")

    print("\n" + "=" * 60)
    print("ALL DATA PREPARATION COMPLETE!")
    print("=" * 60)

    print("\nData Summary:")
    print("  English: data/english/ (train, validation, test)")
    print("  Hindi: data/hindi/ (train, validation)")
    print("  Bhojpuri: data/bhojpuri/ (test)")

    print("\nNext steps:")
    print("  1. Run notebooks/01_eda.ipynb for data exploration")
    print("  2. Begin Phase 3: Model Development")


if __name__ == "__main__":
    main()
