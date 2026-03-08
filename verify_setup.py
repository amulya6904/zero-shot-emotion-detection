"""
Complete environment verification script for the Zero-Shot Emotion Detection project.
"""

from __future__ import annotations

import importlib
import sys


def check_version(package_name: str, import_name: str | None = None) -> bool:
    """Check whether a package imports successfully and print its version."""
    module_name = import_name or package_name

    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError:
        print(f"[MISSING] {package_name}: not installed")
        return False


def print_section(title: str) -> None:
    divider = "=" * 60
    print(f"\n{divider}")
    print(title)
    print(divider)


def main() -> int:
    print("=" * 60)
    print("ZERO-SHOT EMOTION DETECTION - ENVIRONMENT VERIFICATION")
    print("=" * 60)

    print(f"\nPython Version: {sys.version}")
    if sys.version_info >= (3, 9):
        print("[OK] Python version is compatible (3.9+)")
    else:
        print("[FAIL] Python version must be 3.9 or higher")

    print_section("CHECKING CORE DEPENDENCIES")
    core_packages = [
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
    ]

    core_results = [check_version(display_name, import_name) for display_name, import_name in core_packages]

    print_section("CHECKING NLP DEPENDENCIES")
    nlp_packages = [
        ("Transformers", "transformers"),
        ("Datasets", "datasets"),
        ("NLTK", "nltk"),
    ]

    nlp_results = [check_version(display_name, import_name) for display_name, import_name in nlp_packages]

    print_section("CHECKING VISUALIZATION DEPENDENCIES")
    viz_packages = [
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("Plotly", "plotly"),
    ]

    viz_results = [check_version(display_name, import_name) for display_name, import_name in viz_packages]

    print_section("CHECKING DEVELOPMENT DEPENDENCIES")
    dev_packages = [
        ("Jupyter", "jupyter"),
        ("PyTest", "pytest"),
        ("Black", "black"),
    ]

    dev_results = [check_version(display_name, import_name) for display_name, import_name in dev_packages]

    print_section("ADVANCED CHECKS")

    try:
        import torch

        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Running on CPU. This is fine for setup and early experimentation.")
    except Exception as exc:
        print(f"[FAIL] PyTorch GPU check failed: {exc}")

    print("\nTesting Hugging Face tokenizer loading...")
    try:
        from transformers import AutoTokenizer

        print("Loading xlm-roberta-base tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        print("[OK] Successfully loaded XLM-RoBERTa tokenizer")
        print(f"Vocabulary size: {len(tokenizer)}")
    except Exception as exc:
        print(f"[WARN] Could not load tokenizer: {exc}")
        print("This usually means the model is not cached yet and internet access is unavailable.")

    all_results = core_results + nlp_results + viz_results + dev_results
    installed_count = sum(all_results)
    total_count = len(all_results)

    print_section("SUMMARY")
    print(f"Packages available: {installed_count}/{total_count}")
    if installed_count == total_count and sys.version_info >= (3, 9):
        print("[OK] Base environment checks passed.")
    else:
        print("[WARN] Some checks failed. Review the output above.")

    print("\nRun this script with:")
    print("python verify_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
