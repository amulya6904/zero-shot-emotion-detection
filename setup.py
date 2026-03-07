#!/usr/bin/env python3
"""Packaging configuration for the zero-shot emotion detection project.

This setup script is designed to be robust enough for local installs and PyPI
submission. It reads the project README for the long description, parses
runtime dependencies from ``requirements.txt``, and maps the current ``src/``
source tree into the requested ``zero_shot_emotion`` package namespace.
"""

from __future__ import annotations

from pathlib import Path
import sys

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"
SOURCE_DIR_NAME = "src"
SOURCE_DIR = BASE_DIR / SOURCE_DIR_NAME


def read_text_file(path: Path, default: str = "") -> str:
    """Read a text file safely and return a fallback value on failure."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Warning: {path.name} was not found. Using fallback content.", file=sys.stderr)
    except OSError as exc:
        print(f"Warning: Could not read {path.name}: {exc}", file=sys.stderr)
    return default


def parse_requirements(path: Path) -> list[str]:
    """Parse install requirements while skipping comments and blank lines."""

    requirements: list[str] = []
    raw_text = read_text_file(path)

    if not raw_text:
        print("Warning: requirements.txt is empty or unavailable.", file=sys.stderr)
        return requirements

    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue

        # Remove inline comments while preserving the requirement pin.
        requirement = cleaned.split("#", maxsplit=1)[0].strip()
        if requirement:
            requirements.append(requirement)

    return requirements


def discover_package_mapping() -> tuple[list[str], dict[str, str]]:
    """Map the existing ``src/`` tree into the ``zero_shot_emotion`` namespace.

    The repository currently stores package contents directly under ``src/``
    rather than under ``src/zero_shot_emotion/``. To keep packaging aligned
    with the requested module name without forcing a filesystem restructure,
    this function:

    1. Discovers subpackages beneath ``src/`` using ``find_packages()``
    2. Prefixes them with ``zero_shot_emotion``
    3. Builds a matching ``package_dir`` mapping for setuptools
    """

    package_dir: dict[str, str] = {"zero_shot_emotion": SOURCE_DIR_NAME}
    discovered = find_packages(where=SOURCE_DIR_NAME)
    packages = ["zero_shot_emotion"]

    for package in discovered:
        logical_name = f"zero_shot_emotion.{package}"
        filesystem_path = Path(SOURCE_DIR_NAME) / Path(package.replace(".", "/"))
        package_dir[logical_name] = str(filesystem_path)
        packages.append(logical_name)

    return packages, package_dir


INSTALL_REQUIRES = parse_requirements(REQUIREMENTS_PATH)
PACKAGES, PACKAGE_DIR = discover_package_mapping()


setup(
    name="zero-shot-emotion-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Zero-Shot Cross-Lingual Emotion Detection for Indian Dialects",
    long_description=read_text_file(
        README_PATH,
        default="Zero-Shot Cross-Lingual Emotion Detection for Indian Dialects",
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/zero-shot-emotion-detection",
    license="MIT",
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "black==23.12.0",
            "flake8==6.1.0",
            "mypy==1.7.1",
        ],
        # Note: setuptools extras cannot express the correct PyTorch CUDA wheel
        # index URL. Users requiring GPU builds should install the appropriate
        # torch wheel from https://pytorch.org/get-started/locally/.
        "gpu": [
            "torch==2.1.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "emotion-detection",
        "cross-lingual-nlp",
        "zero-shot-learning",
        "indian-dialects",
        "transformers",
        "pytorch",
    ],
    project_urls={
        "Source": "https://github.com/YOUR_USERNAME/zero-shot-emotion-detection",
        "Issues": "https://github.com/YOUR_USERNAME/zero-shot-emotion-detection/issues",
    },
    # Enable console scripts once a stable CLI entrypoint exists, for example:
    # entry_points={
    #     "console_scripts": [
    #         "zero-shot-emotion=zero_shot_emotion.inference.predictor:main",
    #     ],
    # },
    zip_safe=False,
)
