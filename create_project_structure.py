#!/usr/bin/env python3
"""Create the scaffold for a zero-shot cross-lingual emotion detection project.

The script creates the requested directory hierarchy, adds ``__init__.py``
package markers, writes placeholder Python modules with meaningful docstrings,
places ``.gitkeep`` files in intentionally empty directories, and prints a
tree-style summary of the resulting structure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_NAME = "Zero-Shot Cross-Lingual Emotion Detection for Indian Dialects"


@dataclass(frozen=True)
class FileSpec:
    """Describe a file that should exist in the scaffold."""

    relative_path: Path
    content: str


DIRECTORIES = [
    Path("data"),
    Path("data/english"),
    Path("data/hindi"),
    Path("data/bhojpuri"),
    Path("data/processed"),
    Path("src"),
    Path("src/models"),
    Path("src/data"),
    Path("src/training"),
    Path("src/evaluation"),
    Path("src/inference"),
    Path("notebooks"),
    Path("results"),
    Path("results/models"),
    Path("results/predictions"),
    Path("results/visualizations"),
    Path("paper"),
    Path("docs"),
    Path("tests"),
]


FILES = [
    FileSpec(
        Path("src/__init__.py"),
        '"""Top-level package for the emotion detection project."""\n',
    ),
    FileSpec(
        Path("src/models/__init__.py"),
        '"""Model definitions for multilingual and zero-shot emotion detection."""\n',
    ),
    FileSpec(
        Path("src/models/multilingual_model.py"),
        '"""Multilingual model interfaces and baseline implementations for emotion detection."""\n',
    ),
    FileSpec(
        Path("src/models/zero_shot_classifier.py"),
        '"""Zero-shot classification components for cross-lingual emotion prediction."""\n',
    ),
    FileSpec(
        Path("src/models/ensemble_model.py"),
        '"""Ensemble strategies for combining multilingual emotion detection models."""\n',
    ),
    FileSpec(
        Path("src/data/__init__.py"),
        '"""Data loading and preprocessing utilities for multilingual corpora."""\n',
    ),
    FileSpec(
        Path("src/data/dataset.py"),
        '"""Dataset abstractions for English, Hindi, and Bhojpuri emotion data."""\n',
    ),
    FileSpec(
        Path("src/data/preprocessing.py"),
        '"""Text normalization and preprocessing routines for Indian dialect data."""\n',
    ),
    FileSpec(
        Path("src/data/utils.py"),
        '"""Utility helpers for data access, validation, and transformation."""\n',
    ),
    FileSpec(
        Path("src/training/__init__.py"),
        '"""Training workflows and experiment configuration helpers."""\n',
    ),
    FileSpec(
        Path("src/training/trainer.py"),
        '"""Training loop orchestration for zero-shot cross-lingual emotion models."""\n',
    ),
    FileSpec(
        Path("src/training/config.py"),
        '"""Configuration objects and constants for training experiments."""\n',
    ),
    FileSpec(
        Path("src/evaluation/__init__.py"),
        '"""Evaluation utilities for model performance assessment."""\n',
    ),
    FileSpec(
        Path("src/evaluation/metrics.py"),
        '"""Metric definitions for evaluating emotion detection quality."""\n',
    ),
    FileSpec(
        Path("src/evaluation/analysis.py"),
        '"""Analysis tools for inspecting cross-lingual emotion prediction results."""\n',
    ),
    FileSpec(
        Path("src/evaluation/visualization.py"),
        '"""Visualization helpers for emotion detection experiments and findings."""\n',
    ),
    FileSpec(
        Path("src/inference/__init__.py"),
        '"""Inference interfaces for serving and batch prediction workflows."""\n',
    ),
    FileSpec(
        Path("src/inference/predictor.py"),
        '"""Prediction entry points for zero-shot emotion inference across dialects."""\n',
    ),
    FileSpec(
        Path("tests/__init__.py"),
        '"""Test suite package for project modules and data pipelines."""\n',
    ),
    FileSpec(
        Path("tests/test_models.py"),
        '"""Placeholder tests for model components and inference behavior."""\n',
    ),
    FileSpec(
        Path("tests/test_data.py"),
        '"""Placeholder tests for dataset loading and preprocessing logic."""\n',
    ),
]


EMPTY_DIRECTORIES = [
    Path("data/english"),
    Path("data/hindi"),
    Path("data/bhojpuri"),
    Path("data/processed"),
    Path("notebooks"),
    Path("results/models"),
    Path("results/predictions"),
    Path("results/visualizations"),
    Path("paper"),
    Path("docs"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for scaffold creation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Target directory in which to create the project scaffold.",
    )
    return parser.parse_args()


def build_tree_lines(paths: Iterable[Path]) -> list[str]:
    """Return a tree-style representation for the supplied relative paths."""

    children: dict[str, set[str]] = {"": set()}

    for path in sorted(paths, key=lambda item: item.parts):
        parent = ""
        for part in path.parts:
            node = f"{parent}/{part}" if parent else part
            children.setdefault(parent, set()).add(part)
            children.setdefault(node, set())
            parent = node

    lines = ["."]

    def walk(parent: str, prefix: str) -> None:
        entries = sorted(children[parent])
        for index, name in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = "\\-- " if is_last else "+-- "
            lines.append(f"{prefix}{connector}{name}")
            child_key = f"{parent}/{name}" if parent else name
            extension = "    " if is_last else "|   "
            walk(child_key, prefix + extension)

    walk("", "")
    return lines


def create_directories(base_dir: Path, directories: Iterable[Path]) -> list[Path]:
    """Create directories in parent-first order and return newly created ones."""

    created: list[Path] = []
    for directory in sorted(directories, key=lambda path: (len(path.parts), path.parts)):
        target = base_dir / directory
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created


def create_files(base_dir: Path, files: Iterable[FileSpec]) -> list[Path]:
    """Create scaffold files without overwriting existing content."""

    created: list[Path] = []
    for file_spec in files:
        target = base_dir / file_spec.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(file_spec.content, encoding="utf-8")
            created.append(file_spec.relative_path)
    return created


def create_gitkeep_files(base_dir: Path, directories: Iterable[Path]) -> list[Path]:
    """Create ``.gitkeep`` markers for directories intended to remain empty."""

    created: list[Path] = []
    for directory in directories:
        marker = base_dir / directory / ".gitkeep"
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text("", encoding="utf-8")
            created.append(marker.relative_to(base_dir))
    return created


def print_summary(base_dir: Path, created_dirs: list[Path], created_files: list[Path]) -> None:
    """Print a concise summary of the scaffold creation result."""

    all_paths = sorted(set(DIRECTORIES) | set(created_files), key=lambda path: path.parts)

    print(f"Project scaffold ready: {PROJECT_NAME}")
    print(f"Base directory: {base_dir.resolve()}")
    print(f"Created {len(created_dirs)} directories and {len(created_files)} files.")
    print()
    for line in build_tree_lines(all_paths):
        print(line)


def main() -> None:
    """Create the requested project structure relative to the chosen target."""

    args = parse_args()
    base_dir = Path(args.target).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    created_dirs = create_directories(base_dir, DIRECTORIES)
    created_files = create_files(base_dir, FILES)
    created_gitkeeps = create_gitkeep_files(base_dir, EMPTY_DIRECTORIES)

    print_summary(base_dir, created_dirs, created_files + created_gitkeeps)


if __name__ == "__main__":
    main()
