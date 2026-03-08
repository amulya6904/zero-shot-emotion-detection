"""
Data preprocessing pipeline for emotion detection.

Handles text normalization, dataset loading, label conversion, and basic
dataset statistics for English, Hindi, and Bhojpuri data files.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pandas as pd


class EmotionTextPreprocessor:
    """Preprocess text for emotion detection."""

    def __init__(
        self,
        language: str = "english",
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
    ) -> None:
        self.language = language
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        normalized = str(text)

        if self.remove_urls:
            normalized = re.sub(r"http\S+|www\S+|https\S+", "", normalized, flags=re.MULTILINE)

        if self.remove_mentions:
            normalized = re.sub(r"@\w+|#\w+", "", normalized)

        normalized = re.sub(r"\s+", " ", normalized).strip()

        if self.lowercase and self.language == "english":
            normalized = normalized.lower()

        return normalized

    def preprocess(self, text: str) -> str:
        """Run the full preprocessing pipeline."""
        return self.clean_text(text)


class EmotionDataset:
    """Load and manage emotion datasets."""

    EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
    EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
    ID_TO_EMOTION = {idx: emotion for emotion, idx in EMOTION_TO_ID.items()}

    def __init__(self, language: str = "english") -> None:
        self.language = language
        self.data: pd.DataFrame | None = None
        self.preprocessor = EmotionTextPreprocessor(language=language)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        print(f"Loading data from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        print(f"  Loaded {len(df)} examples")

        if "text" not in df.columns or "emotions" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'emotions' columns")

        self.data = df
        return df

    def preprocess_texts(self) -> pd.DataFrame:
        """Preprocess all texts in the loaded dataset."""
        if self.data is None:
            raise ValueError("Load data first with load_csv()")

        print(f"Preprocessing {len(self.data)} texts...")
        self.data["text_processed"] = self.data["text"].apply(self.preprocessor.preprocess)
        print("  Preprocessing complete")
        return self.data

    def convert_emotions_to_ids(self) -> pd.DataFrame:
        """Convert comma-separated emotion labels into integer ID lists."""
        if self.data is None:
            raise ValueError("Load data first")

        def emotions_to_ids(emotion_str: str) -> list[int]:
            emotions = [emotion.strip() for emotion in str(emotion_str).split(",") if emotion.strip()]
            return [self.EMOTION_TO_ID[emotion] for emotion in emotions if emotion in self.EMOTION_TO_ID]

        self.data["emotion_ids"] = self.data["emotions"].apply(emotions_to_ids)
        return self.data

    def get_statistics(self) -> None:
        """Print dataset statistics."""
        if self.data is None:
            print("No data loaded")
            return

        print(f"\n{'=' * 60}")
        print(f"DATASET STATISTICS: {self.language.upper()}")
        print(f"{'=' * 60}")
        print("\nOverall:")
        print(f"  Total examples: {len(self.data)}")
        print(f"  Avg text length: {self.data['text'].str.len().mean():.1f} chars")
        print(f"  Min text length: {self.data['text'].str.len().min()} chars")
        print(f"  Max text length: {self.data['text'].str.len().max()} chars")

        print("\nEmotion distribution:")
        emotion_counts: dict[str, int] = {}
        for emotions_str in self.data["emotions"]:
            for emotion in str(emotions_str).split(","):
                label = emotion.strip()
                if not label:
                    continue
                emotion_counts[label] = emotion_counts.get(label, 0) + 1

        for emotion in self.EMOTION_LABELS:
            count = emotion_counts.get(emotion, 0)
            pct = 100 * count / len(self.data) if len(self.data) else 0
            print(f"  {emotion:12s}: {count:5d} ({pct:5.1f}%)")

        print()


def prepare_data(
    language: str,
    train_path: str,
    val_path: str | None = None,
    test_path: str | None = None,
) -> dict[str, EmotionDataset]:
    """Prepare train, validation, and test datasets for a given language."""
    print(f"\n{'=' * 60}")
    print(f"PREPARING DATA: {language.upper()}")
    print(f"{'=' * 60}\n")

    datasets: dict[str, EmotionDataset] = {}

    if train_path:
        print("TRAINING DATA:")
        train_dataset = EmotionDataset(language=language)
        train_dataset.load_csv(train_path)
        train_dataset.preprocess_texts()
        train_dataset.convert_emotions_to_ids()
        train_dataset.get_statistics()
        datasets["train"] = train_dataset

    if val_path:
        print("\nVALIDATION DATA:")
        val_dataset = EmotionDataset(language=language)
        val_dataset.load_csv(val_path)
        val_dataset.preprocess_texts()
        val_dataset.convert_emotions_to_ids()
        val_dataset.get_statistics()
        datasets["val"] = val_dataset

    if test_path:
        print("\nTEST DATA:")
        test_dataset = EmotionDataset(language=language)
        test_dataset.load_csv(test_path)
        test_dataset.preprocess_texts()
        test_dataset.convert_emotions_to_ids()
        test_dataset.get_statistics()
        datasets["test"] = test_dataset

    return datasets


if __name__ == "__main__":
    print("This module provides data preprocessing utilities.")
    print("Import and use it from training or data preparation scripts.")
