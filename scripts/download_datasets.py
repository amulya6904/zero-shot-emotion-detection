"""
Script to download the GoEmotions dataset from Hugging Face and save splits to
CSV files under ``data/english/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset


EMOTION_MAP = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "neutral",
    21: "optimism",
    22: "pride",
    23: "realization",
    24: "relief",
    25: "remorse",
    26: "sadness",
    27: "surprise",
}

OUTPUT_DIR = Path("data") / "english"


def download_goemotions():
    """Download the GoEmotions dataset and return train, validation, and test splits."""
    print("Downloading GoEmotions dataset...")
    dataset = load_dataset("go_emotions")
    available_splits = list(dataset.keys())

    if {"train", "validation", "test"}.issubset(available_splits):
        train_data = dataset["train"]
        validation_data = dataset["validation"]
        test_data = dataset["test"]
    elif "train" in dataset:
        # Some cached or older dataset variants only expose a train split.
        # Create deterministic validation and test splits so the script still
        # produces the expected CSV outputs.
        print(
            "Validation/test splits not provided by this dataset version. "
            "Creating deterministic splits from train."
        )
        split_data = dataset["train"].train_test_split(test_size=0.2, seed=42)
        val_test = split_data["test"].train_test_split(test_size=0.5, seed=42)
        train_data = split_data["train"]
        validation_data = val_test["train"]
        test_data = val_test["test"]
    else:
        raise ValueError(f"Unexpected dataset splits: {available_splits}")

    print("Downloaded GoEmotions")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation: {len(validation_data)} examples")
    print(f"  Test: {len(test_data)} examples")

    return train_data, validation_data, test_data


def preprocess_goemotions(dataset_split, split_name: str) -> pd.DataFrame:
    """Convert a split into a normalized CSV-friendly DataFrame."""
    data_list: list[dict[str, object]] = []

    for example in dataset_split:
        text = example["text"]
        labels = example["labels"]
        emotion_names = [EMOTION_MAP[label_idx] for label_idx in labels]

        data_list.append(
            {
                "text": text,
                "emotions": ",".join(emotion_names),
                "num_emotions": len(emotion_names),
            }
        )

    df = pd.DataFrame(data_list)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{split_name}.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {split_name}: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOADING GOEMOTIONS DATASET")
    print("=" * 60)

    train, val, test = download_goemotions()

    print("\n" + "=" * 60)
    print("PREPROCESSING DATASETS")
    print("=" * 60 + "\n")

    train_df = preprocess_goemotions(train, "train")
    val_df = preprocess_goemotions(val, "validation")
    test_df = preprocess_goemotions(test, "test")

    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    print("\nSample from training data:")
    print(train_df.head(3).to_string())

    print("\nGoEmotions dataset ready in data/english/")
