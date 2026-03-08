"""
Simplify GoEmotions labels into 7 core emotions for easier modeling.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EMOTION_SIMPLIFICATION = {
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "relief": "joy",
    "gratitude": "joy",
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "embarrassment": "fear",
    "surprise": "surprise",
    "curiosity": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
    "confusion": "neutral",
    "desire": "neutral",
    "realization": "neutral",
    "optimism": "neutral",
    "approval": "neutral",
    "caring": "neutral",
    "love": "neutral",
    "pride": "neutral",
    "remorse": "neutral",
}

DATA_DIR = Path("data") / "english"


def simplify_emotions(emotions_str: str) -> str:
    """Convert comma-separated emotions into a simplified label set."""
    emotions = [emotion.strip() for emotion in str(emotions_str).split(",") if emotion.strip()]
    simplified = {
        EMOTION_SIMPLIFICATION[emotion]
        for emotion in emotions
        if emotion in EMOTION_SIMPLIFICATION
    }
    return ",".join(sorted(simplified))


def process_file(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read a CSV, simplify its labels, and write the output CSV."""
    print(f"Processing: {input_path}")

    df = pd.read_csv(input_path)
    df["emotions_simplified"] = df["emotions"].apply(simplify_emotions)
    df = df[["text", "emotions_simplified"]].copy()
    df.columns = ["text", "emotions"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"  Unique label combinations: {df['emotions'].nunique()}")
    print(f"  Top combinations:\n{df['emotions'].value_counts().head(10)}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLIFYING EMOTIONS TO 7 CORE")
    print("=" * 60 + "\n")

    for split in ["train", "validation", "test"]:
        input_path = DATA_DIR / f"{split}.csv"
        output_path = DATA_DIR / f"{split}_simplified.csv"
        process_file(input_path, output_path)
        print()

    print("All files simplified and saved.")
