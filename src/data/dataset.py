"""
PyTorch Dataset and DataLoader utilities for emotion detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from src.data.preprocessing import EmotionDataset


class EmotionDetectionDataset(Dataset):
    """PyTorch Dataset for multi-label emotion detection."""

    def __init__(
        self,
        texts: list[str],
        emotion_ids: list[list[int]],
        tokenizer,
        max_length: int = 512,
        num_emotions: int = 7,
    ) -> None:
        self.texts = texts
        self.emotion_ids = emotion_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_emotions = num_emotions

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Tokenize one example and return multi-label targets."""
        text = self.texts[idx]
        emotions = self.emotion_ids[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        emotion_targets = torch.zeros(self.num_emotions, dtype=torch.float32)
        for emotion_id in emotions:
            if 0 <= emotion_id < self.num_emotions:
                emotion_targets[emotion_id] = 1.0

        token_type_ids = encoding.get(
            "token_type_ids",
            torch.zeros((1, self.max_length), dtype=torch.long),
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "labels": emotion_targets,
            "text": text,
        }


def create_data_loaders(
    train_data: "EmotionDataset" | None,
    val_data: "EmotionDataset" | None = None,
    test_data: "EmotionDataset" | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    model_name: str = "xlm-roberta-base",
    max_length: int = 512,
) -> dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test splits."""
    print(f"\n{'=' * 60}")
    print("CREATING DATA LOADERS")
    print(f"{'=' * 60}\n")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Tokenizer loaded (vocab size: {len(tokenizer)})")

    loaders: dict[str, DataLoader] = {}

    if train_data is not None:
        print("\nCreating training DataLoader:")
        print(f"  Examples: {len(train_data.data)}")
        print(f"  Batch size: {batch_size}")

        train_dataset = EmotionDetectionDataset(
            texts=train_data.data["text_processed"].tolist(),
            emotion_ids=train_data.data["emotion_ids"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            num_emotions=len(train_data.EMOTION_LABELS),
        )

        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  Training loader created ({len(loaders['train'])} batches)")

    if val_data is not None:
        print("\nCreating validation DataLoader:")
        print(f"  Examples: {len(val_data.data)}")

        val_dataset = EmotionDetectionDataset(
            texts=val_data.data["text_processed"].tolist(),
            emotion_ids=val_data.data["emotion_ids"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            num_emotions=len(val_data.EMOTION_LABELS),
        )

        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  Validation loader created ({len(loaders['val'])} batches)")

    if test_data is not None:
        print("\nCreating test DataLoader:")
        print(f"  Examples: {len(test_data.data)}")

        test_dataset = EmotionDetectionDataset(
            texts=test_data.data["text_processed"].tolist(),
            emotion_ids=test_data.data["emotion_ids"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
            num_emotions=len(test_data.EMOTION_LABELS),
        )

        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  Test loader created ({len(loaders['test'])} batches)")

    print("\nAll DataLoaders created successfully.")
    return loaders


if __name__ == "__main__":
    print("This module provides PyTorch Dataset and DataLoader utilities.")
    print("Import and use it in training scripts.")
