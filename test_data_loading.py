"""
Test that the data loading pipeline works correctly.
"""

from __future__ import annotations

import sys

sys.path.append(".")

from src.data.dataset import create_data_loaders
from src.data.preprocessing import prepare_data


print("=" * 60)
print("TESTING DATA LOADING PIPELINE")
print("=" * 60)

print("\nPreparing English data...")
english_datasets = prepare_data(
    language="english",
    train_path="data/english/train_simplified.csv",
    val_path="data/english/validation_simplified.csv",
    test_path="data/english/test_simplified.csv",
)

print("\nPreparing Hindi data...")
hindi_datasets = prepare_data(
    language="hindi",
    train_path="data/hindi/train.csv",
    val_path="data/hindi/validation.csv",
)

print("\nPreparing Bhojpuri data...")
bhojpuri_datasets = prepare_data(
    language="bhojpuri",
    train_path="",
    test_path="data/bhojpuri/test.csv",
)

print("\n\nCreating data loaders for English...")
english_loaders = create_data_loaders(
    train_data=english_datasets["train"],
    val_data=english_datasets.get("val"),
    test_data=english_datasets.get("test"),
    batch_size=32,
    model_name="xlm-roberta-base",
)

print("\nTesting a batch from English training data...")
train_loader = english_loaders["train"]
batch = next(iter(train_loader))

print(f"Batch keys: {batch.keys()}")
print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Attention mask shape: {batch['attention_mask'].shape}")
print(f"Labels shape: {batch['labels'].shape}")
print(f"Sample text: {batch['text'][0]}")

print("\nData loading pipeline test passed!")
