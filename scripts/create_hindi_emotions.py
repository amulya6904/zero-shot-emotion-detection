"""
Create a small curated Hindi emotion dataset for early experimentation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


HINDI_EMOTIONS = [
    {"text": "मुझे बहुत खुशी है कि आप यहाँ हैं!", "emotions": "joy"},
    {"text": "यह बिल्कुल शानदार है!", "emotions": "joy"},
    {"text": "मैं इतना खुश हूँ कि मेरे सपने सच हुए!", "emotions": "joy"},
    {"text": "वाह! यह तो अद्भुत है!", "emotions": "joy"},
    {"text": "मेरी टीम ने जीत लिया! मैं बहुत खुश हूँ!", "emotions": "joy"},
    {"text": "मुझे बहुत दुख हुआ यह सुनकर।", "emotions": "sadness"},
    {"text": "मैं बहुत उदास हूँ।", "emotions": "sadness"},
    {"text": "इस खबर ने मेरा दिल तोड़ दिया।", "emotions": "sadness"},
    {"text": "मुझे बहुत निराशा हुई।", "emotions": "sadness"},
    {"text": "मेरा प्रिय मित्र चला गया। मेरा दिल टूट गया।", "emotions": "sadness"},
    {"text": "यह बिल्कुल गलत है! मैं बहुत गुस्से में हूँ!", "emotions": "anger"},
    {"text": "मैं इस अन्याय से नाराज हूँ!", "emotions": "anger"},
    {"text": "यह अस्वीकार्य है!", "emotions": "anger"},
    {"text": "मुझे यह व्यवहार पसंद नहीं है!", "emotions": "anger"},
    {"text": "यह भयानक है! मेरा क्रोध सीमा से बाहर है!", "emotions": "anger"},
    {"text": "मुझे बहुत डर है।", "emotions": "fear"},
    {"text": "मैं घबराया हुआ हूँ।", "emotions": "fear"},
    {"text": "यह बहुत भयानक है।", "emotions": "fear"},
    {"text": "मुझे चिंता है कि क्या होगा।", "emotions": "fear"},
    {"text": "मेरा दिल तेज़ी से धड़क रहा है। मैं परेशान हूँ।", "emotions": "fear"},
    {"text": "वाह! मुझे यह उम्मीद नहीं था!", "emotions": "surprise"},
    {"text": "यह तो अचानक है!", "emotions": "surprise"},
    {"text": "मैं चकित हूँ!", "emotions": "surprise"},
    {"text": "यह घटना मेरे लिए बिल्कुल अप्रत्याशित था!", "emotions": "surprise"},
    {"text": "मेरा तो मुँह खुल गया! यह कैसे संभव है?", "emotions": "surprise"},
    {"text": "यह बहुत घिनौना है।", "emotions": "disgust"},
    {"text": "मुझे इससे घृणा है।", "emotions": "disgust"},
    {"text": "यह असहनीय है।", "emotions": "disgust"},
    {"text": "मुझे इस तरह की बातें पसंद नहीं।", "emotions": "disgust"},
    {"text": "यह बिल्कुल अरुचिकर है।", "emotions": "disgust"},
    {"text": "आज का मौसम ठीक है।", "emotions": "neutral"},
    {"text": "मैं कल बाजार जाऊँगा।", "emotions": "neutral"},
    {"text": "यह एक सामान्य दिन है।", "emotions": "neutral"},
    {"text": "उन्होंने कहा कि वह 5 बजे आएँगे।", "emotions": "neutral"},
    {"text": "मैं सुबह 8 बजे काम के लिए निकलता हूँ।", "emotions": "neutral"},
]

OUTPUT_DIR = Path("data") / "hindi"


def create_hindi_dataset() -> None:
    """Create and save train/validation splits for Hindi emotion examples."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("CREATING HINDI EMOTION DATASET")
    print("=" * 60 + "\n")

    df = pd.DataFrame(HINDI_EMOTIONS)

    print(f"Total examples: {len(df)}")
    print("\nEmotion distribution:")
    print(df["emotions"].value_counts())

    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)

    print(f"\nSaved train: {OUTPUT_DIR / 'train.csv'} ({len(train_df)} examples)")
    print(f"Saved val: {OUTPUT_DIR / 'validation.csv'} ({len(val_df)} examples)")
    print("\nSample Hindi emotions:")
    print(train_df.head(5).to_string(index=False))


if __name__ == "__main__":
    create_hindi_dataset()
