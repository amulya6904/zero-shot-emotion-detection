"""
Create a curated Bhojpuri emotion test set for zero-shot evaluation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


BHOJPURI_EMOTIONS = [
    {"text": "अरे, ई तो बहुत शानदार बा!", "emotions": "joy"},
    {"text": "मैं बहुत खुस हूँ!", "emotions": "joy"},
    {"text": "वाह! ई तो अच्छा हु!", "emotions": "joy"},
    {"text": "मोर प्रिय गावँ में आके खुसी हु!", "emotions": "joy"},
    {"text": "ऊ खेल में जीत गइल तो मैं खुस हु!", "emotions": "joy"},
    {"text": "अरे बाबा, कवन खुसी की बा ई!", "emotions": "joy"},
    {"text": "अरे, ई खबर सुनके मन बहुत दुखत हे।", "emotions": "sadness"},
    {"text": "मैं बहुत उदास हूँ।", "emotions": "sadness"},
    {"text": "ई बात सुनके दिल टूट गइल।", "emotions": "sadness"},
    {"text": "मोर दोस्त चल गइल तो बहुत दुख हु।", "emotions": "sadness"},
    {"text": "ई सुनके रोना आ गइल।", "emotions": "sadness"},
    {"text": "भगवान, ई दुख काहे भेजलस!", "emotions": "sadness"},
    {"text": "ई तो बहुत गलत बा! मैं बहुत गुस्से में हूँ!", "emotions": "anger"},
    {"text": "अरे, ई का करलस! मैं गुस्सा हु!", "emotions": "anger"},
    {"text": "अइसन व्यवहार ठीक नइखे! बड़ा गुस्सा आत हे!", "emotions": "anger"},
    {"text": "मैं इस अन्याय से नाराज हूँ!", "emotions": "anger"},
    {"text": "ऊ का करलस! मोर क्रोध सीमा में नइखे!", "emotions": "anger"},
    {"text": "ई तो सहन होखे का नइखे!", "emotions": "anger"},
    {"text": "अरे, मैं बहुत डरत हूँ।", "emotions": "fear"},
    {"text": "ई बात सुनके भय लग गइल।", "emotions": "fear"},
    {"text": "मोर दिल जोर से धड़कत हे।", "emotions": "fear"},
    {"text": "का होखे वाला हे, मैं घबराए हूँ।", "emotions": "fear"},
    {"text": "ई रास्ता बहुत अँधियारा हे, मैं भयभीत हूँ।", "emotions": "fear"},
    {"text": "भगवान, इहाँ मैं बहुत डरत हूँ।", "emotions": "fear"},
    {"text": "वाह! मैं यह नइ सोचलेहूँ!", "emotions": "surprise"},
    {"text": "अरे, ई तो अचानक हु!", "emotions": "surprise"},
    {"text": "मैं चकित हूँ! यह कइसे होलस!", "emotions": "surprise"},
    {"text": "मोर मुँह खुल गइल! ई संभव हु का!", "emotions": "surprise"},
    {"text": "ई तो बहुत अप्रत्याशित हु!", "emotions": "surprise"},
    {"text": "का! ई सच हे! मैं तो विश्वास नइ करत हूँ!", "emotions": "surprise"},
    {"text": "अरे, ई तो बहुत घिनौना हु।", "emotions": "disgust"},
    {"text": "मैं इससे नफरत करत हूँ।", "emotions": "disgust"},
    {"text": "ई सहन होखे का नइखे।", "emotions": "disgust"},
    {"text": "अइसन व्यवहार भोत बुरा हे।", "emotions": "disgust"},
    {"text": "मोर तो उल्टी आ जात हे।", "emotions": "disgust"},
    {"text": "ई बहुत अरुचिकर हु।", "emotions": "disgust"},
    {"text": "आज का मौसम अच्छा हे।", "emotions": "neutral"},
    {"text": "मैं कल बाजार जाइहीं।", "emotions": "neutral"},
    {"text": "यह एक सामान्य दिन हे।", "emotions": "neutral"},
    {"text": "ऊ 5 बजे आइहै।", "emotions": "neutral"},
    {"text": "मैं सुबह 8 बजे काम पर निकलत हूँ।", "emotions": "neutral"},
    {"text": "अइसन कुछ खास नइखे।", "emotions": "neutral"},
]

OUTPUT_DIR = Path("data") / "bhojpuri"


def create_bhojpuri_dataset() -> None:
    """Create and save a curated Bhojpuri emotion test set."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("CREATING BHOJPURI EMOTION TEST SET")
    print("=" * 60 + "\n")

    df = pd.DataFrame(BHOJPURI_EMOTIONS)

    print(f"Total examples: {len(df)}")
    print("\nEmotion distribution:")
    print(df["emotions"].value_counts())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "test.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved test set: {output_path} ({len(df)} examples)")
    print("\nSample Bhojpuri emotions:")
    print(df.head(6).to_string(index=False))
    print("\nNOTE: This is a curated test set for evaluation.")
    print("Real-world Bhojpuri may vary in writing style.")
    print("This set demonstrates the language characteristics.")


if __name__ == "__main__":
    create_bhojpuri_dataset()
