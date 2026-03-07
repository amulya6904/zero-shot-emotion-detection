[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)](#project-status)

# 🌏 Zero-Shot Cross-Lingual Emotion Detection for Indian Dialects

## Abstract
Emotion detection systems remain heavily concentrated around high-resource languages, while Indian dialects such as Bhojpuri continue to suffer from severe annotation scarcity, domain shift, and code-mixed linguistic variation. This project investigates a zero-shot and few-shot cross-lingual transfer pipeline that leverages multilingual transformer encoders, lightweight fine-tuning, and dialect-aware preprocessing to transfer emotion recognition capability from English and Hindi into low-resource settings. Its main contribution is a reproducible research framework for benchmarking multilingual transfer, evaluating robustness across dialects, and supporting future dataset expansion for underrepresented Indian languages.

## Key Features
- Zero-shot cross-lingual emotion classification for low-resource Indian dialects
- Support for English, Hindi, and Bhojpuri data organization and processing
- Benchmarking of multilingual transformer backbones including mBERT and XLM-RoBERTa
- Modular PyTorch and Hugging Face pipeline for training, evaluation, and inference
- Planned support for dialect normalization, script variation handling, and code-mixed text
- Reproducible experiment structure with configuration-driven workflows
- Evaluation utilities for macro F1, per-emotion analysis, and transfer-gap reporting
- Research-oriented project layout for experiments, visualizations, and paper artifacts

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Training and Fine-Tuning](#training-and-fine-tuning)
- [Inference](#inference)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact and Author](#contact-and-author)

## Installation

### System Requirements
- Operating system: Windows, Linux, or macOS
- Python: 3.9 or newer
- Memory: 8 GB RAM minimum, 16 GB recommended
- Storage: 5 GB free disk space for code, caches, checkpoints, and processed data
- GPU: Optional but recommended for training and large-batch inference

### Python Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Detailed Installation Steps
```powershell
git clone https://github.com/amulya6904/zero-shot-emotion-detection.git
cd zero-shot-emotion-detection
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU vs CPU Note
- CPU execution is sufficient for repository setup, data preprocessing, and small-scale experiments.
- GPU execution is strongly recommended for transformer fine-tuning and evaluation across large datasets.
- For NVIDIA GPUs, install the `torch`, `torchvision`, and `torchaudio` wheels that match your CUDA runtime before installing the rest of the dependencies.
- With PyTorch `2.1.0`, CUDA `11.8` and CUDA `12.1` wheels are common deployment choices; make sure your local NVIDIA driver supports the selected build.

## Quick Start
The code snippets below describe the intended public workflow for this repository. Some interfaces are placeholders in the current development version and will be implemented incrementally.

### 1. Basic Inference
```python
from src.inference.predictor import EmotionPredictor

predictor = EmotionPredictor(model_name="xlm-roberta-base")
result = predictor.predict("हम बहुत खुश बानी आज।")
print(result)
```

### 2. Batch Processing
```python
from src.inference.predictor import EmotionPredictor

texts = [
    "I am excited about the new opportunity.",
    "आज मन थोड़ा उदास है।",
    "ई खबर सुन के गुस्सा आ गइल।",
]

predictor = EmotionPredictor(model_name="bert-base-multilingual-cased")
predictions = predictor.predict_batch(texts, batch_size=8)
for item in predictions:
    print(item)
```

### 3. Custom Text with Scores
```python
from src.inference.predictor import EmotionPredictor

predictor = EmotionPredictor(
    model_name="xlm-roberta-base",
    return_probabilities=True,
)

output = predictor.predict("This feels uncertain, but I am still hopeful.")
print(output["label"], output["scores"])
```

### 4. Command-Line Style Workflow
```powershell
python -m src.training.trainer --config config.yaml
python -m src.inference.predictor --text "आज बहुत अच्छा लग रहा है"
```

## Project Structure
```text
zero-shot-emotion-detection/
├── create_project_structure.py
├── requirements.txt
├── README.md
├── data/
│   ├── english/
│   ├── hindi/
│   ├── bhojpuri/
│   └── processed/
├── docs/
├── notebooks/
├── paper/
├── results/
│   ├── models/
│   ├── predictions/
│   └── visualizations/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocessing.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── analysis.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── inference/
│   │   └── predictor.py
│   ├── models/
│   │   ├── ensemble_model.py
│   │   ├── multilingual_model.py
│   │   └── zero_shot_classifier.py
│   └── training/
│       ├── config.py
│       └── trainer.py
└── tests/
    ├── test_data.py
    └── test_models.py
```

### Folder Descriptions
| Path | Description |
| --- | --- |
| `data/` | Raw and processed corpora organized by language or dialect |
| `data/processed/` | Tokenized, normalized, and split-ready artifacts |
| `src/models/` | Model wrappers, zero-shot heads, and ensemble logic |
| `src/data/` | Dataset readers, preprocessing, and utility functions |
| `src/training/` | Training loops, experiment control, and hyperparameter configuration |
| `src/evaluation/` | Metric computation, error analysis, and visual reporting |
| `src/inference/` | Prediction interfaces for single-text and batch inference |
| `results/` | Saved checkpoints, generated predictions, and figures |
| `paper/` | Manuscript drafts, tables, and supporting material |
| `tests/` | Unit and regression tests for data and modeling components |

## Dataset
This repository is designed for multilingual emotion detection across English, Hindi, and Bhojpuri. Because dialect resources are fragmented and not uniformly standardized, the project separates data ingestion from modeling so that corpora can be revised, expanded, or replaced without changing the full training stack.

### Suggested Data Sources
- English: [GoEmotions](https://arxiv.org/abs/2005.00547), a fine-grained human-annotated emotion dataset released by Google Research
- Hindi: project-specific Hindi emotion corpus, translated emotion benchmarks, or manually annotated social media text derived from public sources
- Bhojpuri: a curated low-resource corpus built through manual annotation, weak supervision, or translation-assisted labeling

### Planned Dataset Statistics
| Language | Source | Split Status | Approx. Size |
| --- | --- | --- | --- |
| English | GoEmotions or equivalent emotion corpus | Train and validation | TBD |
| Hindi | Curated or translated social media emotion corpus | Train and validation | TBD |
| Bhojpuri | Project-specific annotated corpus | Validation and test, optionally few-shot train | TBD |

### Data Preparation Workflow
1. Place raw language-specific files into `data/english/`, `data/hindi/`, and `data/bhojpuri/`.
2. Normalize scripts, remove duplicates, and harmonize label names using `src/data/preprocessing.py`.
3. Export cleaned train, validation, and test splits to `data/processed/`.
4. Store metadata such as label maps, source provenance, and annotation notes alongside the processed artifacts.

### Data Considerations
- Maintain a consistent label inventory across languages, for example `joy`, `sadness`, `anger`, `fear`, `surprise`, and `neutral`.
- Track script choices explicitly when working with Devanagari, Latin transliteration, or mixed-script social media text.
- Document annotation quality, inter-annotator agreement, and class imbalance before comparing cross-lingual transfer results.

## Models
The project focuses on multilingual transformer encoders that offer strong cross-lingual transfer performance and practical support through the Hugging Face ecosystem.

| Model | Approx. Size | Pre-Training | Why It Matters |
| --- | --- | --- | --- |
| `bert-base-multilingual-cased` | ~177M parameters | Multilingual Wikipedia across 100+ languages | Strong baseline with wide language coverage and stable fine-tuning behavior |
| `xlm-roberta-base` | ~270M parameters | CommonCrawl-based multilingual pre-training on 100 languages | Often the strongest zero-shot multilingual baseline for transfer-heavy settings |
| `distilbert-base-multilingual-cased` or compact distilled variant | ~130M parameters | Distilled multilingual transformer | Useful when inference latency or memory constraints matter |

### Selection Rationale
- mBERT provides a historically important multilingual baseline with strong reproducibility.
- XLM-RoBERTa generally offers better cross-lingual representation quality for zero-shot transfer.
- Distilled multilingual models help evaluate the trade-off between accuracy, speed, and deployment cost.

### Reference Papers and Resources
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)

## Training and Fine-Tuning
Fine-tuning is expected to follow a standard supervised or few-shot transfer pipeline in which English and Hindi labeled data form the main training signal while Bhojpuri is reserved for zero-shot evaluation or small adaptation experiments.

### Example Training Command
```powershell
python -m src.training.trainer --config config.yaml
```

### Recommended Starting Hyperparameters
| Hyperparameter | Suggested Value |
| --- | --- |
| Learning rate | `2e-5` |
| Batch size | `16` |
| Epochs | `3-5` |
| Max sequence length | `128` or `256` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| Evaluation strategy | End of each epoch |
| Early stopping patience | `2` epochs |

### Fine-Tuning on Custom Data
1. Convert the custom dataset into a tabular or JSON format with `text`, `label`, and `language` fields.
2. Update the dataset paths and label schema in `config.yaml`.
3. Choose a multilingual backbone such as `xlm-roberta-base`.
4. Train on the source languages first, then evaluate zero-shot performance on Bhojpuri.
5. Optionally run few-shot adaptation experiments using small labeled Bhojpuri subsets.

### Expected Training Time
- CPU: several hours for a medium-sized transformer run, depending on dataset size
- Single mid-range GPU such as T4 or RTX 3060: roughly 30 to 120 minutes per model for moderate corpora
- High-end GPUs can reduce this substantially, especially when mixed precision is enabled

## Inference
The inference layer is intended to expose a clean API for single-instance prediction, batch scoring, and downstream integration into notebooks or services.

### Single Text Example
```python
from src.inference.predictor import EmotionPredictor

predictor = EmotionPredictor(model_name="xlm-roberta-base")
prediction = predictor.predict("हमरा ई बात सुन के बहुत दुख भइल।")
print(prediction)
```

### Batch Inference Example
```python
from src.inference.predictor import EmotionPredictor

predictor = EmotionPredictor(model_name="xlm-roberta-base")
predictions = predictor.predict_batch(
    [
        "I am feeling optimistic today.",
        "आज बहुत गुस्सा आ रहा है।",
        "हम खुश बानी कि काम पूरा हो गइल।",
    ],
    batch_size=4,
)
```

### Example Output Format
```json
{
  "text": "आज बहुत गुस्सा आ रहा है।",
  "language": "hindi",
  "label": "anger",
  "confidence": 0.91,
  "scores": {
    "anger": 0.91,
    "sadness": 0.04,
    "neutral": 0.03,
    "joy": 0.02
  }
}
```

## Results
This section is intentionally a placeholder until training and evaluation runs are added to the repository. The values below should be treated as target reporting slots rather than completed claims.

### Placeholder Benchmark Table
| Experiment | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- |
| English in-domain baseline | TBD | TBD | Reference supervised benchmark |
| Hindi transfer baseline | TBD | TBD | Cross-lingual or multilingual fine-tuning |
| Bhojpuri zero-shot evaluation | TBD | TBD | Main target setting |
| Bhojpuri few-shot adaptation | TBD | TBD | Optional low-resource adaptation |

### Planned Comparisons
- Monolingual baseline versus multilingual zero-shot transfer
- mBERT versus XLM-RoBERTa versus distilled multilingual encoder
- Zero-shot transfer versus few-shot adaptation on Bhojpuri
- Macro F1 drop between source-language validation and target-language evaluation

### Expected Bhojpuri Performance
Given the scarcity of high-quality dialectal emotion annotations, a realistic early-stage expectation is that Bhojpuri zero-shot performance will lag behind English and Hindi supervised scores. The primary research objective is not merely absolute accuracy, but understanding transfer robustness, failure modes, and which multilingual architectures degrade most gracefully under dialect shift.

## Evaluation Metrics
The evaluation suite should prioritize class-balanced metrics because emotion datasets are often highly skewed.

### Core Metrics
- Accuracy
- Macro F1-score
- Weighted F1-score
- Precision
- Recall

### Recommended Reporting
- Per-emotion precision, recall, and F1
- Confusion matrix across emotion labels
- Cross-lingual transfer gap between source-language and target-language performance
- Calibration or confidence analysis for uncertain predictions

## Configuration
Although the current repository includes `src/training/config.py`, a top-level `config.yaml` is recommended for experiment reproducibility and clean separation between code and run-time settings.

### Example `config.yaml`
```yaml
project:
  name: zero-shot-cross-lingual-emotion-detection
  seed: 42

data:
  train_path: data/processed/train.json
  validation_path: data/processed/validation.json
  test_path: data/processed/test.json
  labels: [joy, sadness, anger, fear, surprise, neutral]
  max_length: 128

model:
  backbone: xlm-roberta-base
  num_labels: 6
  dropout: 0.1

training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 4
  weight_decay: 0.01
  warmup_ratio: 0.1
  fp16: true

inference:
  threshold: 0.5
  batch_size: 8
```

### Key Hyperparameters
- `backbone`: selects the multilingual encoder used for transfer
- `max_length`: controls truncation and memory footprint
- `learning_rate`: critical for stable transformer fine-tuning
- `batch_size`: balances throughput against GPU memory limits
- `fp16`: improves speed and memory efficiency on supported GPUs
- `threshold`: useful for confidence filtering or multi-label extensions

### How to Modify Settings
1. Adjust model and dataset fields inside `config.yaml`.
2. Keep the label list identical across training, validation, and inference.
3. Increase `max_length` only if the corpus contains longer examples and the hardware budget supports it.
4. Log every configuration used for published or comparative experiments.

## Contributing
Contributions are welcome, especially in dataset curation, preprocessing for dialectal variation, evaluation design, and reproducibility improvements.

### Contribution Workflow
1. Fork the repository and create a feature branch.
2. Implement the change with tests where appropriate.
3. Format the code with Black before opening a pull request.
4. Run the test suite and lint checks locally.
5. Submit a pull request with a concise research or engineering rationale.

### Code Style
```powershell
black src tests
pytest
flake8 src tests
mypy src
```

### Pull Request Expectations
- Keep changes focused and documented.
- Add or update tests for data or model behavior changes.
- Explain whether the change affects reproducibility, metrics, or dataset assumptions.

## Citation
If this repository supports your research, please cite the project and any associated paper once available.

### BibTeX Template
```bibtex
@misc{zero_shot_cross_lingual_emotion_detection_2026,
  title        = {Zero-Shot Cross-Lingual Emotion Detection for Indian Dialects},
  author       = {Project Maintainer},
  year         = {2026},
  howpublished = {\url{https://github.com/your-username/zero-shot-emotion-detection}},
  note         = {GitHub repository}
}
```

### How to Cite
Use the BibTeX entry above for software citation and update it with the finalized author list, URL, and publication metadata once the paper is released.

## License
This project is released under the MIT License.

```text
MIT License

Copyright (c) 2026 Project Maintainer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments
- [GoEmotions](https://arxiv.org/abs/2005.00547) for a strong English emotion-detection benchmark
- [BERT](https://arxiv.org/abs/1810.04805), [XLM-R](https://arxiv.org/abs/1911.02116), and [DistilBERT](https://arxiv.org/abs/1910.01108) for multilingual modeling foundations
- [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for the modeling ecosystem
- Researchers and annotators working on low-resource and Indic NLP

## Contact and Author
The fields below are intentionally easy to update once the repository metadata is finalized.

- Name: `Project Maintainer`
- Email: `your.email@example.com`
- GitHub: `https://github.com/your-username`

## Project Status
This repository is currently in active development. The present codebase provides the initial project scaffold, dependency definition, and documentation needed to begin implementing the full training and evaluation pipeline.
