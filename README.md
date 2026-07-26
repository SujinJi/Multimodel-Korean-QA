# Multimodal Korean QA

## About
Research on a multimodal, LLM-based Korean question-answering system —
originally a 3-person team project (university, web frontend). This
repository is an individual reimplementation of the **data preprocessing
and modeling** portion I was responsible for. Code owned by teammates
(e.g. the web frontend) is not included here.

## My Role
- Preprocessing of QA data for training/evaluation (text-image pair cleaning, format normalization)
- Design and implementation of the multimodal (text + image) QA model
- Development of the model evaluation pipeline (Exact Match, character-level F1)

## Overview
A research codebase for a multimodal (text + image) LLM-based Korean QA
system. Visual information is summarized with a CLIP-style image
encoder and injected as a prompt into a Korean generation model (causal
LM) — a vision-to-prompt approach. Beyond modeling, the **evaluation
pipeline (EM / character-level F1)** is kept as a separate module so
different backbones can be swapped in and compared.

## Tech Stack

**Team project overall**
| Area | Tech |
|---|---|
| Frontend | Web (teammate's part) |
| Modeling/preprocessing (my part) | Python, PyTorch, HuggingFace Transformers, CLIP |

**This repository (my preprocessing/modeling, reimplemented)**
- Python, PyTorch, HuggingFace Transformers
- CLIP (image encoding), Korean causal LM (text generation)
- Pillow (image loading), pytest

## Structure
```
src/
  preprocess.py # Cleaning/normalization/deduplication of raw QA records
  data.py       # Multimodal/text-only QA dataset loader (PyTorch Dataset)
  model.py      # BaseKoreanQAModel interface + HF backbone + offline DummyModel
  evaluate.py   # Exact Match / character-level F1 metrics
  train.py      # Fine-tuning loop skeleton
scripts/
  run_evaluation.py   # CLI evaluation script
sample_data/
  sample_qa.json      # 5 example Korean QA items (2 text-only / 3 multimodal)
tests/
  test_preprocess.py   # Unit tests for the preprocessing module
  test_evaluate.py    # Unit tests for evaluation metrics
  test_pipeline.py     # End-to-end test from data loading to evaluation
```

## Evaluation Metrics
Since Korean word segmentation via a morphological analyzer adds
complexity, the following two metrics are used for stable comparison
without that dependency:
- **Exact Match (EM)**: exact match after whitespace/punctuation normalization
- **Character-level F1**: harmonic mean of precision/recall over character sets between prediction and gold answer

## Running
```bash
pip install -r requirements.txt

# Validate the pipeline offline/as a smoke test with the dummy model
python scripts/run_evaluation.py --data sample_data/sample_qa.json --dummy

# Evaluate with the real HF backbone (requires downloading model weights)
python scripts/run_evaluation.py --data sample_data/sample_qa.json
```

## Tests
```bash
pytest
```

## Design Notes
- `BaseKoreanQAModel` is defined as an interface so other multimodal
  backbones (BLIP-2, Qwen-VL, etc.) can easily replace the CLIP+KoAlpaca combo.
- `DummyKoreanQAModel` exists to validate the evaluation pipeline itself
  in network-restricted CI environments.
- The current image captioning is a minimal implementation comparing
  CLIP similarity against candidate labels; a dedicated captioning
  model such as BLIP is recommended for production use.

## Future Improvements
- Fine-tune and benchmark on a real Korean multimodal QA dataset (e.g. KVQA-Ko)
- Add semantic metrics such as KoBERTScore alongside character-level F1
- Support lightweight fine-tuning via LoRA
