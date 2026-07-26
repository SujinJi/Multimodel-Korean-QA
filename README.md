# Multimodal Korean QA System

A multimodal LLM-based question-answering system for Korean

---

## Project Overview

**Multimodal Korean QA System** is a question-answering system that takes
an image and a Korean-language question together as input and generates a
natural Korean answer.

When a user submits an image (a document, a table, a photo, a screenshot,
etc.) along with a Korean question, the model interprets the visual content
of the image together with the text of the question to produce an answer.
The project targets the specific gap of "Korean questions about images,"
which plain text-only QA cannot handle, and includes an evaluation
pipeline to measure answer quality quantitatively.

## Purpose

- **Validate multimodal QA capability in Korean**: Most multimodal LLM
  ecosystems are English-centric; this project builds a QA pipeline
  specifically for Korean input and output.
- **Build model application and evaluation skills**: Apply a pretrained
  multimodal model to a real task — with or without lightweight
  fine-tuning — and go through the full cycle of quantitative evaluation.
- **Explore practical applicability**: Prototype use cases such as document
  understanding or image-based customer inquiry response that could scale
  to a real service.
- **Portfolio purpose**: Demonstrate LLM/VLM modeling, inference pipeline
  design, and evaluation framework skills through a self-directed project.

## My Role

This was a project with a team of 4 or more. My role was **preprocessing
and encoders**:
- Implementing image preprocessing (resizing/normalization) and Korean
  text tokenization
- Designing and training the vision encoder from scratch, using a
  CNN/ViT-based architecture to extract image features for this task
- Designing and training the text encoder from scratch, using an
  LSTM-based architecture to embed Korean questions, built around a
  Korean tokenizer
- Handing off the trained encoder representations to the teammates
  responsible for fusion, decoding, and evaluation

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Korean tokenization behaves differently from English (no whitespace-based word boundaries, agglutinative morphology), which hurt downstream embedding quality when following English-first tokenizer conventions | Built a tokenizer trained on Korean corpora and validated segmentation on sample questions before feeding it into the text encoder |
| No off-the-shelf model was suited to the specific image types (documents, screenshots, not just natural photos), so training a CNN/ViT-based vision encoder from scratch meant starting without strong priors | Trained on a curated dataset spanning document/screenshot/photo inputs and iterated between CNN and ViT-style layers based on feature quality checks, rather than assuming a generic image-classification setup would transfer |
| Training an LSTM-based text encoder for Korean from scratch risked producing embeddings that didn't separate distinct questions well | Evaluated embedding quality directly (e.g. checking that semantically different questions produced clearly separated embeddings) rather than relying on training loss alone |
| Coordinating output dimensions and formats with the teammates building the fusion layer, since both encoders were being trained in parallel with the downstream fusion work | Agreed on embedding dimensions and normalization conventions with the team up front, and iterated with them early rather than discovering a mismatch after training was complete |

## Architecture

```
   Image input                    Korean question
 (doc / photo / screenshot)          (text query)
        │                                  │
        ▼                                  ▼
  Vision encoder                     Text encoder
 (image feature extraction)   (Korean tokenizer + embedding)
        │                                  │
        └───────────────┬──────────────────┘
                         ▼
                Cross-modal fusion
         (aligns image and text vectors)
                         │
                         ▼
                    LLM decoder
           (generates the Korean answer)
                         │
                         ▼
                Evaluation pipeline
        (accuracy / similarity vs. reference answers)
```

**Modules**
1. **Preprocessing**: Handles image preprocessing and Korean text
   tokenization as a unified data pipeline, feeding both encoders below.
2. **Multimodal encoder**: A vision encoder and a text encoder produce
   separate vectors, which the cross-modal fusion layer then combines.
3. **LLM decoder**: Generates a natural Korean-language answer from the
   fused multimodal representation.
4. **Evaluation pipeline**: Compares generated answers against reference
   answers, computes quantitative metrics (accuracy, similarity), and logs
   the results for analysis.

**What I owned in this architecture**

This was a project with a team of 4 or more. My part was preprocessing
and the encoders — the first two stages below.

| Module | Owner |
|---|---|
| Preprocessing | Me — implemented image preprocessing and integrated a Korean-specific tokenizer in place of default English-first tokenization |
| Vision encoder | Me — designed and trained a CNN/ViT-based model from scratch to extract image features for this task |
| Text encoder | Me — designed and trained an LSTM-based model from scratch to embed Korean questions, built on the Korean tokenizer |
| Cross-modal fusion | Teammate |
| LLM decoder | Teammate |
| Evaluation pipeline | Teammate |

## Key Features

| Feature | Description |
|---|---|
| Multimodal input handling | Accepts an image and a Korean-language question together |
| Korean answer generation | Output optimized for natural Korean language generation |
| Performance evaluation pipeline | Includes scripts to quantitatively measure answer accuracy/quality |
| Result logging | Stores input-output-evaluation results for later analysis |

## Tech Stack
- **Language**: Python
- **Framework**: PyTorch
- **Model**: CNN/ViT-based vision encoder and LSTM-based text encoder designed and trained from scratch by me; fusion layer and Korean-capable decoder built by teammates

## Running
```bash
pip install -r requirements.txt
python main.py
```
