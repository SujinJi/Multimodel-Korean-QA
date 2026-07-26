"""Evaluation metrics for Korean QA.

Because Korean word segmentation via a morphological analyzer is
non-trivial and adds a dependency, we use character-level F1 alongside
normalized Exact Match as the default metrics — both are stable without
extra tooling.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from src.data import KoreanQADataset
from src.model import BaseKoreanQAModel


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\.,!?\"'()\[\]{}]", "", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def char_f1(prediction: str, gold: str) -> float:
    pred_chars = list(normalize_answer(prediction))
    gold_chars = list(normalize_answer(gold))

    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0

    common = Counter(pred_chars) & Counter(gold_chars)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_chars)
    recall = num_same / len(gold_chars)
    return 2 * precision * recall / (precision + recall)


def evaluate_model(model: BaseKoreanQAModel, dataset: KoreanQADataset) -> Dict[str, float]:
    em_scores: List[float] = []
    f1_scores: List[float] = []

    for i in range(len(dataset)):
        item = dataset[i]
        prediction = model.answer(item["question"], item.get("image"))

        em_scores.append(1.0 if exact_match(prediction, item["answer"]) else 0.0)
        f1_scores.append(char_f1(prediction, item["answer"]))

    n = len(dataset)
    return {
        "exact_match": sum(em_scores) / n if n else 0.0,
        "char_f1": sum(f1_scores) / n if n else 0.0,
        "num_examples": n,
    }
