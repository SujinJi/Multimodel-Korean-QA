"""Preprocessing for raw QA data.

Crowdsourced/collected raw QA records often have inconsistent formatting
(stray whitespace/control characters, duplicates, missing required
fields), so we run a cleaning pass before training/evaluation.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


def clean_text(text: str) -> str:
    """Strip leading/trailing whitespace, collapse repeated whitespace, remove control characters."""
    if text is None:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_example(record: Dict) -> bool:
    """Check that required fields (question/answer) are non-empty."""
    question = record.get("question")
    answer = record.get("answer")
    if not question or not str(question).strip():
        return False
    if not answer or not str(answer).strip():
        return False
    return True


def normalize_record(record: Dict) -> Dict:
    normalized = dict(record)
    normalized["question"] = clean_text(record.get("question", ""))
    normalized["answer"] = clean_text(record.get("answer", ""))
    normalized["modality"] = record.get("modality") or (
        "multimodal" if record.get("image_path") else "text_only"
    )
    return normalized


def dedupe_records(records: List[Dict]) -> List[Dict]:
    """Remove exact duplicate records based on (question, answer, image_path). Keeps first occurrence."""
    seen = set()
    deduped = []
    for record in records:
        key = (record.get("question"), record.get("answer"), record.get("image_path"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def preprocess_records(raw_records: List[Dict]) -> List[Dict]:
    """Raw record list -> cleaned, normalized, deduplicated record list."""
    cleaned = [normalize_record(r) for r in raw_records if is_valid_example(r)]
    return dedupe_records(cleaned)
