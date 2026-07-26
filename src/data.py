"""Multimodal Korean QA dataset loader.

Each JSON entry follows this schema:
{
    "id": str,
    "question": str,
    "answer": str,
    "image_path": Optional[str],   # None for text-only queries
    "modality": "multimodal" | "text_only"
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class QAExample:
    id: str
    question: str
    answer: str
    image_path: Optional[str]
    modality: str


class KoreanQADataset(Dataset):
    """PyTorch Dataset that loads multimodal/text-only Korean QA examples."""

    def __init__(self, json_path: str, load_images: bool = False):
        self.json_path = Path(json_path)
        self.load_images = load_images
        self.examples: List[QAExample] = self._load(self.json_path)

    @staticmethod
    def _load(json_path: Path) -> List[QAExample]:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        examples = []
        for item in raw:
            examples.append(
                QAExample(
                    id=item["id"],
                    question=item["question"],
                    answer=item["answer"],
                    image_path=item.get("image_path"),
                    modality=item.get("modality", "text_only"),
                )
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        item = {
            "id": ex.id,
            "question": ex.question,
            "answer": ex.answer,
            "modality": ex.modality,
        }

        if self.load_images and ex.image_path:
            base_dir = self.json_path.parent.parent  # assumes repo-root-relative paths
            image_full_path = base_dir / ex.image_path
            if image_full_path.exists():
                item["image"] = Image.open(image_full_path).convert("RGB")
            else:
                item["image"] = None
        else:
            item["image"] = None

        return item

    def text_only(self) -> "KoreanQADataset":
        """Return a new dataset view containing only text-only examples (for slice-level eval)."""
        subset = KoreanQADataset.__new__(KoreanQADataset)
        subset.json_path = self.json_path
        subset.load_images = self.load_images
        subset.examples = [e for e in self.examples if e.modality == "text_only"]
        return subset

    def multimodal_only(self) -> "KoreanQADataset":
        subset = KoreanQADataset.__new__(KoreanQADataset)
        subset.json_path = self.json_path
        subset.load_images = self.load_images
        subset.examples = [e for e in self.examples if e.modality == "multimodal"]
        return subset
