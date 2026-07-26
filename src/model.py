"""Multimodal Korean QA model wrapper.

In production, this would extract visual features with a CLIP-style image
encoder and inject them as a prompt into a Korean generation model (e.g.
polyglot-ko, KoAlpaca) — a vision-to-prompt approach. This repo defines a
`BaseKoreanQAModel` interface so backbones are easy to swap, and offers
`DummyKoreanQAModel` for offline/test environments to validate the
evaluation pipeline without network access.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class BaseKoreanQAModel(ABC):
    @abstractmethod
    def answer(self, question: str, image: Optional[Image.Image] = None) -> str:
        """Return a Korean answer string given a question (and optional image)."""
        raise NotImplementedError


class HFMultimodalKoreanQAModel(BaseKoreanQAModel):
    """Real HuggingFace-backed model wrapper.

    - vision_model_name: a CLIP-style image encoder
    - text_model_name: a Korean generative (causal) LM

    Heavy weight loading is lazy, so pipeline-only work (e.g. testing the
    data pipeline) doesn't trigger a download until the model is actually used.
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "beomi/KoAlpaca-Polyglot-5.8B",
        device: str = "cpu",
    ):
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.device = device
        self._vision_model = None
        self._vision_processor = None
        self._text_model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._text_model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor

        self._vision_model = CLIPModel.from_pretrained(self.vision_model_name).to(self.device)
        self._vision_processor = CLIPProcessor.from_pretrained(self.vision_model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self._text_model = AutoModelForCausalLM.from_pretrained(self.text_model_name).to(self.device)

    def _build_prompt(self, question: str, image_caption: Optional[str]) -> str:
        if image_caption:
            return (
                f"Here is a description of the image: {image_caption}\n"
                f"Question: {question}\n"
                f"Answer:"
            )
        return f"Question: {question}\nAnswer:"

    def _caption_image(self, image: Image.Image) -> str:
        """Convert an image into a short text caption (a simple CLIP-similarity-based approach).

        For production use, a dedicated captioning model such as BLIP is recommended.
        This is a minimal example for validating the pipeline, using similarity
        against a small set of candidate labels only.
        """
        import torch

        candidate_labels = ["sky", "dog", "cat", "people", "food", "car", "tree", "ocean"]
        inputs = self._vision_processor(
            text=candidate_labels, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self._vision_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        best_idx = int(probs.argmax())
        return candidate_labels[best_idx]

    def answer(self, question: str, image: Optional[Image.Image] = None) -> str:
        self._lazy_load()

        image_caption = self._caption_image(image) if image is not None else None
        prompt = self._build_prompt(question, image_caption)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self._text_model.generate(
            **inputs, max_new_tokens=32, do_sample=False
        )
        generated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated.split("Answer:")[-1].strip()


class DummyKoreanQAModel(BaseKoreanQAModel):
    """A dummy model that lets the evaluation pipeline be validated without
    network access.

    Uses simple keyword matching to produce answers. Intended for CI,
    offline testing, and pipeline smoke tests.
    """

    _RULES = {
        "수도": "서울",
        "명절": "추석",
        "하늘": "파란색",
    }

    def answer(self, question: str, image: Optional[Image.Image] = None) -> str:
        for keyword, response in self._RULES.items():
            if keyword in question:
                return response
        return "I'm not sure"
