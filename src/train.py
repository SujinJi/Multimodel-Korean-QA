"""Training skeleton.

Actual fine-tuning requires a GPU environment and downloading pretrained
weights, so this script exists to make the training loop structure
(dataloader -> forward -> loss -> step) explicit. When extending this
project, add a trainable forward/loss inside `HFMultimodalKoreanQAModel`
and wire it up here.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data import KoreanQADataset


def collate_fn(batch):
    return {
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "images": [b["image"] for b in batch],
    }


def train_loop(data_path: str, epochs: int, batch_size: int, lr: float):
    dataset = KoreanQADataset(data_path, load_images=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"[train] {len(dataset)} examples, {epochs} epochs, batch_size={batch_size}, lr={lr}")

    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(loader):
            # NOTE: hook up forward/backward/optimizer.step() here for real training.
            # loss = model.compute_loss(batch["questions"], batch["images"], batch["answers"])
            # loss.backward()
            # optimizer.step()
            dummy_loss = torch.tensor(0.0)
            running_loss += dummy_loss.item()

        print(f"[epoch {epoch + 1}/{epochs}] avg_loss={running_loss / max(len(loader), 1):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    train_loop(args.data, args.epochs, args.batch_size, args.lr)
