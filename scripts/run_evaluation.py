"""Example usage:
    python scripts/run_evaluation.py --data sample_data/sample_qa.json --dummy
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import KoreanQADataset
from src.evaluate import evaluate_model
from src.model import DummyKoreanQAModel, HFMultimodalKoreanQAModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to evaluation JSON data")
    parser.add_argument("--dummy", action="store_true", help="Use the dummy model to validate the pipeline only")
    parser.add_argument("--load-images", action="store_true")
    args = parser.parse_args()

    dataset = KoreanQADataset(args.data, load_images=args.load_images)
    model = DummyKoreanQAModel() if args.dummy else HFMultimodalKoreanQAModel()

    results = evaluate_model(model, dataset)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
