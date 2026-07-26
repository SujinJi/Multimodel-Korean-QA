from pathlib import Path

from src.data import KoreanQADataset
from src.evaluate import evaluate_model
from src.model import DummyKoreanQAModel

SAMPLE_DATA = str(Path(__file__).resolve().parent.parent / "sample_data" / "sample_qa.json")


def test_dataset_loads_all_examples():
    dataset = KoreanQADataset(SAMPLE_DATA)
    assert len(dataset) == 5


def test_dataset_modality_split():
    dataset = KoreanQADataset(SAMPLE_DATA)
    assert len(dataset.text_only()) == 2
    assert len(dataset.multimodal_only()) == 3


def test_dummy_model_answers_known_questions():
    model = DummyKoreanQAModel()
    assert model.answer("대한민국의 수도는 어디인가요?") == "서울"
    assert model.answer("완전히 새로운 질문입니다") == "I'm not sure"


def test_end_to_end_evaluation_pipeline_runs():
    dataset = KoreanQADataset(SAMPLE_DATA)
    model = DummyKoreanQAModel()
    results = evaluate_model(model, dataset)

    assert results["num_examples"] == 5
    assert 0.0 <= results["exact_match"] <= 1.0
    assert 0.0 <= results["char_f1"] <= 1.0
