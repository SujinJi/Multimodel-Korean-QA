from src.preprocess import (
    clean_text,
    dedupe_records,
    is_valid_example,
    normalize_record,
    preprocess_records,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("  서울   입니다  ") == "서울 입니다"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_is_valid_example_rejects_missing_fields():
    assert is_valid_example({"question": "질문", "answer": "답"}) is True
    assert is_valid_example({"question": "", "answer": "답"}) is False
    assert is_valid_example({"question": "질문", "answer": ""}) is False
    assert is_valid_example({"question": "질문"}) is False


def test_normalize_record_infers_modality():
    record = {"question": " 질문 ", "answer": " 답 ", "image_path": "img.jpg"}
    normalized = normalize_record(record)
    assert normalized["question"] == "질문"
    assert normalized["modality"] == "multimodal"

    record2 = {"question": "질문", "answer": "답", "image_path": None}
    normalized2 = normalize_record(record2)
    assert normalized2["modality"] == "text_only"


def test_dedupe_records_removes_exact_duplicates():
    records = [
        {"question": "q", "answer": "a", "image_path": None},
        {"question": "q", "answer": "a", "image_path": None},
        {"question": "q2", "answer": "a", "image_path": None},
    ]
    deduped = dedupe_records(records)
    assert len(deduped) == 2


def test_preprocess_records_end_to_end():
    raw = [
        {"question": " 대한민국 수도는? ", "answer": " 서울 ", "image_path": None},
        {"question": "", "answer": "invalid data"},
        {"question": " 대한민국 수도는? ", "answer": " 서울 ", "image_path": None},  # duplicate
    ]
    result = preprocess_records(raw)
    assert len(result) == 1
    assert result[0]["question"] == "대한민국 수도는?"
    assert result[0]["modality"] == "text_only"
