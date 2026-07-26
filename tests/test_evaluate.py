from src.evaluate import char_f1, exact_match, normalize_answer


def test_normalize_answer_strips_punctuation_and_spaces():
    assert normalize_answer("서울 !") == "서울"
    assert normalize_answer("  파란색.") == "파란색"


def test_exact_match_ignores_whitespace_and_punctuation():
    assert exact_match("서울입니다", "서울입니다") is True
    assert exact_match(" 서울! ", "서울") is True  # whitespace/punctuation differences are ignored
    assert exact_match("서울입니다", "서울") is False  # substantively different text should fail


def test_exact_match_identical():
    assert exact_match("추석", "추석") is True


def test_char_f1_identical_is_one():
    assert char_f1("파란색", "파란색") == 1.0


def test_char_f1_partial_overlap():
    score = char_f1("하늘색", "파란색")
    assert 0.0 < score < 1.0


def test_char_f1_no_overlap_is_zero():
    assert char_f1("사과", "자동차") >= 0.0
    assert char_f1("xyz", "가나다") == 0.0


def test_char_f1_empty_strings():
    assert char_f1("", "") == 1.0
    assert char_f1("", "서울") == 0.0
