"""
test_filtering.py

Проверяет каждый механизм фильтрации по отдельности и всю связку apply_filters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "filtering"))
import filtering as ft  # noqa: E402


def make_candidates():
    return [
        {"text": "Уникальный релевантный фрагмент текста номер один.", "score": 0.9, "reranker_score": 5.0, "book_number": 1},
        {"text": "Уникальный релевантный фрагмент текста номер один.", "score": 0.85, "reranker_score": 4.5, "book_number": 1},  # дубликат
        {"text": "Да.", "score": 0.8, "reranker_score": 4.0, "book_number": 2},  # слишком короткий
        {"text": "Ещё один релевантный и достаточно длинный фрагмент текста.", "score": 0.3, "reranker_score": -2.0, "book_number": 3},  # низкий score
        {"text": "Третий нормальный фрагмент текста подходящей длины.", "score": 0.6, "reranker_score": 1.0, "book_number": 1},
    ]


def test_filter_by_threshold():
    candidates = make_candidates()
    result = ft.filter_by_threshold(candidates, "score", min_score=0.5)
    assert len(result) == 4  # исключён только score=0.3
    assert all(c["score"] >= 0.5 for c in result)


def test_deduplicate_removes_exact_text_duplicates():
    candidates = make_candidates()
    result = ft.deduplicate(candidates)
    texts = [c["text"] for c in result]
    assert len(texts) == len(set(texts))
    assert len(result) == 4  # было 5, один дубликат убран


def test_filter_short_removes_below_min_length():
    candidates = make_candidates()
    result = ft.filter_short(candidates, min_length=20)
    assert all(len(c["text"]) >= 20 for c in result)
    assert len(result) == 4  # "Да." исключено


def test_filter_by_metadata_book_number():
    candidates = make_candidates()
    result = ft.filter_by_metadata(candidates, book_number=1)
    assert all(c["book_number"] == 1 for c in result)
    assert len(result) == 3


def test_filter_by_metadata_none_is_noop():
    candidates = make_candidates()
    result = ft.filter_by_metadata(candidates, book_number=None)
    assert len(result) == len(candidates)


def test_limit_results():
    candidates = make_candidates()
    result = ft.limit_results(candidates, max_results=2)
    assert len(result) == 2
    assert result == candidates[:2]


def test_apply_filters_full_pipeline():
    candidates = make_candidates()
    config = ft.FilterConfig(
        score_field="score", min_score=0.5, min_text_length=20, max_results=10, book_number=None
    )
    result = ft.apply_filters(candidates, config)
    # Должны остаться: убран дубликат, убран короткий, убран низкий score (0.3 < 0.5)
    assert len(result) == 2
    texts = [c["text"] for c in result]
    assert len(texts) == len(set(texts))
    assert all(len(t) >= 20 for t in texts)