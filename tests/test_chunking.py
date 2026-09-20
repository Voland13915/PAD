"""
test_chunking.py

Проверяет все три стратегии chunking'а. Использует FakeTokenizer
(разбиение по словам вместо реального HF-токенайзера) — так тесты быстрые
и не требуют скачивания модели, а сама логика алгоритма чанкинга это
не затрагивает: она работает с любым объектом, у которого есть
encode/decode/count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "preprocessing"))
import chunking  # noqa: E402


class FakeTokenizer:
    """1 токен = 1 слово — упрощение для предсказуемых тестов."""

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)

    def count(self, text):
        return len(self.encode(text))


def make_chapter(n_paragraphs=10, words_per_paragraph=10):
    text = "\n".join(
        " ".join(f"слово{i}_{j}" for j in range(words_per_paragraph)) for i in range(n_paragraphs)
    )
    return {
        "document_id": "hp1_ch01",
        "book_number": 1,
        "book_title": "Тестовая книга",
        "chapter_number": 1,
        "chapter_title": "Тестовая глава",
        "text": text,
    }


def test_fixed_size_produces_correct_chunk_count():
    chapter = make_chapter(n_paragraphs=10, words_per_paragraph=10)  # 100 токенов всего
    tokenizer = FakeTokenizer()
    chunks = chunking.chunk_fixed_size(chapter, tokenizer, max_tokens=20)
    assert len(chunks) == 5  # 100 / 20
    assert all(c.token_count == 20 for c in chunks)


def test_fixed_size_overlap_produces_more_chunks_than_no_overlap():
    chapter = make_chapter(n_paragraphs=10, words_per_paragraph=10)
    tokenizer = FakeTokenizer()
    no_overlap = chunking.chunk_fixed_size(chapter, tokenizer, max_tokens=20)
    with_overlap = chunking.chunk_fixed_size_overlap(chapter, tokenizer, max_tokens=20, overlap=5)
    assert len(with_overlap) > len(no_overlap)


def test_paragraph_chunking_respects_target_size():
    chapter = make_chapter(n_paragraphs=10, words_per_paragraph=10)  # каждый абзац = 10 токенов
    tokenizer = FakeTokenizer()
    chunks = chunking.chunk_by_paragraphs(chapter, tokenizer, target_tokens=30)
    assert len(chunks) == 4  # 10 абзацев по 10 токенов -> группы по 3: 3+3+3+1
    # ни один chunk не должен намного превышать target (с учётом жадной упаковки целыми абзацами)
    assert all(c.token_count <= 30 for c in chunks)


def test_paragraph_chunking_splits_oversized_paragraph():
    # Один абзац длиннее лимита — должен быть разрезан отдельно, а не пропущен
    chapter = make_chapter(n_paragraphs=1, words_per_paragraph=50)
    tokenizer = FakeTokenizer()
    chunks = chunking.chunk_by_paragraphs(chapter, tokenizer, target_tokens=10)
    assert len(chunks) > 1
    total_tokens = sum(c.token_count for c in chunks)
    assert total_tokens == 50


def test_all_chunks_preserve_chapter_metadata():
    chapter = make_chapter()
    tokenizer = FakeTokenizer()
    chunks = chunking.chunk_fixed_size(chapter, tokenizer, max_tokens=20)
    for c in chunks:
        assert c.document_id == "hp1_ch01"
        assert c.book_number == 1
        assert c.chapter_title == "Тестовая глава"


def test_clean_text_collapses_whitespace():
    dirty = "Текст   с     лишними\n\n\nпробелами  ."
    cleaned = chunking.clean_text(dirty)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned