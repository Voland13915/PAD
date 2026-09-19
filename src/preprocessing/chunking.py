"""
chunking.py

Preprocessing и chunking для RAG-системы по книгам "Гарри Поттер".

Реализованы 3 стратегии разбиения на chunks (для последующего сравнения
в experiments/ — это и есть исследовательская часть задания):

  1. fixed_size          — фиксированный размер chunk'а в токенах, без overlap.
  2. fixed_size_overlap   — фиксированный размер + overlap между соседними chunks.
  3. paragraph            — разбиение по естественным границам (абзацам),
                             с объединением мелких абзацев до целевого размера.

Размер chunk'а считается в **реальных токенах** embedding-модели
(intfloat/multilingual-e5-base по умолчанию) — не по словам и не по символам,
чтобы это было напрямую привязано к контекстному окну модели, которую
будем использовать для векторизации.

Вход: JSON-файлы data/processed/book_0X.json (результат grabber'а).
Выход: JSON-файлы data/processed/chunks_<strategy>.json.
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Очистка текста
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Базовая нормализация текста главы:
      - схлопывает повторяющиеся пробелы/табы;
      - убирает пробелы перед знаками препинания;
      - убирает пустые строки, оставшиеся после парсинга fb2.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Абстракция токенайзера
# ---------------------------------------------------------------------------

class Tokenizer:
    """
    Обёртка над HuggingFace-токенайзером embedding-модели.

    Используем реальный токенайзер модели, а не подсчёт по словам, — чтобы
    размер chunk'а был напрямую привязан к тому, что модель реально "видит"
    (её лимиту контекста в токенах), а не к произвольной метрике.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text))


# ---------------------------------------------------------------------------
# Модель chunk'а
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    document_id: str          # id родительской главы, напр. hp1_ch01
    book_number: int
    book_title: str
    chapter_number: int
    chapter_title: str
    strategy: str              # fixed_size | fixed_size_overlap | paragraph
    text: str
    token_count: int


# ---------------------------------------------------------------------------
# Стратегия 1: fixed-size (без overlap)
# ---------------------------------------------------------------------------

def chunk_fixed_size(chapter: dict, tokenizer: Tokenizer, max_tokens: int) -> list[Chunk]:
    return _sliding_window(chapter, tokenizer, max_tokens, overlap=0, strategy="fixed_size")


# ---------------------------------------------------------------------------
# Стратегия 2: fixed-size + overlap
# ---------------------------------------------------------------------------

def chunk_fixed_size_overlap(
    chapter: dict, tokenizer: Tokenizer, max_tokens: int, overlap: int
) -> list[Chunk]:
    return _sliding_window(
        chapter, tokenizer, max_tokens, overlap=overlap, strategy="fixed_size_overlap"
    )


def _sliding_window(
    chapter: dict, tokenizer: Tokenizer, max_tokens: int, overlap: int, strategy: str
) -> list[Chunk]:
    text = clean_text(chapter["text"])
    token_ids = tokenizer.encode(text)

    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap должен быть меньше max_tokens")

    chunks: list[Chunk] = []
    start = 0
    idx = 1
    while start < len(token_ids):
        window = token_ids[start : start + max_tokens]
        chunk_text = tokenizer.decode(window)
        chunks.append(
            Chunk(
                chunk_id=f"{chapter['document_id']}_{strategy}_{idx:03d}",
                document_id=chapter["document_id"],
                book_number=chapter["book_number"],
                book_title=chapter["book_title"],
                chapter_number=chapter["chapter_number"],
                chapter_title=chapter["chapter_title"],
                strategy=strategy,
                text=chunk_text,
                token_count=len(window),
            )
        )
        idx += 1
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Стратегия 3: по абзацам (с объединением до целевого размера)
# ---------------------------------------------------------------------------

def chunk_by_paragraphs(chapter: dict, tokenizer: Tokenizer, target_tokens: int) -> list[Chunk]:
    """
    Идём по абзацам главы и жадно объединяем их в chunk, пока не наберём
    ~target_tokens. Если один абзац сам по себе больше target_tokens —
    режем его дополнительно fixed-size окном (редкий случай, но защищаемся).
    """
    paragraphs = [p for p in clean_text(chapter["text"]).split("\n") if p.strip()]

    chunks: list[Chunk] = []
    idx = 1
    buffer_paragraphs: list[str] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer_paragraphs, buffer_tokens, idx
        if not buffer_paragraphs:
            return
        chunk_text = "\n".join(buffer_paragraphs)
        chunks.append(
            Chunk(
                chunk_id=f"{chapter['document_id']}_paragraph_{idx:03d}",
                document_id=chapter["document_id"],
                book_number=chapter["book_number"],
                book_title=chapter["book_title"],
                chapter_number=chapter["chapter_number"],
                chapter_title=chapter["chapter_title"],
                strategy="paragraph",
                text=chunk_text,
                token_count=buffer_tokens,
            )
        )
        idx += 1
        buffer_paragraphs = []
        buffer_tokens = 0

    for para in paragraphs:
        para_tokens = tokenizer.count(para)

        if para_tokens > target_tokens:
            # Абзац сам по себе больше лимита — сначала сбрасываем буфер,
            # затем режем этот абзац отдельно как mini fixed-size chunk.
            flush()
            fake_chapter = {**chapter, "text": para}
            sub_chunks = _sliding_window(
                fake_chapter, tokenizer, target_tokens, overlap=0, strategy="paragraph"
            )
            for sc in sub_chunks:
                sc.chunk_id = f"{chapter['document_id']}_paragraph_{idx:03d}"
                idx += 1
            chunks.extend(sub_chunks)
            continue

        if buffer_tokens + para_tokens > target_tokens:
            flush()

        buffer_paragraphs.append(para)
        buffer_tokens += para_tokens

    flush()
    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_chapters(processed_dir: Path) -> list[dict]:
    chapters = []
    for book_file in sorted(processed_dir.glob("book_*.json")):
        chapters.extend(json.loads(book_file.read_text(encoding="utf-8")))
    return chapters


def run_chunking(
    processed_dir: str,
    out_dir: str,
    model_name: str,
    fixed_size: int,
    overlap: int,
    paragraph_target: int,
) -> None:
    processed_path = Path(processed_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chapters = load_chapters(processed_path)
    if not chapters:
        print(f"В {processed_dir} не найдено ни одной обработанной главы. Сначала запусти grabber.")
        return

    tokenizer = Tokenizer(model_name)

    strategies = {
        "fixed_size": lambda ch: chunk_fixed_size(ch, tokenizer, fixed_size),
        "fixed_size_overlap": lambda ch: chunk_fixed_size_overlap(ch, tokenizer, fixed_size, overlap),
        "paragraph": lambda ch: chunk_by_paragraphs(ch, tokenizer, paragraph_target),
    }

    for strategy_name, fn in strategies.items():
        all_chunks: list[Chunk] = []
        for chapter in chapters:
            all_chunks.extend(fn(chapter))

        out_file = out_path / f"chunks_{strategy_name}.json"
        out_file.write_text(
            json.dumps([asdict(c) for c in all_chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
        print(
            f"[{strategy_name}] chunks: {len(all_chunks)}, "
            f"средний размер: {avg_tokens:.0f} токенов -> {out_file.name}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunking для глав Гарри Поттер")
    parser.add_argument("--processed-dir", default="../../data/processed")
    parser.add_argument("--out-dir", default="../../data/processed")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--fixed-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--paragraph-target", type=int, default=400)
    args = parser.parse_args()

    run_chunking(
        args.processed_dir,
        args.out_dir,
        args.model,
        args.fixed_size,
        args.overlap,
        args.paragraph_target,
    )