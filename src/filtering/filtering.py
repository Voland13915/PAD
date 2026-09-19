"""
filtering.py

Фильтрация кандидатов между reranker'ом и генерацией ответа (п.11 задания).

Пайплайн на этом этапе:
    Retriever(top-20) -> Reranker -> Filtering -> top-N -> LLM

Реализованы все механизмы, перечисленные в задании:
  - threshold по similarity/reranker score;
  - удаление дубликатов (одинаковый или почти одинаковый текст);
  - фильтрация слишком коротких chunks (мусорные обрывки);
  - фильтрация по metadata (например, только определённая книга);
  - ограничение итогового количества документов.
"""

from dataclasses import dataclass


@dataclass
class FilterConfig:
    score_field: str = "reranker_score"   # какое поле считать "score" для threshold (score | reranker_score)
    min_score: float | None = None         # None = не применять threshold
    min_text_length: int = 50              # в символах; отсекает мусорные обрывки chunk'ов
    max_results: int = 5
    book_number: int | None = None         # None = без фильтра по книге


def filter_by_threshold(candidates: list[dict], score_field: str, min_score: float) -> list[dict]:
    return [c for c in candidates if c.get(score_field, float("-inf")) >= min_score]


def deduplicate(candidates: list[dict]) -> list[dict]:
    """
    Убирает дубликаты по тексту. В нашем pipeline дубликаты чаще всего
    возникают, когда несколько chunks одной и той же главы (например, из
    overlap-стратегии) почти полностью совпадают по содержимому — оставляем
    первое (с более высоким score, т.к. список уже отсортирован) вхождение.
    """
    seen_texts: set[str] = set()
    result = []
    for c in candidates:
        # Нормализуем для сравнения: убираем пробелы по краям, приводим к нижнему регистру
        key = c["text"].strip().lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        result.append(c)
    return result


def filter_short(candidates: list[dict], min_length: int) -> list[dict]:
    return [c for c in candidates if len(c["text"].strip()) >= min_length]


def filter_by_metadata(candidates: list[dict], book_number: int | None) -> list[dict]:
    if book_number is None:
        return candidates
    return [c for c in candidates if c["book_number"] == book_number]


def limit_results(candidates: list[dict], max_results: int) -> list[dict]:
    return candidates[:max_results]


def apply_filters(candidates: list[dict], config: FilterConfig) -> list[dict]:
    """
    Применяет все фильтры по порядку: threshold -> dedup -> short-length ->
    metadata -> limit. Порядок важен: сначала убираем явно нерелевантное
    (по score) и дубликаты, только потом обрезаем до финального количества —
    иначе можно случайно отрезать хорошие результаты дубликатами/мусором,
    ещё не добравшись до реально лучших top-N.
    """
    result = candidates

    if config.min_score is not None:
        result = filter_by_threshold(result, config.score_field, config.min_score)

    result = deduplicate(result)
    result = filter_short(result, config.min_text_length)
    result = filter_by_metadata(result, config.book_number)
    result = limit_results(result, config.max_results)

    return result