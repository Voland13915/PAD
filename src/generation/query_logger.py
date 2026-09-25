"""
query_logger.py

Структурированное логирование запросов к RAG-системе. Каждый вызов
pipeline.answer() дописывает одну JSON-строку в logs/query_log.jsonl —
формат JSON Lines (по объекту на строку) выбран специально: файл можно
как читать построчно программой, так и просматривать в текстовом
редакторе, и он не ломается, если процесс прервётся посреди записи
следующей строки (в отличие от одного большого JSON-массива).

Логируются: время, вопрос, конфигурация (strategy/model/temperature),
сколько кандидатов нашёл retriever и сколько дошло до LLM после
фильтрации, сам ответ, источники и время выполнения запроса — этого
достаточно, чтобы потом разобрать, как вела себя система в реальном
использовании (полезно и для отладки, и как материал для отчёта).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "query_log.jsonl"

# Обычный logging — для человекочитаемого вывода в консоль параллельно с JSON-логом
logger = logging.getLogger("rag_pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


def log_query(
    question: str,
    strategy: str,
    embedding_model: str,
    llm_model: str,
    temperature: float,
    use_reranker: bool,
    n_candidates_retrieved: int,
    n_chunks_used: int,
    answer: str,
    sources: list[dict],
    duration_seconds: float,
    error: str | None = None,
) -> None:
    """Пишет одну запись в logs/query_log.jsonl и дублирует краткую сводку в консоль."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "config": {
            "strategy": strategy,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "temperature": temperature,
            "use_reranker": use_reranker,
        },
        "n_candidates_retrieved": n_candidates_retrieved,
        "n_chunks_used": n_chunks_used,
        "answer": answer,
        "sources": sources,
        "duration_seconds": round(duration_seconds, 2),
        "error": error,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if error:
        logger.error(f"Запрос завершился ошибкой ({duration_seconds:.2f}s): {question!r} -> {error}")
    else:
        logger.info(
            f"Запрос обработан за {duration_seconds:.2f}s: {question!r} "
            f"(использовано {n_chunks_used}/{n_candidates_retrieved} фрагментов)"
        )