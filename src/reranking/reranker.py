"""
reranker.py

Reranker для RAG-системы по книгам "Гарри Поттер".

Retriever (FAISS + embeddings) — быстрый, но грубый способ поиска:
он сравнивает вектор запроса с векторами чанков независимо друг от друга
(bi-encoder). Reranker — это cross-encoder: он подаёт пару
(вопрос, текст_чанка) в модель ВМЕСТЕ, и модель, видя их одновременно,
даёт более точную оценку релевантности. Дороже по вычислениям, поэтому
применяется не ко всей базе, а только к top-N кандидатов от retriever'а.

Пайплайн:
    Query -> Retriever (top-20) -> Reranker (пересортировка) -> top-5 -> LLM
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
import retriever as rt  # noqa: E402


class Reranker:
    """Обёртка над cross-encoder моделью для пересортировки кандидатов."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        print(f"Загружаю reranker {model_name}...")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """
        candidates — результаты retriever'а (список dict с полем 'text').
        Возвращает top_n кандидатов, пересортированных по reranker_score
        (добавляется как отдельное поле, исходный retriever 'score' сохраняется).
        """
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self._model.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
        return reranked[:top_n]


def compare_with_without_reranker(
    retriever: rt.Retriever, reranker: Reranker, question: str, retrieve_k: int = 20, final_n: int = 5
) -> None:
    """
    Печатает бок о бок: top-N от чистого retriever'а vs top-N после
    retriever(top-K) -> reranker -> top-N. Наглядно показывает, меняет ли
    reranker порядок/состав финальных результатов.
    """
    retriever_only = retriever.search(question, top_k=final_n)
    retrieve_candidates = retriever.search(question, top_k=retrieve_k)
    reranked = reranker.rerank(question, retrieve_candidates, top_n=final_n)

    print(f"\n{'=' * 70}\nВопрос: {question}\n{'=' * 70}")

    print(f"\n-- Только retriever (top-{final_n}) --")
    for i, r in enumerate(retriever_only, start=1):
        print(f"  {i}. [{r['score']:.4f}] книга {r['book_number']}, {r['chapter_title']}")

    print(f"\n-- Retriever(top-{retrieve_k}) + Reranker -> top-{final_n} --")
    for i, r in enumerate(reranked, start=1):
        print(
            f"  {i}. [rerank={r['reranker_score']:.4f}, "
            f"orig_score={r['score']:.4f}] книга {r['book_number']}, {r['chapter_title']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranker для RAG по Гарри Поттеру")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed_size", "fixed_size_overlap", "paragraph"])
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--final-n", type=int, default=5)
    args = parser.parse_args()

    retriever_ = rt.Retriever(args.faiss_dir, args.strategy, args.embedding_model)
    reranker_ = Reranker(args.reranker_model)

    test_questions = [
        "Кто убил Дамблдора?",
        "Как зовут домашнего эльфа Добби?",
        "Что такое философский камень?",
        "Кто такой Сириус Блэк?",
        "Какое заклинание убивает?",
    ]

    for q in test_questions:
        compare_with_without_reranker(retriever_, reranker_, q, args.retrieve_k, args.final_n)