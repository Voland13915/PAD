"""
retriever.py

Retriever для RAG-системы по книгам "Гарри Поттер".

Поверх embed_store.py (загрузка FAISS-индекса + поиск) добавляет:
  - удобный класс Retriever, который один раз грузит модель+индекс
    и дальше просто отвечает на вопросы;
  - CLI для интерактивного тестирования ("вручную позадавать вопросы");
  - режим сравнения разных K на одном и том же наборе вопросов
    (для исследовательской части: "9. Retriever" -> исследовать влияние K).
"""

import argparse
import sys
from pathlib import Path

# embed_store.py лежит в соседней папке src/embeddings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
import embed_store as es  # noqa: E402


class Retriever:
    """
    Обёртка над (embedding-модель + FAISS-индекс) для конкретной комбинации
    (стратегия chunking'а, embedding-модель). Модель и индекс грузятся один
    раз в конструкторе — дальше .search() быстрый.
    """

    def __init__(self, faiss_dir: str, strategy: str, model_name: str):
        self.strategy = strategy
        self.model_name = model_name
        model_short = es._short_name(model_name)
        self.index_name = f"{strategy}__{model_short}"

        print(f"Загружаю модель {model_name}...")
        self.model = es.EmbeddingModel(model_name)

        print(f"Загружаю индекс {self.index_name}...")
        self.index, self.metadata = es.load_index(Path(faiss_dir), self.index_name)
        print(f"Готово: {self.index.ntotal} векторов в индексе.\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return es.search(query, self.model, self.index, self.metadata, top_k=top_k)


def print_results(results: list[dict]) -> None:
    for i, r in enumerate(results, start=1):
        print(
            f"{i}. [score={r['score']:.4f}] книга {r['book_number']} "
            f"«{r['book_title']}», {r['chapter_title']}"
        )
        preview = r["text"][:150].replace("\n", " ")
        print(f"   {preview}...")
        print()


def interactive_mode(retriever: Retriever, top_k: int) -> None:
    print("Введи вопрос (или 'exit' для выхода):\n")
    while True:
        query = input("> ").strip()
        if query.lower() in ("exit", "quit", "выход"):
            break
        if not query:
            continue
        results = retriever.search(query, top_k=top_k)
        print()
        print_results(results)


def compare_k(retriever: Retriever, questions: list[str], k_values: list[int]) -> None:
    """
    Прогоняет один и тот же список вопросов через retriever с разными K,
    печатает для каждого вопроса, что нашлось на каждом K — чтобы визуально
    сравнить, как меняется набор результатов с ростом K.
    """
    for question in questions:
        print(f"\n{'=' * 70}\nВопрос: {question}\n{'=' * 70}")
        for k in k_values:
            results = retriever.search(question, top_k=k)
            top_chapters = [f"{r['book_number']}/{r['chapter_number']}" for r in results]
            avg_score = sum(r["score"] for r in results) / len(results)
            print(f"  K={k:<3} avg_score={avg_score:.4f}  главы: {top_chapters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retriever для RAG по Гарри Поттеру")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed_size", "fixed_size_overlap", "paragraph"])
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--compare-k",
        action="store_true",
        help="Режим сравнения K=3/5/10/20 на нескольких тестовых вопросах вместо интерактивного режима",
    )
    args = parser.parse_args()

    retriever = Retriever(args.faiss_dir, args.strategy, args.model)

    if args.compare_k:
        test_questions = [
            "Кто убил Дамблдора?",
            "Как зовут домашнего эльфа Добби?",
            "Что такое философский камень?",
            "Кто такой Сириус Блэк?",
            "Какое заклинание убивает?",
        ]
        compare_k(retriever, test_questions, k_values=[3, 5, 10, 20])
    else:
        interactive_mode(retriever, args.top_k)