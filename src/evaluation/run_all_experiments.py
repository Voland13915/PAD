"""
run_all_experiments.py

Прогоняет eval_questions.json через ВСЕ 6 построенных FAISS-индексов
(3 стратегии chunking'а x 2 embedding-модели) и сохраняет сравнительную
таблицу метрик в experiments/retrieval_comparison.csv — это и есть
итоговый артефакт исследовательской части (п.14 задания: сравнение
chunking-стратегий и embedding-моделей).

Модель для каждого имени грузится один раз и переиспользуется для всех
трёх стратегий (индексы разные, но модель, которой считать embedding
запроса, — та же).
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
import retriever as rt  # noqa: E402
import embed_store as es  # noqa: E402
import eval_metrics as em  # noqa: E402


STRATEGIES = ["fixed_size", "fixed_size_overlap", "paragraph"]
MODELS = ["intfloat/multilingual-e5-base", "paraphrase-multilingual-mpnet-base-v2"]
K_VALUES = [3, 5, 10, 20]


def run_all(questions_path: str, faiss_dir: str, out_csv: str) -> None:
    questions = em.load_questions(questions_path)
    faiss_path = Path(faiss_dir)

    rows = []

    for model_name in MODELS:
        print(f"\n{'#' * 70}\nМодель: {model_name}\n{'#' * 70}")
        embedding_model = es.EmbeddingModel(model_name)  # грузим один раз на 3 стратегии

        for strategy in STRATEGIES:
            model_short = es._short_name(model_name)
            index_name = f"{strategy}__{model_short}"
            index_path = faiss_path / f"{index_name}.index"
            if not index_path.exists():
                print(f"[!] Индекс {index_name} не найден, пропуск")
                continue

            # Собираем Retriever вручную (минуя повторную загрузку модели)
            retriever = rt.Retriever.__new__(rt.Retriever)
            retriever.strategy = strategy
            retriever.model_name = model_name
            retriever.index_name = index_name
            retriever.model = embedding_model
            retriever.index, retriever.metadata = es.load_index(faiss_path, index_name)

            summary = em.evaluate(retriever, questions, K_VALUES)
            if not summary:
                continue

            row = {
                "strategy": strategy,
                "model": model_short,
                "n_questions": summary["n_questions_evaluated"],
                "mrr": round(summary["mrr"], 4),
            }
            for k in K_VALUES:
                row[f"recall@{k}"] = round(summary["per_k"][k]["recall"], 4)
                row[f"precision@{k}"] = round(summary["per_k"][k]["precision"], 4)
            rows.append(row)

            em.print_summary(summary, f"{strategy} / {model_short}")

    # Сохраняем таблицу
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nСохранено: {out_path}")

    # Печатаем markdown-таблицу для прямой вставки в README/отчёт
    print("\n\n--- Markdown-таблица (для отчёта) ---\n")
    if rows:
        headers = list(rows[0].keys())
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for row in rows:
            print("| " + " | ".join(str(row[h]) for h in headers) + " |")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Сравнение всех комбинаций strategy x model на eval_questions.json")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--out-csv", default="../../experiments/retrieval_comparison.csv")
    args = parser.parse_args()

    run_all(args.questions, args.faiss_dir, args.out_csv)