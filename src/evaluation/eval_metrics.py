"""
eval_metrics.py

Считает метрики качества retrieval по набору eval_questions.json:
  - Recall@K   — доля вопросов, для которых хотя бы один релевантный документ
                 попал в top-K (для вопросов с одним релевантным документом
                 это совпадает с Hit Rate@K).
  - Precision@K— доля релевантных документов среди top-K.
  - MRR        — Mean Reciprocal Rank: 1/позиция первого релевантного документа,
                 усреднённое по всем вопросам.
  - Hit Rate@K — доля вопросов, где вообще нашёлся хоть один релевантный документ в top-K.

Вопросы типа "no_answer" (relevant_document_ids == []) исключаются из этих
метрик (для них нет "правильного" документа by design) и оцениваются
отдельно — вручную, при генерации ответа: система ДОЛЖНА сказать
"информации недостаточно", а не выдумать ответ (это и есть defense от галлюцинаций,
п.13 задания, но эта проверка требует запущенной LLM-генерации, а не только retrieval).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
import retriever as rt  # noqa: E402


def load_questions(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["questions"]


def evaluate(retriever: rt.Retriever, questions: list[dict], k_values: list[int]) -> dict:
    """
    Возвращает {k: {"recall": ..., "precision": ..., "hit_rate": ...}} + отдельно MRR
    (MRR не зависит от K в этой реализации — считается по самому глубокому K).
    """
    # Оцениваем только вопросы с известным релевантным документом.
    scored_questions = [
        q for q in questions if q.get("relevant_document_ids")
    ]
    skipped = [q for q in questions if not q.get("relevant_document_ids")]

    if skipped:
        print(
            f"[!] Пропущено {len(skipped)} вопросов без заполненных relevant_document_ids "
            f"(в т.ч. все вопросы типа 'no_answer' — это ожидаемо):"
        )
        for q in skipped:
            if q["type"] != "no_answer":
                print(f"    - {q['id']} ({q['type']}): {q['question']}  <- ЗАПОЛНИ relevant_document_ids")

    if not scored_questions:
        print("Нет вопросов с заполненными relevant_document_ids — нечего считать.")
        return {}

    max_k = max(k_values)
    reciprocal_ranks = []
    results_per_k = {k: {"recall_hits": 0, "precision_sum": 0.0} for k in k_values}

    for q in scored_questions:
        relevant = set(q["relevant_document_ids"])
        retrieved = retriever.search(q["question"], top_k=max_k)
        retrieved_doc_ids = [r["document_id"] for r in retrieved]

        # MRR: позиция первого релевантного документа (1-indexed)
        rank = next(
            (i + 1 for i, doc_id in enumerate(retrieved_doc_ids) if doc_id in relevant),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        for k in k_values:
            top_k_ids = retrieved_doc_ids[:k]
            hit = any(doc_id in relevant for doc_id in top_k_ids)
            if hit:
                results_per_k[k]["recall_hits"] += 1

            relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant)
            results_per_k[k]["precision_sum"] += relevant_in_top_k / k

    n = len(scored_questions)
    summary = {
        "n_questions_evaluated": n,
        "mrr": sum(reciprocal_ranks) / n,
        "per_k": {
            k: {
                "recall": results_per_k[k]["recall_hits"] / n,
                "precision": results_per_k[k]["precision_sum"] / n,
            }
            for k in k_values
        },
    }
    return summary


def print_summary(summary: dict, label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(f"Оценено вопросов: {summary['n_questions_evaluated']}")
    print(f"MRR: {summary['mrr']:.4f}")
    for k, metrics in summary["per_k"].items():
        print(f"  K={k:<3} Recall@K={metrics['recall']:.4f}  Precision@K={metrics['precision']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка качества retrieval по eval_questions.json")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed_size", "fixed_size_overlap", "paragraph"])
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10, 20])
    args = parser.parse_args()

    questions = load_questions(args.questions)
    retriever_ = rt.Retriever(args.faiss_dir, args.strategy, args.model)

    summary = evaluate(retriever_, questions, args.k_values)
    if summary:
        print_summary(summary, f"{args.strategy} / {args.model}")