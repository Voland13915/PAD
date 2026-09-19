"""
eval_reranker.py

Количественное сравнение "только retriever" vs "retriever + reranker"
на том же наборе eval_questions.json и тех же метриках (Recall@K, Precision@K, MRR),
что и eval_metrics.py — чтобы можно было прямо сравнить цифры (п.10 задания:
показать влияние reranking на качество системы).

Пайплайн для варианта "с reranker":
    Query -> Retriever(top-20) -> Reranker -> пересортированный список (all 20)
далее из пересортированного списка берём top-K для тех же K, что и в baseline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reranking"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import retriever as rt  # noqa: E402
import reranker as rr  # noqa: E402
import eval_metrics as em  # noqa: E402


def evaluate_with_reranker(
    retriever: rt.Retriever,
    reranker: rr.Reranker,
    questions: list[dict],
    k_values: list[int],
    retrieve_k: int = 20,
) -> dict:
    scored_questions = [q for q in questions if q.get("relevant_document_ids")]

    n = len(scored_questions)
    reciprocal_ranks = []
    results_per_k = {k: {"recall_hits": 0, "precision_sum": 0.0} for k in k_values}

    for q in scored_questions:
        relevant = set(q["relevant_document_ids"])

        candidates = retriever.search(q["question"], top_k=retrieve_k)
        reranked = reranker.rerank(q["question"], candidates, top_n=len(candidates))
        retrieved_doc_ids = [r["document_id"] for r in reranked]

        rank = next(
            (i + 1 for i, doc_id in enumerate(retrieved_doc_ids) if doc_id in relevant),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        for k in k_values:
            top_k_ids = retrieved_doc_ids[:k]
            if any(doc_id in relevant for doc_id in top_k_ids):
                results_per_k[k]["recall_hits"] += 1
            relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant)
            results_per_k[k]["precision_sum"] += relevant_in_top_k / k

    return {
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retriever-only vs Retriever+Reranker на eval_questions.json")
    parser.add_argument("--questions", default="../evaluation/eval_questions.json")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed_size", "fixed_size_overlap", "paragraph"])
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--reranker-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 10, 20])
    parser.add_argument("--retrieve-k", type=int, default=20)
    args = parser.parse_args()

    questions = em.load_questions(args.questions)
    retriever_ = rt.Retriever(args.faiss_dir, args.strategy, args.embedding_model)
    reranker_ = rr.Reranker(args.reranker_model)

    print("\n\n>>> BASELINE: только retriever")
    baseline_summary = em.evaluate(retriever_, questions, args.k_values)
    em.print_summary(baseline_summary, f"Retriever-only ({args.strategy} / {args.embedding_model})")

    print("\n\n>>> С RERANKER'ом")
    reranked_summary = evaluate_with_reranker(
        retriever_, reranker_, questions, args.k_values, args.retrieve_k
    )
    em.print_summary(reranked_summary, f"Retriever + Reranker (top-{args.retrieve_k} -> rerank)")

    print(f"\n\n{'=' * 60}\nИТОГ: изменение метрик после добавления reranker'а\n{'=' * 60}")
    print(f"MRR: {baseline_summary['mrr']:.4f} -> {reranked_summary['mrr']:.4f} "
          f"(delta {reranked_summary['mrr'] - baseline_summary['mrr']:+.4f})")
    for k in args.k_values:
        b = baseline_summary["per_k"][k]
        r = reranked_summary["per_k"][k]
        print(
            f"K={k:<3} Recall: {b['recall']:.4f} -> {r['recall']:.4f} "
            f"(delta {r['recall'] - b['recall']:+.4f})   "
            f"Precision: {b['precision']:.4f} -> {r['precision']:.4f} "
            f"(delta {r['precision'] - b['precision']:+.4f})"
        )