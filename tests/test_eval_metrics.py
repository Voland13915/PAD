"""
test_eval_metrics.py

Проверяет формулы Recall@K, Precision@K, MRR на контролируемом наборе данных
с заранее известным правильным ответом (см. ручной расчёт в комментариях).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))
import eval_metrics as em  # noqa: E402


class FakeRetriever:
    """
    Возвращает предопределённые результаты для двух вопросов:
      - 'q1': релевантный документ на позиции 2 (индекс 1)
      - 'q2': релевантный документ отсутствует среди результатов
    """

    def search(self, question, top_k):
        if question == "q1":
            docs = ["other1", "target1", "other2", "other3", "other4"]
        else:
            docs = ["x1", "x2", "x3", "x4", "x5"]
        return [{"document_id": d} for d in docs[:top_k]]


def test_mrr_calculation():
    questions = [
        {"id": "q1", "type": "simple_fact", "question": "q1", "relevant_document_ids": ["target1"]},
        {"id": "q2", "type": "simple_fact", "question": "q2", "relevant_document_ids": ["target2"]},
    ]
    summary = em.evaluate(FakeRetriever(), questions, k_values=[5])
    # q1: релевантный на позиции 2 -> RR=1/2; q2: не найден -> RR=0. MRR = (0.5+0)/2 = 0.25
    assert summary["mrr"] == 0.25


def test_recall_at_k():
    questions = [
        {"id": "q1", "type": "simple_fact", "question": "q1", "relevant_document_ids": ["target1"]},
        {"id": "q2", "type": "simple_fact", "question": "q2", "relevant_document_ids": ["target2"]},
    ]
    summary = em.evaluate(FakeRetriever(), questions, k_values=[1, 3, 5])
    # K=1: target1 на позиции 2, не входит в top-1 -> recall q1=miss, q2=miss -> 0.0
    assert summary["per_k"][1]["recall"] == 0.0
    # K=3: target1 на позиции 2 <= 3 -> hit; q2 miss -> 0.5
    assert summary["per_k"][3]["recall"] == 0.5
    # K=5: тот же результат, что и K=3 (target1 уже найден на позиции 2)
    assert summary["per_k"][5]["recall"] == 0.5


def test_precision_at_k():
    questions = [{"id": "q1", "type": "simple_fact", "question": "q1", "relevant_document_ids": ["target1"]}]
    summary = em.evaluate(FakeRetriever(), questions, k_values=[5])
    # top-5 для q1 содержит ровно 1 релевантный документ из 5 -> precision = 1/5 = 0.2
    assert summary["per_k"][5]["precision"] == 0.2


def test_questions_without_ground_truth_are_skipped():
    questions = [
        {"id": "q1", "type": "simple_fact", "question": "q1", "relevant_document_ids": ["target1"]},
        {"id": "q_no_gt", "type": "simple_fact", "question": "q?", "relevant_document_ids": None},
        {"id": "q_no_answer", "type": "no_answer", "question": "q??", "relevant_document_ids": []},
    ]
    summary = em.evaluate(FakeRetriever(), questions, k_values=[5])
    assert summary["n_questions_evaluated"] == 1


def test_evaluate_returns_empty_dict_when_no_scorable_questions():
    questions = [{"id": "q1", "type": "no_answer", "question": "q1", "relevant_document_ids": []}]
    summary = em.evaluate(FakeRetriever(), questions, k_values=[5])
    assert summary == {}