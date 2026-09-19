"""
fill_ground_truth.py

Помощник для разметки eval_questions.json.

Для каждого вопроса с relevant_document_ids == null (пропускает вопросы
типа 'no_answer' — там [] осталось намеренно) прогоняет вопрос через
retriever, показывает top-10 кандидатов, и просит выбрать номера
правильных ответов. Результат сразу сохраняется в файл после каждого
вопроса (можно прерваться в любой момент и продолжить позже).

Ввод:
  - номера через запятую (например "1,3") — если ответ раскрыт в нескольких
    местах (актуально для multi_document / context_understanding вопросов);
  - пусто (Enter) — пропустить вопрос, оставить null, вернуться к нему позже;
  - "none" — явно сказать "ни один из показанных вариантов не подходит"
    (тогда стоит увеличить top_k или переформулировать вопрос).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
import retriever as rt  # noqa: E402


def fill_ground_truth(questions_path: str, retriever: rt.Retriever, top_k: int) -> None:
    path = Path(questions_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]

    todo = [q for q in questions if q["relevant_document_ids"] is None]
    print(f"Вопросов для разметки: {len(todo)}\n")

    for q in todo:
        print(f"\n{'=' * 70}")
        print(f"[{q['id']}] ({q['type']}) {q['question']}")
        print(f"Ожидаемый ответ (подсказка): {q['expected_answer']}")
        print(f"{'=' * 70}")

        results = retriever.search(q["question"], top_k=top_k)
        for i, r in enumerate(results, start=1):
            preview = r["text"][:100].replace("\n", " ")
            print(f"  {i}. книга {r['book_number']}, {r['chapter_title']}  [{r['document_id']}]")
            print(f"     {preview}...")

        choice = input(
            "\nНомера правильных (через запятую), Enter=пропустить, 'none'=не найдено: "
        ).strip()

        if not choice:
            print("Пропущено.")
            continue

        if choice.lower() == "none":
            print("Отмечено как 'нет подходящего среди показанных' — оставляю null.")
            continue

        try:
            indices = [int(x.strip()) for x in choice.split(",")]
            selected_ids = [results[i - 1]["document_id"] for i in indices]
        except (ValueError, IndexError):
            print("Не понял ввод, пропускаю этот вопрос.")
            continue

        q["relevant_document_ids"] = selected_ids
        print(f"Сохранено: {selected_ids}")

        # Сохраняем после каждого вопроса — не потеряем прогресс при прерывании.
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    remaining = [q for q in questions if q["relevant_document_ids"] is None]
    print(f"\nГотово. Осталось незаполненных: {len(remaining)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разметка ground truth для eval_questions.json")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed_size", "fixed_size_overlap", "paragraph"])
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    retriever_ = rt.Retriever(args.faiss_dir, args.strategy, args.model)
    fill_ground_truth(args.questions, retriever_, args.top_k)