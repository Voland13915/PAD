"""
embed_store.py

Векторизация чанков и построение FAISS-индексов для RAG-системы
по книгам "Гарри Поттер".

Строит индекс для КАЖДОЙ комбинации (стратегия chunking'а x embedding-модель),
чтобы потом в experiments/ можно было сравнить, как каждая комбинация влияет
на качество retrieval (это и есть исследовательская часть задания).

Поддерживаются 2 embedding-модели для сравнения:
  - intfloat/multilingual-e5-base            (требует префиксы "query: "/"passage: ")
  - paraphrase-multilingual-mpnet-base-v2     (без префиксов)

Вход:  data/processed/chunks_<strategy>.json  (результат chunking.py)
Выход: data/processed/faiss/<strategy>__<model>.index   (сам FAISS-индекс)
       data/processed/faiss/<strategy>__<model>.meta.json (metadata чанков, тот же порядок, что и в индексе)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import faiss


# Модели, для которых E5-style префиксы обязательны (иначе просадка качества).
E5_STYLE_MODELS = {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large"}


def _short_name(model_name: str) -> str:
    """intfloat/multilingual-e5-base -> e5-base (для имени файла)."""
    return model_name.split("/")[-1].replace("multilingual-", "")


class EmbeddingModel:
    """
    Обёртка над sentence-transformers, учитывающая, что E5-модели
    требуют префиксы "query: " / "passage: " перед текстом.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.needs_prefix = model_name in E5_STYLE_MODELS
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if self.needs_prefix:
            texts = [f"passage: {t}" for t in texts]
        return self._model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )

    def encode_query(self, text: str) -> np.ndarray:
        if self.needs_prefix:
            text = f"query: {text}"
        return self._model.encode([text], normalize_embeddings=True)[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def build_index(chunks: list[dict], model: EmbeddingModel) -> tuple[faiss.Index, list[dict]]:
    """
    Строит FAISS-индекс (Inner Product поверх нормализованных векторов =
    косинусное сходство) для списка чанков.

    Возвращает (индекс, metadata) — metadata[i] соответствует вектору под
    номером i в индексе, это и есть связь "вектор -> исходный текст/глава".
    """
    texts = [c["text"] for c in chunks]
    vectors = model.encode_passages(texts).astype("float32")

    index = faiss.IndexFlatIP(model.dimension)
    index.add(vectors)

    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "document_id": c["document_id"],
            "book_number": c["book_number"],
            "book_title": c["book_title"],
            "chapter_number": c["chapter_number"],
            "chapter_title": c["chapter_title"],
            "strategy": c["strategy"],
            "text": c["text"],
        }
        for c in chunks
    ]
    return index, metadata


def save_index(index: faiss.Index, metadata: list[dict], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / f"{name}.index"))
    (out_dir / f"{name}.meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(out_dir: Path, name: str) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(str(out_dir / f"{name}.index"))
    metadata = json.loads((out_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
    return index, metadata


def search(
    query: str, model: EmbeddingModel, index: faiss.Index, metadata: list[dict], top_k: int = 5
) -> list[dict]:
    """Возвращает top_k чанков, отсортированных по убыванию similarity."""
    query_vector = model.encode_query(query).astype("float32").reshape(1, -1)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(metadata[idx])
        item["score"] = float(score)
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_embedding_pipeline(processed_dir: str, faiss_dir: str, strategies: list[str], models: list[str]) -> None:
    processed_path = Path(processed_dir)
    faiss_path = Path(faiss_dir)

    for model_name in models:
        print(f"\nЗагружаю модель: {model_name}")
        model = EmbeddingModel(model_name)
        model_short = _short_name(model_name)

        for strategy in strategies:
            chunks_file = processed_path / f"chunks_{strategy}.json"
            if not chunks_file.exists():
                print(f"  [!] Пропуск {strategy}: файл {chunks_file.name} не найден")
                continue

            chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            print(f"  Строю индекс: {strategy} x {model_short} ({len(chunks)} чанков)...")

            index, metadata = build_index(chunks, model)
            index_name = f"{strategy}__{model_short}"
            save_index(index, metadata, faiss_path, index_name)
            print(f"  -> сохранено: {index_name}.index / .meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение FAISS-индексов для всех комбинаций strategy x model")
    parser.add_argument("--processed-dir", default="../../data/processed")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed_size", "fixed_size_overlap", "paragraph"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["intfloat/multilingual-e5-base", "paraphrase-multilingual-mpnet-base-v2"],
    )
    args = parser.parse_args()

    run_embedding_pipeline(args.processed_dir, args.faiss_dir, args.strategies, args.models)