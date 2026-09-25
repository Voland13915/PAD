"""
rag_pipeline.py

Полный RAG pipeline: Query -> Retriever -> (опционально Reranker) -> Filtering -> LLM -> Ответ с источниками.

По умолчанию reranker ОТКЛЮЧЁН — наши эксперименты (eval_reranker.py) показали,
что обе протестированные reranker-модели ухудшают метрики на этом корпусе
(вероятно, domain mismatch: reranker'ы обучены на веб-поиске, а не на
художественной прозе). Можно включить через --use-reranker для сравнения.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reranking"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filtering"))
import retriever as rt  # noqa: E402
import reranker as rr  # noqa: E402
import filtering as ft  # noqa: E402
import generator as gen  # noqa: E402
import query_logger as ql  # noqa: E402


class RAGPipeline:
    def __init__(
        self,
        faiss_dir: str,
        strategy: str,
        embedding_model: str,
        llm_model: str,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        retrieve_k: int = 20,
        filter_config: ft.FilterConfig | None = None,
        temperature: float = 0.1,
    ):
        self.retriever = rt.Retriever(faiss_dir, strategy, embedding_model)
        self.use_reranker = use_reranker
        self.reranker = rr.Reranker(reranker_model) if use_reranker else None
        self.retrieve_k = retrieve_k
        self.llm_model = llm_model
        self.temperature = temperature

        # score_field зависит от того, включён ли reranker: если да — фильтруем
        # по reranker_score (он есть только после rerank), иначе по обычному score.
        self.filter_config = filter_config or ft.FilterConfig(
            score_field="reranker_score" if use_reranker else "score",
            min_score=None,
            min_text_length=50,
            max_results=5,
        )

    def answer(self, question: str) -> dict:
        start_time = time.time()
        error_message = None
        result = {}
        try:
            candidates = self.retriever.search(question, top_k=self.retrieve_k)

            if self.use_reranker:
                candidates = self.reranker.rerank(question, candidates, top_n=len(candidates))

            filtered = ft.apply_filters(candidates, self.filter_config)

            result = gen.generate_answer(question, filtered, model=self.llm_model, temperature=self.temperature)
            result["n_candidates_retrieved"] = len(candidates)
            result["n_chunks_used"] = len(filtered)
            return result
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            ql.log_query(
                question=question,
                strategy=self.retriever.strategy,
                embedding_model=self.retriever.model_name,
                llm_model=self.llm_model,
                temperature=self.temperature,
                use_reranker=self.use_reranker,
                n_candidates_retrieved=result.get("n_candidates_retrieved", 0),
                n_chunks_used=result.get("n_chunks_used", 0),
                answer=result.get("answer", ""),
                sources=result.get("sources", []),
                duration_seconds=time.time() - start_time,
                error=error_message,
            )


def print_answer(result: dict) -> None:
    print(f"\n{'=' * 70}")
    print(result["answer"])
    print(f"{'=' * 70}")
    print(f"(использовано {result['n_chunks_used']} из {result['n_candidates_retrieved']} найденных фрагментов)")
    if result["sources"]:
        print("\nИсточники:")
        for s in result["sources"]:
            print(f"  - книга {s['book_number']} «{s['book_title']}», {s['chapter_title']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полный RAG pipeline для Гарри Поттера")
    parser.add_argument("--faiss-dir", default="../../data/processed/faiss")
    parser.add_argument("--strategy", default="paragraph", choices=["fixed_size", "fixed_size_overlap", "paragraph"])
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--llm-model", default="qwen2.5")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    filter_config = ft.FilterConfig(
        score_field="reranker_score" if args.use_reranker else "score",
        min_score=None,
        min_text_length=50,
        max_results=args.max_results,
    )

    pipeline = RAGPipeline(
        faiss_dir=args.faiss_dir,
        strategy=args.strategy,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        use_reranker=args.use_reranker,
        filter_config=filter_config,
        temperature=args.temperature,
    )

    print("RAG-система готова. Введи вопрос (или 'exit' для выхода):\n")
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit", "выход"):
            break
        if not question:
            continue
        try:
            result = pipeline.answer(question)
            print_answer(result)
        except RuntimeError as e:
            print(f"\n[Ошибка] {e}\n")