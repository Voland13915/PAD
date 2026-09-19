"""
app.py

Простой веб-интерфейс для RAG-системы по книгам "Гарри Поттер" на Streamlit.

Запуск:
    streamlit run app.py
(из папки src/)
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "embeddings"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reranking"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "filtering"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "generation"))
import filtering as ft  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402


st.set_page_config(page_title="RAG: Гарри Поттер", page_icon="⚡", layout="centered")


@st.cache_resource(show_spinner="Загружаю модели и индекс (это может занять минуту при первом запуске)...")
def load_pipeline(
    strategy: str, embedding_model: str, llm_model: str, use_reranker: bool, temperature: float
) -> RAGPipeline:
    """
    st.cache_resource гарантирует, что модели (embedding + LLM-клиент) и FAISS-индекс
    грузятся ОДИН раз за сессию Streamlit, а не при каждом вопросе пользователя —
    иначе интерфейс был бы невыносимо медленным.
    """
    filter_config = ft.FilterConfig(
        score_field="reranker_score" if use_reranker else "score",
        min_score=None,
        min_text_length=50,
        max_results=5,
    )
    return RAGPipeline(
        faiss_dir="../data/processed/faiss",
        strategy=strategy,
        embedding_model=embedding_model,
        llm_model=llm_model,
        use_reranker=use_reranker,
        filter_config=filter_config,
        temperature=temperature,
    )


def main() -> None:
    st.title("⚡ RAG-система: Гарри Поттер")
    st.caption("Задай вопрос по семи книгам — ответ строится только на основе найденных фрагментов текста.")

    with st.sidebar:
        st.header("Настройки")
        strategy = st.selectbox(
            "Стратегия chunking'а",
            ["paragraph", "fixed_size", "fixed_size_overlap"],
            index=0,
            help="'paragraph' показала лучшие метрики в экспериментах (см. experiments/retrieval_comparison.csv)",
        )
        embedding_model = st.selectbox(
            "Embedding-модель",
            ["intfloat/multilingual-e5-base", "paraphrase-multilingual-mpnet-base-v2"],
            index=0,
        )
        llm_model = st.selectbox("LLM (Ollama)", ["qwen2.5", "llama3"], index=0)
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Ниже = более предсказуемо и менее склонно к выдумкам/утечкам в другой язык (см. эксперименты в отчёте)",
        )
        use_reranker = st.checkbox(
            "Использовать reranker",
            value=False,
            help="В наших экспериментах reranker ухудшал метрики на этом корпусе (см. experiments/) — по умолчанию отключён",
        )

        if st.button("Очистить историю чата"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Загружаем (или берём из кэша) pipeline с текущими настройками
    try:
        pipeline = load_pipeline(strategy, embedding_model, llm_model, use_reranker, temperature)
    except Exception as e:
        st.error(f"Не удалось загрузить модели/индекс: {e}")
        st.stop()

    # История чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Источники"):
                    for s in message["sources"]:
                        st.markdown(f"- книга {s['book_number']} «{s['book_title']}», {s['chapter_title']}")

    question = st.chat_input("Задай вопрос по книгам о Гарри Поттере...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ..."):
                try:
                    result = pipeline.answer(question)
                    st.markdown(result["answer"])
                    if result["sources"]:
                        with st.expander("Источники"):
                            for s in result["sources"]:
                                st.markdown(f"- книга {s['book_number']} «{s['book_title']}», {s['chapter_title']}")
                    st.caption(
                        f"Использовано {result['n_chunks_used']} из {result['n_candidates_retrieved']} "
                        f"найденных фрагментов · модель: {result['model']}"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                except RuntimeError as e:
                    st.error(f"Ошибка при генерации ответа: {e}")


if __name__ == "__main__":
    main()