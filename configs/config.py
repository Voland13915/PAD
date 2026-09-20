"""
config.py

Центральные настройки pipeline. Значения здесь — это то же самое, что
значения по умолчанию у CLI-флагов в src/*/*.py, но собранные в одном месте
для удобства и для документирования выбора параметров (со ссылкой на
эксперимент, который его обосновывает — см. README.md §10).
"""

# --- Chunking ---
FIXED_CHUNK_SIZE = 500          # токенов; лимит контекста intfloat/multilingual-e5-base
FIXED_CHUNK_OVERLAP = 100       # токенов
PARAGRAPH_TARGET_SIZE = 400     # токенов, целевой размер при объединении абзацев

# --- Embeddings ---
# Выбрана по результатам сравнения в experiments/retrieval_comparison.csv:
# даёт MRR=0.592 против MRR=0.184-0.214 у paraphrase-multilingual-mpnet-base-v2
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODELS_COMPARED = [
    "intfloat/multilingual-e5-base",
    "paraphrase-multilingual-mpnet-base-v2",
]

# --- Chunking strategy ---
# 'paragraph' показала лучший MRR (0.592) среди трёх стратегий на e5-base
DEFAULT_CHUNKING_STRATEGY = "paragraph"
CHUNKING_STRATEGIES = ["fixed_size", "fixed_size_overlap", "paragraph"]

# --- Retriever ---
DEFAULT_RETRIEVE_K = 20         # сколько кандидатов достаёт retriever до filtering/reranking

# --- Reranker ---
# ОТКЛЮЧЁН по умолчанию: эксперименты (experiments/, README §10.4) показали,
# что обе протестированные reranker-модели ухудшают метрики на этом корпусе
# (вероятно, domain mismatch — модели обучены на веб-поиске, не на литературе).
USE_RERANKER_DEFAULT = False
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# --- Filtering ---
MIN_TEXT_LENGTH = 50            # символов; отсекает мусорные обрывки chunks
MAX_RESULTS_FOR_LLM = 5         # финальное количество chunks, передаваемых в LLM

# --- Generation ---
DEFAULT_LLM_MODEL = "qwen2.5"
LLM_MODELS_COMPARED = ["qwen2.5", "llama3"]
# Низкая temperature: эксперименты показали, что temperature ~0.7 (Ollama default)
# приводила к утечкам в другой язык и выдуманным деталям даже при строгом промпте.
DEFAULT_TEMPERATURE = 0.1
OLLAMA_HOST = "http://localhost:11434"

# --- Evaluation ---
EVAL_K_VALUES = [3, 5, 10, 20]