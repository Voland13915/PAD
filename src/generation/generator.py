"""
generator.py

Генерация ответа через локальную LLM (Ollama), с защитой от галлюцинаций
и указанием источников (п.12-13 задания).

Требует установленный и запущенный Ollama (https://ollama.com) с
загруженной моделью, например:
    ollama pull qwen2.5

Ollama по умолчанию поднимает локальный HTTP API на localhost:11434.
"""

import json
import urllib.request
import urllib.error


SYSTEM_PROMPT = """Ты — ассистент, отвечающий на вопросы по книгам о Гарри Поттере.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе текста, предоставленного в разделе "Контекст" ниже. Не используй никакие другие знания о Гарри Поттере, даже если ты их знаешь.
2. Если в контексте недостаточно информации для ответа на вопрос — прямо скажи об этом: "В предоставленном контексте недостаточно информации для ответа на этот вопрос." Не придумывай и не додумывай ответ.
3. Обязательно указывай источники в конце ответа — книгу и главу, откуда взята информация, в формате: "Источники: книга N, глава M «Название»".
4. Отвечай на русском языке, кратко и по существу.
5. Если контекст противоречив (разные фрагменты говорят разное) — укажи это в ответе, а не выбирай произвольно одну версию.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Форматирует отфильтрованные chunks в текстовый блок для промпта."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Фрагмент {i} — книга {c['book_number']}, {c['chapter_title']}]\n{c['text']}"
        )
    return "\n\n".join(parts)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return f"""Контекст:
{context}

Вопрос: {question}"""


def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model: str = "qwen2.5",
    host: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """
    Вызывает локальный Ollama API (/api/chat). Бросает RuntimeError с понятным
    сообщением, если Ollama не запущен или модель не найдена — чтобы
    generate_answer() мог явно сообщить об ошибке, а не упасть с трудночитаемым traceback.
    """
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Не удалось подключиться к Ollama по адресу {host}. "
            f"Убедись, что Ollama запущен (команда 'ollama serve' или запущен как служба). "
            f"Исходная ошибка: {e}"
        )
    except KeyError:
        raise RuntimeError(f"Неожиданный формат ответа от Ollama: {result}")


def generate_answer(question: str, chunks: list[dict], model: str = "qwen2.5") -> dict:
    """
    Полный цикл генерации: строит промпт из отфильтрованных chunks,
    вызывает LLM, возвращает ответ вместе с использованными источниками
    (отдельно от текста ответа — удобно для UI, чтобы не парсить ответ модели).
    """
    if not chunks:
        return {
            "answer": "В базе знаний не найдено релевантной информации для ответа на этот вопрос.",
            "sources": [],
            "model": model,
        }

    user_prompt = build_user_prompt(question, chunks)
    answer_text = call_ollama(SYSTEM_PROMPT, user_prompt, model=model)

    sources = [
        {
            "book_number": c["book_number"],
            "book_title": c["book_title"],
            "chapter_title": c["chapter_title"],
            "document_id": c["document_id"],
        }
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources, "model": model}