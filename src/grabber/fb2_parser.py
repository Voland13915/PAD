"""
fb2_parser.py

Парсер fb2-файлов (формат FictionBook 2.0) для книг серии "Гарри Поттер".

Каждый fb2-файл в нашем случае содержит одну книгу:
  <description><title-info>...</title-info></description>  -> метаданные книги
  <body>
    <title>...</title>                                       -> общий заголовок книги (не глава)
    <section>                                                 -> одна глава
      <title><p>Глава N</p><p>Название главы</p></title>
      <p>...текст...</p>
      <p>...текст...</p>
    </section>
    <section>...</section>
    ...
  </body>

Результат парсинга одного файла — список "документов" (по одному на главу),
пригодных для дальнейшего preprocessing/chunking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from lxml import etree
import hashlib


FB2_NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}


@dataclass
class Chapter:
    """Один документ уровня 'глава' — единица, которую отдаём дальше в pipeline."""
    document_id: str
    book_title: str
    book_number: int
    author: str
    chapter_number: int
    chapter_title: str
    text: str
    source_file: str
    content_hash: str


def _tag_text(section_title_el) -> str:
    """Собирает текст из <title><p>...</p><p>...</p></title> в одну строку."""
    if section_title_el is None:
        return ""
    parts = [p.text.strip() for p in section_title_el.findall("fb:p", FB2_NS) if p.text]
    return " — ".join(parts)


def _section_text(section_el) -> str:
    """
    Собирает весь читаемый текст главы, пропуская вложенный <title>
    (он уже обработан отдельно) и вложенные <section> (в fb2-книгах ГП их нет,
    но на всякий случай не спускаемся рекурсивно).
    """
    paragraphs = []
    for child in section_el:
        tag = etree.QName(child).localname
        if tag == "title":
            continue
        if tag == "p" and child.text:
            paragraphs.append(child.text.strip())
        elif tag == "empty-line":
            continue
    return "\n".join(paragraphs)


def parse_fb2(file_path: str) -> list[Chapter]:
    """
    Парсит один fb2-файл и возвращает список глав (Chapter).

    Бросает ValueError, если не удалось найти обязательные элементы
    (author/book-title/body) — считаем такой файл "битым" и сообщаем
    об этом вызывающему коду (grabber), а не падаем молча.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"fb2 file not found: {file_path}")

    file_bytes = path.read_bytes()
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    try:
        root = etree.fromstring(file_bytes)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Не удалось распарсить XML в файле {file_path}: {e}")

    title_info = root.find(".//fb:description/fb:title-info", FB2_NS)
    if title_info is None:
        raise ValueError(f"В файле {file_path} отсутствует <title-info>")

    book_title_el = title_info.find("fb:book-title", FB2_NS)
    book_title = book_title_el.text.strip() if book_title_el is not None and book_title_el.text else "Неизвестно"

    author_el = title_info.find("fb:author", FB2_NS)
    author = "Неизвестен"
    if author_el is not None:
        first = author_el.findtext("fb:first-name", default="", namespaces=FB2_NS)
        last = author_el.findtext("fb:last-name", default="", namespaces=FB2_NS)
        author = f"{first} {last}".strip()

    seq_el = title_info.find("fb:sequence", FB2_NS)
    book_number = int(seq_el.get("number")) if seq_el is not None and seq_el.get("number") else 0

    body = root.find("fb:body", FB2_NS)
    if body is None:
        raise ValueError(f"В файле {file_path} отсутствует <body>")

    sections = body.findall("fb:section", FB2_NS)
    if not sections:
        raise ValueError(f"В файле {file_path} не найдено ни одной <section> (главы)")

    chapters: list[Chapter] = []
    for i, section in enumerate(sections, start=1):
        title_el = section.find("fb:title", FB2_NS)
        chapter_title = _tag_text(title_el) or f"Глава {i}"
        text = _section_text(section)

        chapters.append(
            Chapter(
                document_id=f"hp{book_number}_ch{i:02d}",
                book_title=book_title,
                book_number=book_number,
                author=author,
                chapter_number=i,
                chapter_title=chapter_title,
                text=text,
                source_file=path.name,
                content_hash=content_hash,
            )
        )

    return chapters


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Использование: python fb2_parser.py <путь_к_файлу.fb2>")
        sys.exit(1)

    chapters = parse_fb2(sys.argv[1])
    print(f"Книга: {chapters[0].book_title} (книга №{chapters[0].book_number})")
    print(f"Автор: {chapters[0].author}")
    print(f"Найдено глав: {len(chapters)}")
    print()
    print("Пример первой главы:")
    print(json.dumps(
        {
            "document_id": chapters[0].document_id,
            "chapter_title": chapters[0].chapter_title,
            "text_preview": chapters[0].text[:200] + "...",
        },
        ensure_ascii=False,
        indent=2,
    ))