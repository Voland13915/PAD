"""
test_fb2_parser.py

Проверяет fb2_parser.py на синтетическом fb2-файле (та же структура,
что и в реальных книгах: title-info с автором/названием/sequence,
body с несколькими section-главами).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "grabber"))
from fb2_parser import parse_fb2  # noqa: E402


SAMPLE_FB2 = """<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description>
  <title-info>
    <author><first-name>Джоан</first-name><last-name>Роулинг</last-name></author>
    <book-title>Тестовая книга</book-title>
    <sequence name="Гарри Поттер" number="1"/>
  </title-info>
</description>
<body>
<title><p>Тестовая книга</p></title>
<section>
<title><p>Глава 1</p><p>Начало</p></title>
<p>Текст первой главы.</p>
<p>Второй абзац первой главы.</p>
</section>
<section>
<title><p>Глава 2</p><p>Продолжение</p></title>
<p>Текст второй главы.</p>
</section>
</body>
</FictionBook>"""


@pytest.fixture
def sample_fb2_file(tmp_path):
    file_path = tmp_path / "test_book.fb2"
    file_path.write_text(SAMPLE_FB2, encoding="utf-8")
    return str(file_path)


def test_parse_fb2_extracts_correct_number_of_chapters(sample_fb2_file):
    chapters = parse_fb2(sample_fb2_file)
    assert len(chapters) == 2


def test_parse_fb2_extracts_book_metadata(sample_fb2_file):
    chapters = parse_fb2(sample_fb2_file)
    assert chapters[0].book_title == "Тестовая книга"
    assert chapters[0].author == "Джоан Роулинг"
    assert chapters[0].book_number == 1


def test_parse_fb2_extracts_chapter_content(sample_fb2_file):
    chapters = parse_fb2(sample_fb2_file)
    assert chapters[0].chapter_number == 1
    assert "Текст первой главы" in chapters[0].text
    assert "Второй абзац" in chapters[0].text
    assert chapters[1].chapter_number == 2
    assert "Текст второй главы" in chapters[1].text


def test_parse_fb2_document_id_format(sample_fb2_file):
    chapters = parse_fb2(sample_fb2_file)
    assert chapters[0].document_id == "hp1_ch01"
    assert chapters[1].document_id == "hp1_ch02"


def test_parse_fb2_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_fb2("/nonexistent/path.fb2")


def test_parse_fb2_content_hash_consistent(sample_fb2_file):
    chapters = parse_fb2(sample_fb2_file)
    # Хеш должен быть одинаковым для всех глав одного файла (это хеш файла целиком)
    assert chapters[0].content_hash == chapters[1].content_hash