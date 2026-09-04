"""
Grabber для сбора текстов молдавского законодательства с legis.md через PDF.

Почему PDF, а не парсинг HTML:
- legis.md — это SPA (текст документа подгружается через JS), простым requests
  его не взять без headless-браузера.
- Сайт официально даёт скачивать документы в PDF — это официальный, предсказуемый
  по структуре источник, который не требует эмуляции браузера.

Как определяются новые/изменённые документы:
- Так как сайт не отдаёт revision_id (в отличие от MediaWiki), используем
  хеш содержимого PDF-файла (SHA-256). Если хеш скачанного файла совпал с тем,
  что в manifest.json — документ не изменился, пропускаем повторную обработку.

ВАЖНО: PDF_URL_TEMPLATE ниже — плейсхолдер. Реальный URL скачивания PDF на legis.md
нужно подсмотреть через DevTools -> Network (см. инструкцию в чате) и подставить сюда.

Запуск:
    python -m src.grabber.pdf_grabber --config configs/sources.yaml
"""

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
import yaml
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("grabber")

REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY_SEC = 2

# TODO: заменить на реальный URL, найденный через DevTools -> Network.
# Скорее всего внутри будет doc_id и lang, по аналогии с getResults?doc_id=...&lang=ru
PDF_URL_TEMPLATE = "https://www.legis.md/PLACEHOLDER/downloadPdf?doc_id={doc_id}&lang=ru"


@dataclass
class DocumentMeta:
    source: str
    source_id: str
    title: str
    document_id: str
    content_hash: str
    updated_at: str
    url: str
    page_count: int


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def download_pdf(url: str, dest_path: Path) -> bool:
    """Скачивает PDF с retry. Возвращает True при успехе."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").find("pdf") == -1 and not resp.content.startswith(b"%PDF"):
                log.warning("Ответ не похож на PDF: %s", url)
                return False
            dest_path.write_bytes(resp.content)
            return True
        except requests.RequestException as e:
            log.warning("Попытка %s/%s скачать %s не удалась: %s", attempt, RETRY_ATTEMPTS, url, e)
            time.sleep(RETRY_DELAY_SEC)
    log.error("Не удалось скачать PDF после %s попыток: %s", RETRY_ATTEMPTS, url)
    return False


def extract_text(pdf_path: Path) -> tuple[str, int]:
    """Достаёт текст из PDF постранично, склеивая с явным разделителем страницы."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    full_text = "\n\n[PAGE_BREAK]\n\n".join(pages_text)
    return full_text, len(pages_text)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["output"]["raw_dir"])
    pdf_dir = raw_dir / "pdf"
    json_dir = raw_dir / "text"
    manifest_path = Path(cfg["output"]["manifest_path"])
    pdf_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    fetched, skipped, failed = 0, 0, 0

    for entry in cfg["legis_md"]["documents"]:
        doc_id, source_id, title = entry["doc_id"], entry["source_id"], entry["title"]
        url = PDF_URL_TEMPLATE.format(doc_id=doc_id)
        pdf_path = pdf_dir / f"{source_id}.pdf"

        if not download_pdf(url, pdf_path):
            failed += 1
            continue

        content_hash = file_hash(pdf_path)

        # Пропускаем повторную обработку, если содержимое не изменилось.
        if manifest.get(source_id, {}).get("content_hash") == content_hash:
            log.info("Без изменений, пропуск: %s", title)
            skipped += 1
            continue

        text, page_count = extract_text(pdf_path)

        meta = DocumentMeta(
            source="legis.md",
            source_id=source_id,
            title=title,
            document_id=source_id,
            content_hash=content_hash,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            url=url,
            page_count=page_count,
        )

        doc_json_path = json_dir / f"{source_id}.json"
        with open(doc_json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": asdict(meta), "text": text}, f, ensure_ascii=False, indent=2)

        manifest[source_id] = asdict(meta)
        fetched += 1
        log.info("Сохранено: %s (%s стр.)", title, page_count)

    save_manifest(manifest_path, manifest)
    log.info("Готово. Новых/обновлённых: %s, без изменений: %s, ошибок: %s", fetched, skipped, failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grabber PDF-документов законодательства с legis.md")
    parser.add_argument("--config", default="configs/sources.yaml")
    args = parser.parse_args()
    run(args.config)