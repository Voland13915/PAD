"""
grabber.py

Автоматический сбор данных для RAG-системы по книгам "Гарри Поттер".

Обязанности grabber'а (согласно заданию лабораторной):
  - получать данные из источника (папка с fb2-файлами);
  - определять новые/изменённые документы (по SHA-256 хешу файла);
  - сохранять полученные данные (главы -> JSON в data/processed/);
  - обрабатывать ошибки (битый fb2 не должен ронять весь grabber);
  - не создавать дубликаты (пропускать файлы, которые не изменились);
  - сохранять metadata документа;
  - запускаться повторно без ручной подготовки данных.

Использование:
    python grabber.py --raw-dir ../../data/raw --out-dir ../../data/processed
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import asdict

from fb2_parser import parse_fb2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("grabber")

MANIFEST_NAME = "_manifest.json"


def load_manifest(out_dir: Path) -> dict:
    """
    Манифест хранит: {имя_fb2_файла: hash_на_момент_последней_обработки}.
    Это и есть механизм 'определять новые/изменённые документы'.
    """
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(out_dir: Path, manifest: dict) -> None:
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_grabber(raw_dir: str, out_dir: str, force: bool = False) -> dict:
    """
    Обрабатывает все .fb2 файлы в raw_dir.

    Возвращает сводку: {"processed": [...], "skipped": [...], "failed": [...]}.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fb2_files = sorted(raw_path.glob("*.fb2"))
    if not fb2_files:
        logger.warning(f"В папке {raw_dir} не найдено ни одного .fb2 файла")
        return {"processed": [], "skipped": [], "failed": []}

    manifest = load_manifest(out_path)
    summary = {"processed": [], "skipped": [], "failed": []}

    for fb2_file in fb2_files:
        try:
            chapters = parse_fb2(str(fb2_file))
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Не удалось обработать {fb2_file.name}: {e}")
            summary["failed"].append(fb2_file.name)
            continue

        current_hash = chapters[0].content_hash
        previous_hash = manifest.get(fb2_file.name)

        if previous_hash == current_hash and not force:
            logger.info(f"Пропуск (не изменился): {fb2_file.name}")
            summary["skipped"].append(fb2_file.name)
            continue

        book_number = chapters[0].book_number
        out_file = out_path / f"book_{book_number:02d}.json"
        out_file.write_text(
            json.dumps([asdict(c) for c in chapters], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest[fb2_file.name] = current_hash
        logger.info(
            f"Обработано: {fb2_file.name} -> {out_file.name} "
            f"({len(chapters)} глав, книга №{book_number})"
        )
        summary["processed"].append(fb2_file.name)

    save_manifest(out_path, manifest)

    logger.info(
        f"Готово. Обработано: {len(summary['processed'])}, "
        f"пропущено: {len(summary['skipped'])}, "
        f"ошибок: {len(summary['failed'])}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grabber для fb2-книг Гарри Поттер")
    parser.add_argument("--raw-dir", default="../../data/raw", help="Папка с исходными .fb2 файлами")
    parser.add_argument("--out-dir", default="../../data/processed", help="Куда сохранять обработанные JSON")
    parser.add_argument("--force", action="store_true", help="Пересобрать все файлы, игнорируя манифест")
    args = parser.parse_args()

    run_grabber(args.raw_dir, args.out_dir, force=args.force)