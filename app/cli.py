import argparse
from pathlib import Path

from app.core.constants import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    SUPPORTED_EXTENSIONS,
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
)
from app.core.presentation import confidence_label, media_label, status_label, verdict_label
from app.services.analysis_service import AnalysisService
from app.services.history_repository import HistoryRepository
from app.services.validation import detect_media_type


EXIT_OK = 0
EXIT_ANALYSIS_ERRORS = 1
EXIT_USAGE_ERROR = 2


def run_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.recursive and not args.folder:
        parser.error("--recursive используется только вместе с --folder")

    if not args.analyze and not args.folder:
        parser.error("укажите --analyze FILE или --folder DIR")

    repository = HistoryRepository()
    analysis_service = AnalysisService()

    if args.analyze:
        return run_file_analysis(args.analyze, analysis_service, repository)

    return run_folder_analysis(args.folder, args.recursive, analysis_service, repository)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="CLI-режим Deepfake Detector",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--analyze", metavar="FILE", help="проанализировать один медиафайл")
    target.add_argument("--folder", metavar="DIR", help="проанализировать поддерживаемые файлы в папке")
    parser.add_argument("--recursive", action="store_true", help="сканировать вложенные папки вместе с --folder")
    return parser


def run_file_analysis(file_path, analysis_service, repository):
    path = Path(file_path)
    if not path.is_file():
        print(f"Ошибка: файл не найден: {path}")
        return EXIT_USAGE_ERROR

    media_type = detect_media_type(path)
    if media_type is None:
        print(f"Ошибка: неподдерживаемый формат: {path.name}")
        print(f"Поддерживаемые форматы: {supported_extensions_label()}")
        return EXIT_USAGE_ERROR

    result = analyze_and_store(path, media_type, analysis_service, repository)
    print_result(result)
    print_summary(build_summary([result], skipped=0))

    return EXIT_ANALYSIS_ERRORS if result.status == STATUS_ERROR else EXIT_OK


def run_folder_analysis(folder_path, recursive, analysis_service, repository):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Ошибка: папка не найдена: {folder}")
        return EXIT_USAGE_ERROR

    files, skipped = collect_media_files(folder, recursive)
    if not files:
        print(f"Ошибка: в папке нет поддерживаемых файлов: {folder}")
        print(f"Поддерживаемые форматы: {supported_extensions_label()}")
        return EXIT_USAGE_ERROR

    results = []
    print(f"Найдено файлов для анализа: {len(files)}")
    for index, (path, media_type) in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {path}")
        result = analyze_and_store(path, media_type, analysis_service, repository)
        print_result(result)
        results.append(result)

    summary = build_summary(results, skipped)
    print_summary(summary)

    return EXIT_ANALYSIS_ERRORS if summary["errors"] else EXIT_OK


def collect_media_files(folder, recursive):
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files = []
    skipped = 0

    for path in sorted(iterator):
        if not path.is_file():
            continue

        media_type = detect_media_type(path)
        if media_type is None:
            skipped += 1
            continue

        files.append((path, media_type))

    return files, skipped


def analyze_and_store(path, media_type, analysis_service, repository):
    result = analysis_service.analyze(media_type, str(path))
    repository.add(result)
    return result


def build_summary(results, skipped):
    return {
        "found": len(results),
        "completed": sum(1 for result in results if result.status == STATUS_COMPLETED),
        "errors": sum(1 for result in results if result.status == STATUS_ERROR),
        "original": sum(1 for result in results if result.verdict == VERDICT_ORIGINAL),
        "deepfake": sum(1 for result in results if result.verdict == VERDICT_DEEPFAKE),
        "skipped": skipped,
    }


def print_result(result):
    verdict = "Ошибка" if result.status == STATUS_ERROR else verdict_label(result.verdict)
    print(f"Файл: {result.file_name}")
    print(f"Тип: {media_label(result.media_type)}")
    print(f"Статус: {status_label(result.status)}")
    print(f"Результат: {verdict}")
    print(f"Вероятность: {confidence_label(result.confidence)}")

    if result.error_message:
        print(f"Ошибка: {result.error_message}")

    if result.findings:
        print("Замечания:")
        for finding in result.findings:
            print(f"  - {finding}")


def print_summary(summary):
    print("\nИтог:")
    print(f"  Найдено: {summary['found']}")
    print(f"  Успешно: {summary['completed']}")
    print(f"  Ошибок: {summary['errors']}")
    print(f"  Original: {summary['original']}")
    print(f"  Deepfake: {summary['deepfake']}")
    print(f"  Пропущено: {summary['skipped']}")


def supported_extensions_label():
    extensions = sorted({extension for values in SUPPORTED_EXTENSIONS.values() for extension in values})
    return ", ".join(extensions)
