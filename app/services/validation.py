from pathlib import Path

from app.core.constants import AUDIO, SUPPORTED_EXTENSIONS, VIDEO
from app.core.models import ValidationResult


def detect_media_type(file_path):
    extension = Path(file_path).suffix.lower()

    for media_type, extensions in SUPPORTED_EXTENSIONS.items():
        if extension in extensions:
            return media_type

    return None


def validate_media_file(file_path, expected_type):
    if expected_type not in SUPPORTED_EXTENSIONS:
        return ValidationResult(False, message="Неизвестный режим анализа.")

    if not file_path:
        return ValidationResult(False, message="Файл не выбран.")

    path = Path(file_path)
    if not path.is_file():
        return ValidationResult(False, message="Файл не найден.")

    media_type = detect_media_type(path)
    if media_type != expected_type:
        expected_label = "аудио" if expected_type == AUDIO else "видео"
        return ValidationResult(False, media_type=media_type, message=f"Неверный формат для {expected_label}.")

    return ValidationResult(True, media_type=media_type)


def file_size(file_path):
    path = Path(file_path)
    if not path.is_file():
        return None
    return path.stat().st_size
