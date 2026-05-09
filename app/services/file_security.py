from dataclasses import dataclass, field
from pathlib import Path
import hashlib


MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024
MAX_FILE_SIZE_MB = MAX_FILE_SIZE_BYTES // (1024 * 1024)

MAX_AUDIO_DURATION_SECONDS = 10 * 60
MAX_VIDEO_DURATION_SECONDS = 5 * 60
MAX_AUDIO_SAMPLE_RATE = 192_000
MAX_AUDIO_CHANNELS = 8
MAX_VIDEO_WIDTH = 3840
MAX_VIDEO_HEIGHT = 2160
MAX_VIDEO_FPS = 120

HEADER_BYTES = 64
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class FileSecurityResult:
    is_valid: bool
    message: str = ""
    details: dict[str, object] = field(default_factory=dict)


def inspect_media_file_security(path: Path, media_type: str) -> FileSecurityResult:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return security_error(f"Файл невозможно безопасно прочитать: {exc}")

    if size == 0:
        return security_error("Файл пустой.")

    if size > MAX_FILE_SIZE_BYTES:
        return security_error(f"Файл слишком большой: максимум {MAX_FILE_SIZE_MB} МБ.")

    try:
        header = read_header(path)
    except OSError as exc:
        return security_error(f"Файл невозможно безопасно прочитать: {exc}")

    if not has_expected_signature(path.suffix.lower(), media_type, header):
        return security_error("Сигнатура файла не соответствует выбранному формату.")

    try:
        sha256 = calculate_sha256(path)
    except OSError as exc:
        return security_error(f"Файл невозможно безопасно прочитать: {exc}")

    return FileSecurityResult(
        True,
        details={
            "sha256": sha256,
            "security_status": "Пройдено",
            "security_checks": [
                f"Размер файла не превышает {MAX_FILE_SIZE_MB} МБ",
                "Сигнатура файла соответствует расширению",
                "SHA-256 рассчитан потоковым чтением",
            ],
        },
    )


def security_error(message: str) -> FileSecurityResult:
    return FileSecurityResult(
        False,
        message=message,
        details={
            "security_status": "Отклонено",
            "security_warning": message,
            "security_checks": [message],
        },
    )


def read_header(path: Path) -> bytes:
    with path.open("rb") as file:
        return file.read(HEADER_BYTES)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def has_expected_signature(extension: str, media_type: str, header: bytes) -> bool:
    if media_type == "audio":
        if extension == ".wav":
            return is_wav(header)
        if extension == ".mp3":
            return is_mp3(header)

    if media_type == "video":
        if extension == ".avi":
            return is_avi(header)
        if extension in {".mp4", ".mov"}:
            return is_iso_base_media(header)

    return False


def is_wav(header: bytes) -> bool:
    return len(header) >= 12 and header[:4] in {b"RIFF", b"RIFX"} and header[8:12] == b"WAVE"


def is_avi(header: bytes) -> bool:
    return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "


def is_mp3(header: bytes) -> bool:
    if header.startswith(b"ID3"):
        return True
    return len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0


def is_iso_base_media(header: bytes) -> bool:
    return len(header) >= 12 and header[4:8] == b"ftyp"
