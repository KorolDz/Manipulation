from pathlib import Path

from app.core.constants import AUDIO, VIDEO
from app.core.models import MediaInfo
from app.services.file_security import (
    MAX_AUDIO_CHANNELS,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_AUDIO_SAMPLE_RATE,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FPS,
    MAX_VIDEO_HEIGHT,
    MAX_VIDEO_WIDTH,
)
from app.services.validation import validate_media_file


class MediaPreprocessor:
    def preprocess(self, media_type, file_path, validation_details=None):
        if validation_details is None:
            validation = validate_media_file(file_path, media_type)
            if not validation.is_valid:
                raise ValueError(validation.message)
            validation_details = validation.details

        path = Path(file_path)
        technical_info = dict(validation_details or {})
        technical_info["extension"] = path.suffix.lower()
        duration = None

        if media_type == AUDIO:
            duration, audio_info = self.inspect_audio(path)
            technical_info.update(audio_info)
            self.ensure_audio_is_within_limits(duration, technical_info)
        elif media_type == VIDEO:
            duration, video_info = self.inspect_video(path)
            technical_info.update(video_info)
            self.ensure_video_is_within_limits(duration, technical_info)

        return MediaInfo(
            file_path=str(path),
            file_name=path.name,
            media_type=media_type,
            file_size=path.stat().st_size,
            extension=path.suffix.lower(),
            duration=duration,
            technical_info=technical_info,
        )

    def inspect_audio(self, path):
        info = {}
        duration = None

        try:
            import soundfile as sf

            sound_info = sf.info(str(path))
            duration = float(sound_info.duration) if sound_info.duration else None
            info.update(
                {
                    "sample_rate": int(sound_info.samplerate),
                    "channels": int(sound_info.channels),
                    "format": sound_info.format,
                    "subtype": sound_info.subtype,
                }
            )
            return duration, info
        except Exception:
            pass

        try:
            import librosa

            try:
                duration = float(librosa.get_duration(path=str(path)))
            except Exception:
                duration = None

            audio, sample_rate = librosa.load(str(path), mono=False, duration=1)
            channels = 1 if audio.ndim == 1 else int(audio.shape[0])
            if duration is None:
                duration = float(librosa.get_duration(y=audio, sr=sample_rate))
            info.update(
                {
                    "sample_rate": int(sample_rate),
                    "channels": channels,
                    "format": path.suffix.lower().lstrip("."),
                }
            )
        except Exception as exc:
            info["preprocess_warning"] = f"Не удалось прочитать аудио-метаданные: {exc}"

        return duration, info

    def ensure_audio_is_within_limits(self, duration, info):
        warning = info.get("preprocess_warning")
        if warning and duration is None:
            raise ValueError(f"Файл невозможно безопасно прочитать: {warning}")

        if duration is not None and duration > MAX_AUDIO_DURATION_SECONDS:
            raise ValueError(f"Аудиофайл длиннее допустимого лимита: максимум {duration_limit_label(MAX_AUDIO_DURATION_SECONDS)}.")

        sample_rate = info.get("sample_rate")
        if sample_rate and int(sample_rate) > MAX_AUDIO_SAMPLE_RATE:
            raise ValueError(f"Частота дискретизации выше допустимого лимита: максимум {MAX_AUDIO_SAMPLE_RATE} Гц.")

        channels = info.get("channels")
        if channels and int(channels) > MAX_AUDIO_CHANNELS:
            raise ValueError(f"Количество аудиоканалов выше допустимого лимита: максимум {MAX_AUDIO_CHANNELS}.")

    def inspect_video(self, path):
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None, {"preprocess_warning": "Не удалось открыть видеофайл."}

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = frame_count / fps if frame_count and fps else None

            return duration, {
                "fps": round(fps, 3) if fps else None,
                "frame_count": frame_count or None,
                "width": width or None,
                "height": height or None,
                "resolution": f"{width}x{height}" if width and height else None,
            }
        finally:
            capture.release()

    def ensure_video_is_within_limits(self, duration, info):
        warning = info.get("preprocess_warning")
        if warning:
            raise ValueError(f"Файл невозможно безопасно прочитать: {warning}")

        if duration is not None and duration > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(f"Видеофайл длиннее допустимого лимита: максимум {duration_limit_label(MAX_VIDEO_DURATION_SECONDS)}.")

        width = info.get("width")
        height = info.get("height")
        if width and height and (int(width) > MAX_VIDEO_WIDTH or int(height) > MAX_VIDEO_HEIGHT):
            raise ValueError(f"Разрешение видео выше допустимого лимита: максимум {MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT}.")

        fps = info.get("fps")
        if fps and float(fps) > MAX_VIDEO_FPS:
            raise ValueError(f"FPS видео выше допустимого лимита: максимум {MAX_VIDEO_FPS}.")


def duration_limit_label(seconds):
    minutes, remaining_seconds = divmod(int(seconds), 60)
    if minutes and not remaining_seconds:
        return f"{minutes} мин"
    return f"{seconds} сек"
