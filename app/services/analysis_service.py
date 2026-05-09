from app.core.constants import (
    AUDIO,
    STATUS_COMPLETED,
    STATUS_ERROR,
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
    VERDICT_UNKNOWN,
)
from app.core.models import AnalysisResult
from app.core.result_parser import parse_model_result
from app.services.media_preprocessor import MediaPreprocessor
from app.services.validation import file_size, validate_media_file


BLOCKING_RESULT_MARKERS = (
    "ЛИЦА НЕ НАЙДЕНЫ",
    "ОШИБКА",
    "ERROR",
)


class AnalysisService:
    def __init__(self):
        self.preprocessor = MediaPreprocessor()

    def analyze(self, media_type, file_path):
        validation = validate_media_file(file_path, media_type)
        if not validation.is_valid:
            return self.error_result(media_type, file_path, validation.message, technical_info=validation.details)

        try:
            media_info = self.preprocessor.preprocess(media_type, file_path, validation.details)
        except Exception as exc:
            return self.error_result(media_type, file_path, f"Ошибка предобработки: {exc}", technical_info=validation.details)

        try:
            if media_type == AUDIO:
                from audio_detection.audio_analyzer import analyze_audio

                raw_result = analyze_audio(file_path)
            else:
                from video_detection.video_analyzer import analyze_video

                raw_result = analyze_video(file_path)
        except Exception as exc:
            return self.error_result(media_type, file_path, f"Ошибка анализа: {exc}", media_info=media_info)

        if self.is_blocking_result(raw_result):
            return self.error_result(media_type, file_path, raw_result, raw_result=raw_result, media_info=media_info)

        parsed = parse_model_result(raw_result)
        if parsed.is_error:
            return self.error_result(media_type, file_path, raw_result, raw_result=raw_result, media_info=media_info)

        findings = self.build_findings(parsed, media_info)

        return AnalysisResult(
            file_path=media_info.file_path,
            file_name=media_info.file_name,
            media_type=media_type,
            file_size=media_info.file_size,
            status=STATUS_COMPLETED,
            verdict=parsed.verdict,
            confidence=parsed.confidence,
            raw_result=raw_result,
            duration=media_info.duration,
            technical_info=media_info.technical_info,
            findings=findings,
        )

    def error_result(self, media_type, file_path, message, raw_result="", media_info=None, technical_info=None):
        file_name = media_info.file_name if media_info else self.file_name(file_path)
        file_path_value = media_info.file_path if media_info else str(file_path or "")
        file_size_value = media_info.file_size if media_info else (file_size(file_path) if file_path else None)
        duration = media_info.duration if media_info else None
        technical_info = media_info.technical_info if media_info else (technical_info or {})

        return AnalysisResult(
            file_path=file_path_value,
            file_name=file_name,
            media_type=media_type,
            file_size=file_size_value,
            status=STATUS_ERROR,
            verdict=VERDICT_UNKNOWN,
            confidence=None,
            raw_result=raw_result or message,
            error_message=message,
            duration=duration,
            technical_info=technical_info,
            findings=self.build_error_findings(message, technical_info),
        )

    @staticmethod
    def is_blocking_result(raw_result):
        text = (raw_result or "").upper()
        return any(marker in text for marker in BLOCKING_RESULT_MARKERS)

    @staticmethod
    def file_name(file_path):
        if not file_path:
            return "Без файла"
        return str(file_path).replace("\\", "/").split("/")[-1] or "Без файла"

    @staticmethod
    def build_findings(parsed, media_info):
        findings = []

        warning = media_info.technical_info.get("preprocess_warning")
        if warning:
            findings.append(str(warning))

        security_warning = media_info.technical_info.get("security_warning")
        if security_warning:
            findings.append(str(security_warning))

        if parsed.verdict == VERDICT_DEEPFAKE:
            if parsed.confidence is None:
                findings.append("Модель обнаружила признаки цифровой манипуляции.")
            else:
                findings.append(f"Высокая вероятность фальсификации: {parsed.confidence:.1f}%.")
        elif parsed.verdict == VERDICT_ORIGINAL:
            if parsed.confidence is None:
                findings.append("Критичные признаки подделки не обнаружены.")
            else:
                findings.append(f"Критичные признаки подделки не обнаружены, оценка оригинальности: {parsed.confidence:.1f}%.")
        else:
            findings.append("Модель не вернула однозначную классификацию.")

        if media_info.duration is None:
            findings.append("Длительность файла не удалось определить на этапе предобработки.")

        return findings

    @staticmethod
    def build_error_findings(message, technical_info):
        findings = []
        warning = technical_info.get("preprocess_warning") if technical_info else None
        security_warning = technical_info.get("security_warning") if technical_info else None

        if warning:
            findings.append(str(warning))
        if security_warning:
            findings.append(str(security_warning))

        text = message or "Анализ не выполнен."
        if "Лица не найдены" in text:
            findings.append("В видео не обнаружены лица, поэтому классификация по лицевым признакам невозможна.")
        elif "Ошибка модели" in text:
            findings.append("Модель не загрузилась или не смогла выполнить классификацию.")
        elif "Ошибка анализа аудио" in text:
            findings.append("Аудиофайл не удалось обработать текущей аудио-моделью.")
        elif "Неверный формат" in text:
            findings.append("Файл не соответствует выбранному типу анализа.")
        else:
            findings.append(text)

        return findings
