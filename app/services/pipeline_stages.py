import base64
import html
from pathlib import Path

from app.core.constants import (
    AUDIO,
    PROJECT_ROOT,
    STATUS_ERROR,
    VIDEO,
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
)
from app.core.models import ReportView
from app.core.presentation import confidence_label, file_size_label, media_label, verdict_label
from app.core.result_parser import parse_model_result
from app.services.media_preprocessor import MediaPreprocessor
from app.services.validation import file_size, validate_media_file


BLOCKING_RESULT_MARKERS = (
    "ЛИЦА НЕ НАЙДЕНЫ",
    "ОШИБКА",
    "ERROR",
)

HIDDEN_TECHNICAL_KEYS = {
    "audio_evidence_frame_label",
    "audio_evidence_frame_path",
    "source_path_policy",
    "video_evidence_frame_label",
    "video_evidence_frame_path",
}

EVIDENCE_PATH_KEYS = (
    "video_evidence_frame_path",
    "audio_evidence_frame_path",
)

EVIDENCE_LABEL_KEYS = (
    "video_evidence_frame_label",
    "audio_evidence_frame_label",
)


class PipelineStageError(Exception):
    def __init__(self, stage, message, raw_result="", media_info=None, technical_info=None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.raw_result = raw_result
        self.media_info = media_info
        self.technical_info = technical_info or {}


class LoadingStage:
    stage_name = "loading"

    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or MediaPreprocessor()

    def load(self, media_type, file_path):
        validation = validate_media_file(file_path, media_type)
        if not validation.is_valid:
            raise PipelineStageError(
                self.stage_name,
                validation.message,
                technical_info=validation.details,
            )

        try:
            return self.preprocessor.preprocess(media_type, file_path, validation.details)
        except Exception as exc:
            raise PipelineStageError(
                self.stage_name,
                f"Ошибка предобработки: {exc}",
                technical_info=validation.details,
            ) from exc


class AnalysisStage:
    stage_name = "analysis"

    def analyze(self, media_info):
        try:
            if media_info.media_type == AUDIO:
                from audio_detection.audio_analyzer import analyze_audio_with_details

                output = analyze_audio_with_details(media_info.file_path)
                media_info.technical_info.update(output.technical_info)
                return output.raw_result

            from video_detection.video_analyzer import analyze_video_with_details

            output = analyze_video_with_details(media_info.file_path)
            media_info.technical_info.update(output.technical_info)
            return output.raw_result
        except Exception as exc:
            raise PipelineStageError(
                self.stage_name,
                f"Ошибка анализа: {exc}",
                media_info=media_info,
            ) from exc


class ClassificationStage:
    stage_name = "classification"

    def classify(self, raw_result, media_info=None):
        if self.is_blocking_result(raw_result):
            raise PipelineStageError(
                self.stage_name,
                raw_result,
                raw_result=raw_result,
                media_info=media_info,
            )

        parsed = parse_model_result(raw_result)
        if parsed.is_error:
            raise PipelineStageError(
                self.stage_name,
                raw_result,
                raw_result=raw_result,
                media_info=media_info,
            )

        return parsed

    @staticmethod
    def is_blocking_result(raw_result):
        text = (raw_result or "").upper()
        return any(marker in text for marker in BLOCKING_RESULT_MARKERS)


class VisualizationStage:
    def build_findings(self, parsed, media_info):
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

        if media_info.media_type == VIDEO:
            findings.extend(self.video_issue_findings(media_info.technical_info))
        elif media_info.media_type == AUDIO:
            findings.extend(self.audio_issue_findings(media_info.technical_info))

        if media_info.duration is None:
            findings.append("Длительность файла не удалось определить на этапе предобработки.")

        return findings

    @staticmethod
    def video_issue_findings(technical_info):
        issue_findings = (technical_info or {}).get("video_issue_findings")
        if not issue_findings:
            return []
        if isinstance(issue_findings, (list, tuple)):
            return [str(finding) for finding in issue_findings if finding]
        return [str(issue_findings)]

    @staticmethod
    def audio_issue_findings(technical_info):
        issue_findings = (technical_info or {}).get("audio_issue_findings")
        if not issue_findings:
            return []
        if isinstance(issue_findings, (list, tuple)):
            return [str(finding) for finding in issue_findings if finding]
        return [str(issue_findings)]

    def build_error_findings(self, message, technical_info):
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

    def build_report_view(self, result):
        return ReportView(
            rows=self.report_rows(result),
            findings=result.findings or ["Замечаний нет."],
        )

    def evidence_frame_path(self, result):
        path_value = self.first_technical_value(result, EVIDENCE_PATH_KEYS)
        if not path_value:
            return None

        path = Path(str(path_value))
        candidate = path if path.is_absolute() else PROJECT_ROOT / path
        try:
            return candidate.resolve(strict=False)
        except OSError:
            return None

    def evidence_frame_label(self, result):
        return self.first_technical_value(result, EVIDENCE_LABEL_KEYS) or self.evidence_section_title(result)

    def evidence_section_title(self, result):
        if (result.technical_info or {}).get("audio_evidence_frame_path"):
            return "Фрагмент проверки"
        return "Кадр проверки"

    def build_evidence_frame_html(self, result, image_width=460):
        evidence_path = self.evidence_frame_path(result)
        if evidence_path is None or not evidence_path.is_file():
            return ""

        label = html.escape(str(self.evidence_frame_label(result)))
        image_src = self.evidence_frame_data_uri(evidence_path)
        if not image_src:
            return ""

        return f"""
            <div class="evidenceBlock">
                <h2>{html.escape(self.evidence_section_title(result))}</h2>
                <img src="{html.escape(image_src, quote=True)}" width="{int(image_width)}" />
                <p>{label}</p>
            </div>
        """

    @staticmethod
    def first_technical_value(result, keys):
        technical_info = result.technical_info or {}
        for key in keys:
            value = technical_info.get(key)
            if value:
                return value
        return None

    @staticmethod
    def evidence_frame_data_uri(evidence_path):
        try:
            payload = evidence_path.read_bytes()
        except OSError:
            return ""

        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def report_rows(self, result):
        rows = [
            ("Файл", result.file_name),
            ("Тип", media_label(result.media_type)),
            ("Размер", file_size_label(result.file_size)),
            ("Длительность", self.duration_label(result.duration)),
            ("Итог", "Ошибка" if result.status == STATUS_ERROR else verdict_label(result.verdict)),
            ("Вероятность", confidence_label(result.confidence)),
            ("Ответ модели", result.raw_result),
        ]

        for key, value in sorted((result.technical_info or {}).items()):
            if key not in HIDDEN_TECHNICAL_KEYS and value is not None and value != "":
                rows.append((self.technical_label(key), self.technical_value(value)))

        return rows

    @staticmethod
    def file_name(file_path):
        if not file_path:
            return "Без файла"
        return str(file_path).replace("\\", "/").split("/")[-1] or "Без файла"

    @staticmethod
    def file_size(file_path):
        return file_size(file_path) if file_path else None

    @staticmethod
    def duration_label(duration):
        if duration is None:
            return "-"

        total_seconds = int(round(duration))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def technical_label(key):
        labels = {
            "channels": "Каналы",
            "database_integrity_status": "Целостность записи",
            "database_record_hash": "Хэш записи БД",
            "extension": "Расширение",
            "format": "Формат",
            "fps": "FPS",
            "frame_count": "Кадры",
            "height": "Высота",
            "preprocess_warning": "Предобработка",
            "resolution": "Разрешение",
            "sample_rate": "Sample rate",
            "security_checks": "Проверки безопасности",
            "security_status": "Статус безопасности",
            "security_warning": "Предупреждение безопасности",
            "sha256": "SHA-256",
            "subtype": "Subtype",
            "audio_clipping_score": "Перегрузка аудиосигнала",
            "audio_evidence_frame_sha256": "SHA-256 фрагмента проверки",
            "audio_evidence_frame_time": "Время фрагмента проверки, сек",
            "audio_evidence_frame_warning": "Фрагмент проверки",
            "audio_issue_categories": "Категории аудио-аномалий",
            "audio_issue_findings": "Пояснения аудиоанализа",
            "audio_noise_score": "Фоновый шум",
            "audio_spectral_flatness_score": "Спектральная шумность",
            "audio_spectral_jump_score": "Спектральные скачки",
            "audio_suspicious_time": "Подозрительный фрагмент, сек",
            "audio_timbre_instability_score": "Нестабильность тембра",
            "video_artifact_score": "Визуальные артефакты лица",
            "video_evidence_frame_index": "Номер кадра проверки",
            "video_evidence_frame_label": "Подпись кадра проверки",
            "video_evidence_frame_sha256": "SHA-256 кадра проверки",
            "video_evidence_frame_time": "Время кадра проверки, сек",
            "video_evidence_frame_warning": "Кадр проверки",
            "video_face_motion_score": "Скачки положения лица",
            "video_faces_analyzed": "Проверено лиц",
            "video_frames_analyzed": "Проверено кадров видео",
            "video_issue_categories": "Категории видео-аномалий",
            "video_issue_findings": "Пояснения видеоанализа",
            "video_lip_motion_score": "Аномалии движения губ",
            "video_score_variance": "Нестабильность оценки лица",
            "width": "Ширина",
        }
        return labels.get(key, key)

    @staticmethod
    def technical_value(value):
        if isinstance(value, (list, tuple)):
            return "; ".join(str(item) for item in value)
        return str(value)
