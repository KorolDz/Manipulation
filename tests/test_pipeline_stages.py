import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core.constants import (
    AUDIO,
    STATUS_COMPLETED,
    STATUS_ERROR,
    VIDEO,
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
)
from app.core.models import AnalysisResult, MediaInfo, ParsedResult
from app.services.analysis_service import AnalysisService
from app.services.pipeline_stages import AnalysisStage, ClassificationStage, PipelineStageError, VisualizationStage


class AnalysisPipelineTests(unittest.TestCase):
    def test_successful_pipeline_runs_stages_in_order(self):
        calls = []
        media_info = MediaInfo(
            file_path="sample.wav",
            file_name="sample.wav",
            media_type=AUDIO,
            file_size=128,
            extension=".wav",
            duration=1.25,
            technical_info={"sha256": "abc123", "security_status": "Пройдено"},
        )

        service = AnalysisService(
            loading_stage=FakeLoadingStage(calls, media_info),
            analysis_stage=FakeAnalysisStage(calls, "DEEPFAKE (Вероятность: 91.2%)"),
            classification_stage=RecordingClassificationStage(calls),
            visualization_stage=RecordingVisualizationStage(calls),
        )

        result = service.analyze(AUDIO, "sample.wav")

        self.assertEqual(calls, ["loading", "analysis", "classification", "visualization"])
        self.assertEqual(result.status, STATUS_COMPLETED)
        self.assertEqual(result.verdict, VERDICT_DEEPFAKE)
        self.assertEqual(result.confidence, 91.2)
        self.assertIn("Высокая вероятность фальсификации: 91.2%.", result.findings)

    def test_loading_error_stops_pipeline(self):
        calls = []
        service = AnalysisService(
            loading_stage=FailingLoadingStage(calls),
            analysis_stage=FakeAnalysisStage(calls, "ORIGINAL (Вероятность: 90.0%)"),
            classification_stage=RecordingClassificationStage(calls),
            visualization_stage=VisualizationStage(),
        )

        result = service.analyze(AUDIO, "missing.wav")

        self.assertEqual(calls, ["loading"])
        self.assertEqual(result.status, STATUS_ERROR)
        self.assertEqual(result.error_message, "Файл не найден.")
        self.assertEqual(result.technical_info["security_warning"], "Файл не найден.")
        self.assertIn("Файл не найден.", result.findings)


class ClassificationStageTests(unittest.TestCase):
    def test_classifies_deepfake_result(self):
        parsed = ClassificationStage().classify("DEEPFAKE (Вероятность: 88.0%)")

        self.assertEqual(parsed.verdict, VERDICT_DEEPFAKE)
        self.assertEqual(parsed.confidence, 88.0)

    def test_classifies_original_result(self):
        parsed = ClassificationStage().classify("ORIGINAL (Вероятность: 90.5%)")

        self.assertEqual(parsed.verdict, VERDICT_ORIGINAL)
        self.assertEqual(parsed.confidence, 90.5)

    def test_rejects_no_faces_result(self):
        with self.assertRaises(PipelineStageError) as context:
            ClassificationStage().classify("Лица не найдены")

        self.assertEqual(context.exception.stage, "classification")
        self.assertEqual(context.exception.raw_result, "Лица не найдены")

    def test_rejects_error_result(self):
        with self.assertRaises(PipelineStageError) as context:
            ClassificationStage().classify("ОШИБКА модели")

        self.assertEqual(context.exception.stage, "classification")
        self.assertEqual(context.exception.raw_result, "ОШИБКА модели")


class AnalysisStageVideoTests(unittest.TestCase):
    def test_video_structured_output_updates_technical_info_and_returns_raw_result(self):
        from video_detection.video_analyzer import VideoAnalysisOutput

        media_info = MediaInfo(
            file_path="sample.mp4",
            file_name="sample.mp4",
            media_type=VIDEO,
            file_size=256,
            extension=".mp4",
            duration=2.0,
            technical_info={"sha256": "abc123"},
        )
        output = VideoAnalysisOutput(
            raw_result="DEEPFAKE (91.0%)",
            technical_info={
                "video_faces_analyzed": 8,
                "video_issue_findings": ["Обнаружена нестабильность признаков лица между кадрами, возможна подмена лица."],
            },
        )

        with patch("video_detection.video_analyzer.analyze_video_with_details", return_value=output):
            raw_result = AnalysisStage().analyze(media_info)

        self.assertEqual(raw_result, "DEEPFAKE (91.0%)")
        self.assertEqual(media_info.technical_info["sha256"], "abc123")
        self.assertEqual(media_info.technical_info["video_faces_analyzed"], 8)
        self.assertIn("подмена лица", media_info.technical_info["video_issue_findings"][0])


class AnalysisStageAudioTests(unittest.TestCase):
    def test_audio_structured_output_updates_technical_info_and_returns_raw_result(self):
        from audio_detection.audio_analyzer import AudioAnalysisOutput

        media_info = MediaInfo(
            file_path="sample.wav",
            file_name="sample.wav",
            media_type=AUDIO,
            file_size=128,
            extension=".wav",
            duration=1.0,
            technical_info={"sha256": "abc123"},
        )
        output = AudioAnalysisOutput(
            raw_result="DEEPFAKE (Вероятность: 92.0%)",
            technical_info={
                "audio_noise_score": 0.8,
                "audio_issue_findings": ["В аудио возможны следы шумовой обработки или нестабильного фонового шума."],
            },
        )

        with patch("audio_detection.audio_analyzer.analyze_audio_with_details", return_value=output):
            raw_result = AnalysisStage().analyze(media_info)

        self.assertEqual(raw_result, "DEEPFAKE (Вероятность: 92.0%)")
        self.assertEqual(media_info.technical_info["sha256"], "abc123")
        self.assertEqual(media_info.technical_info["audio_noise_score"], 0.8)
        self.assertIn("шумовой обработки", media_info.technical_info["audio_issue_findings"][0])


class VisualizationStageTests(unittest.TestCase):
    def test_report_view_contains_public_rows_and_hides_source_path_policy(self):
        result = AnalysisResult(
            file_path="voice.wav",
            file_name="voice.wav",
            media_type=AUDIO,
            file_size=128,
            status=STATUS_COMPLETED,
            verdict=VERDICT_ORIGINAL,
            confidence=97.0,
            raw_result="ORIGINAL (Вероятность: 97.0%)",
            duration=2.0,
            technical_info={
                "source_path_policy": "filename_only",
                "sha256": "abc123",
                "sample_rate": 16000,
            },
            findings=["Критичные признаки подделки не обнаружены."],
        )

        report_view = VisualizationStage().build_report_view(result)
        rows = dict(report_view.rows)

        self.assertEqual(rows["Файл"], "voice.wav")
        self.assertEqual(rows["Тип"], "Аудио")
        self.assertEqual(rows["Размер"], "128 Б")
        self.assertEqual(rows["Итог"], "Оригинал")
        self.assertEqual(rows["Вероятность"], "97.0%")
        self.assertEqual(rows["Ответ модели"], "ORIGINAL (Вероятность: 97.0%)")
        self.assertEqual(rows["SHA-256"], "abc123")
        self.assertNotIn("source_path_policy", rows)
        self.assertNotIn("filename_only", rows.values())
        self.assertEqual(report_view.findings, ["Критичные признаки подделки не обнаружены."])

    def test_video_findings_are_added_from_technical_info(self):
        stage = VisualizationStage()
        media_info = MediaInfo(
            file_path="sample.mp4",
            file_name="sample.mp4",
            media_type=VIDEO,
            file_size=256,
            extension=".mp4",
            duration=2.0,
            technical_info={
                "video_faces_analyzed": 8,
                "video_issue_categories": ["подмена лица"],
                "video_issue_findings": ["Обнаружена нестабильность признаков лица между кадрами, возможна подмена лица."],
            },
        )

        findings = stage.build_findings(ParsedResult(VERDICT_DEEPFAKE, 91.0), media_info)

        self.assertIn("Высокая вероятность фальсификации: 91.0%.", findings)
        self.assertIn("Обнаружена нестабильность признаков лица между кадрами, возможна подмена лица.", findings)

        result = AnalysisResult(
            file_path=media_info.file_path,
            file_name=media_info.file_name,
            media_type=VIDEO,
            file_size=media_info.file_size,
            status=STATUS_COMPLETED,
            verdict=VERDICT_DEEPFAKE,
            confidence=91.0,
            raw_result="DEEPFAKE (91.0%)",
            duration=media_info.duration,
            technical_info=media_info.technical_info,
            findings=findings,
        )
        rows = dict(stage.build_report_view(result).rows)

        self.assertEqual(rows["Проверено лиц"], "8")
        self.assertEqual(rows["Категории видео-аномалий"], "подмена лица")

    def test_audio_findings_are_added_from_technical_info(self):
        stage = VisualizationStage()
        media_info = MediaInfo(
            file_path="sample.wav",
            file_name="sample.wav",
            media_type=AUDIO,
            file_size=128,
            extension=".wav",
            duration=1.0,
            technical_info={
                "audio_noise_score": 0.8,
                "audio_issue_categories": ["шумовая обработка"],
                "audio_issue_findings": ["В аудио возможны следы шумовой обработки или нестабильного фонового шума."],
            },
        )

        findings = stage.build_findings(ParsedResult(VERDICT_DEEPFAKE, 92.0), media_info)

        self.assertIn("Высокая вероятность фальсификации: 92.0%.", findings)
        self.assertIn("В аудио возможны следы шумовой обработки или нестабильного фонового шума.", findings)

        result = AnalysisResult(
            file_path=media_info.file_path,
            file_name=media_info.file_name,
            media_type=AUDIO,
            file_size=media_info.file_size,
            status=STATUS_COMPLETED,
            verdict=VERDICT_DEEPFAKE,
            confidence=92.0,
            raw_result="DEEPFAKE (Вероятность: 92.0%)",
            duration=media_info.duration,
            technical_info=media_info.technical_info,
            findings=findings,
        )
        rows = dict(stage.build_report_view(result).rows)

        self.assertEqual(rows["Фоновый шум"], "0.8")
        self.assertEqual(rows["Категории аудио-аномалий"], "шумовая обработка")

    def test_evidence_frame_html_is_built_only_when_file_exists(self):
        stage = VisualizationStage()
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            image_path.write_bytes(b"jpeg")
            result = AnalysisResult(
                file_path="sample.mp4",
                file_name="sample.mp4",
                media_type=VIDEO,
                file_size=256,
                status=STATUS_COMPLETED,
                verdict=VERDICT_DEEPFAKE,
                confidence=91.0,
                raw_result="DEEPFAKE (91.0%)",
            technical_info={
                "video_evidence_frame_path": str(image_path),
                "video_evidence_frame_label": "Кадр проверки: подмена лица",
                "video_evidence_frame_index": 7,
            },
        )

            html = stage.build_evidence_frame_html(result)

        self.assertIn("Кадр проверки", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertLess(html.index("<img"), html.index("Кадр проверки: подмена лица"))

        rows = dict(stage.build_report_view(result).rows)
        self.assertNotIn("Подпись кадра проверки", rows)
        self.assertEqual(rows["Номер кадра проверки"], "7")

        missing_result = AnalysisResult(
            file_path="sample.mp4",
            file_name="sample.mp4",
            media_type=VIDEO,
            file_size=256,
            status=STATUS_COMPLETED,
            verdict=VERDICT_DEEPFAKE,
            confidence=91.0,
            raw_result="DEEPFAKE (91.0%)",
            technical_info={"video_evidence_frame_path": str(Path(tmp_dir) / "missing.jpg")},
        )
        self.assertEqual(stage.build_evidence_frame_html(missing_result), "")

    def test_audio_evidence_html_uses_fragment_title(self):
        stage = VisualizationStage()
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "audio.jpg"
            image_path.write_bytes(b"jpeg")
            result = AnalysisResult(
                file_path="sample.wav",
                file_name="sample.wav",
                media_type=AUDIO,
                file_size=128,
                status=STATUS_COMPLETED,
                verdict=VERDICT_DEEPFAKE,
                confidence=92.0,
                raw_result="DEEPFAKE (Вероятность: 92.0%)",
                technical_info={
                    "audio_evidence_frame_path": str(image_path),
                    "audio_evidence_frame_label": "Фрагмент проверки: шумовая обработка",
                },
            )

            html = stage.build_evidence_frame_html(result)

        self.assertIn("Фрагмент проверки", html)
        self.assertNotIn("<h2>Кадр проверки</h2>", html)
        self.assertIn("data:image/jpeg;base64,", html)


class FakeLoadingStage:
    def __init__(self, calls, media_info):
        self.calls = calls
        self.media_info = media_info

    def load(self, media_type, file_path):
        self.calls.append("loading")
        return self.media_info


class FailingLoadingStage:
    def __init__(self, calls):
        self.calls = calls

    def load(self, media_type, file_path):
        self.calls.append("loading")
        raise PipelineStageError(
            "loading",
            "Файл не найден.",
            technical_info={"security_warning": "Файл не найден."},
        )


class FakeAnalysisStage:
    def __init__(self, calls, raw_result):
        self.calls = calls
        self.raw_result = raw_result

    def analyze(self, media_info):
        self.calls.append("analysis")
        return self.raw_result


class RecordingClassificationStage(ClassificationStage):
    def __init__(self, calls):
        self.calls = calls

    def classify(self, raw_result, media_info=None):
        self.calls.append("classification")
        return super().classify(raw_result, media_info)


class RecordingVisualizationStage(VisualizationStage):
    def __init__(self, calls):
        self.calls = calls

    def build_findings(self, parsed, media_info):
        self.calls.append("visualization")
        return super().build_findings(parsed, media_info)


if __name__ == "__main__":
    unittest.main()
