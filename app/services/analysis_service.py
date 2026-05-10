from app.core.constants import (
    STATUS_COMPLETED,
    STATUS_ERROR,
    VERDICT_UNKNOWN,
)
from app.core.models import AnalysisResult
from app.services.pipeline_stages import (
    AnalysisStage,
    ClassificationStage,
    LoadingStage,
    PipelineStageError,
    VisualizationStage,
)


class AnalysisService:
    def __init__(self, loading_stage=None, analysis_stage=None, classification_stage=None, visualization_stage=None):
        self.loading_stage = loading_stage or LoadingStage()
        self.analysis_stage = analysis_stage or AnalysisStage()
        self.classification_stage = classification_stage or ClassificationStage()
        self.visualization_stage = visualization_stage or VisualizationStage()

    def analyze(self, media_type, file_path):
        try:
            media_info = self.loading_stage.load(media_type, file_path)
            raw_result = self.analysis_stage.analyze(media_info)
            parsed = self.classification_stage.classify(raw_result, media_info)
            findings = self.visualization_stage.build_findings(parsed, media_info)
        except PipelineStageError as exc:
            return self.error_result(
                media_type,
                file_path,
                exc.message,
                raw_result=exc.raw_result,
                media_info=exc.media_info,
                technical_info=exc.technical_info,
            )

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
        file_name = media_info.file_name if media_info else self.visualization_stage.file_name(file_path)
        file_path_value = media_info.file_path if media_info else str(file_path or "")
        file_size_value = media_info.file_size if media_info else self.visualization_stage.file_size(file_path)
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
            findings=self.visualization_stage.build_error_findings(message, technical_info),
        )

    @staticmethod
    def is_blocking_result(raw_result):
        return ClassificationStage.is_blocking_result(raw_result)

    @staticmethod
    def file_name(file_path):
        return VisualizationStage.file_name(file_path)

    @staticmethod
    def build_findings(parsed, media_info):
        return VisualizationStage().build_findings(parsed, media_info)

    @staticmethod
    def build_error_findings(message, technical_info):
        return VisualizationStage().build_error_findings(message, technical_info)
