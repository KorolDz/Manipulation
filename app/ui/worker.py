from PySide6.QtCore import QObject, Signal

from app.services.analysis_service import AnalysisService


class AnalysisWorker(QObject):
    finished = Signal(object)

    def __init__(self, media_type, file_path):
        super().__init__()
        self.media_type = media_type
        self.file_path = file_path
        self.analysis_service = AnalysisService()

    def run(self):
        result = self.analysis_service.analyze(self.media_type, self.file_path)
        self.finished.emit(result)
