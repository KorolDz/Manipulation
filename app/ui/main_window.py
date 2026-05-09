import html

from PySide6.QtCore import QMarginsF, QThread, Qt
from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import (
    AUDIO,
    STATUS_ERROR,
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
    VIDEO,
)
from app.core.presentation import (
    confidence_label,
    dialog_filter,
    file_size_label,
    media_label,
    status_label,
    verdict_label,
)
from app.services.analysis_service import AnalysisService
from app.services.history_repository import HistoryRepository
from app.services.validation import validate_media_file
from app.ui.styles import APP_STYLESHEET
from app.ui.worker import AnalysisWorker


class DeepfakeDetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_thread = None
        self.current_worker = None
        self.current_result = None
        self.selected_file_path = ""
        self.history_records = []
        self.repository = HistoryRepository()
        self.analysis_service = AnalysisService()

        self.setWindowTitle("Deepfake Detector")
        self.resize(1120, 720)
        self.setMinimumSize(960, 620)

        self.setup_ui()
        self.setStyleSheet(APP_STYLESHEET)
        self.load_history()

    def setup_ui(self):
        root = QFrame()
        root.setObjectName("rootFrame")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(18)
        self.setCentralWidget(root)

        root_layout.addWidget(self.create_sidebar())
        root_layout.addWidget(self.create_workspace(), 1)

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(238)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(22, 24, 22, 24)
        layout.setSpacing(16)

        title = QLabel("Deepfake\nDetector")
        title.setObjectName("appTitle")

        self.audio_button = QPushButton("Аудио")
        self.audio_button.setObjectName("primaryButton")
        self.audio_button.clicked.connect(lambda: self.choose_file(AUDIO))

        self.video_button = QPushButton("Видео")
        self.video_button.setObjectName("secondaryButton")
        self.video_button.clicked.connect(lambda: self.choose_file(VIDEO))

        layout.addWidget(title)
        layout.addSpacing(18)
        layout.addWidget(self.audio_button)
        layout.addWidget(self.video_button)
        layout.addStretch(1)

        return sidebar

    def create_workspace(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        header_row = QHBoxLayout()
        page_title = QLabel("Панель анализа")
        page_title.setObjectName("pageTitle")
        header_row.addWidget(page_title)
        header_row.addStretch(1)

        cards = QGridLayout()
        cards.setSpacing(16)
        cards.addWidget(self.create_status_panel(), 0, 0)
        cards.addWidget(self.create_result_panel(), 0, 1)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 2)

        layout.addLayout(header_row)
        layout.addLayout(cards)
        layout.addWidget(self.create_report_panel(), 3)
        layout.addWidget(self.create_history_panel(), 1)

        return container

    def create_status_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        section_title = QLabel("Статус")
        section_title.setObjectName("sectionTitle")

        self.status_label = QLabel("Готово")
        self.status_label.setObjectName("statusReady")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.file_name_label = QLabel("-")
        self.file_name_label.setObjectName("fileNameLabel")
        self.file_name_label.setWordWrap(True)

        layout.addWidget(section_title)
        layout.addWidget(self.status_label)
        layout.addWidget(self.file_name_label)
        layout.addStretch(1)

        return panel

    def create_result_panel(self):
        panel = QFrame()
        panel.setObjectName("resultPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        section_title = QLabel("Результат")
        section_title.setObjectName("sectionTitle")

        self.result_label = QLabel("Нет данных")
        self.result_label.setObjectName("resultNeutral")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.result_meta_label = QLabel("-")
        self.result_meta_label.setObjectName("resultMeta")
        self.result_meta_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(section_title)
        layout.addStretch(1)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_meta_label)
        layout.addStretch(1)

        return panel

    def create_report_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        section_title = QLabel("Отчет")
        section_title.setObjectName("sectionTitle")

        self.export_pdf_button = QPushButton("Экспорт PDF")
        self.export_pdf_button.setObjectName("exportButton")
        self.export_pdf_button.setEnabled(False)
        self.export_pdf_button.clicked.connect(self.export_report_pdf)

        top_row.addWidget(section_title)
        top_row.addStretch(1)
        top_row.addWidget(self.export_pdf_button)

        tables_row = QHBoxLayout()
        tables_row.setSpacing(14)

        self.report_table = QTableWidget(0, 2)
        self.report_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.report_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.report_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.report_table.verticalHeader().setVisible(False)
        self.report_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.report_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        self.findings_table = QTableWidget(0, 1)
        self.findings_table.setHorizontalHeaderLabels(["Несоответствия"])
        self.findings_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.findings_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.findings_table.verticalHeader().setVisible(False)
        self.findings_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        tables_row.addWidget(self.report_table, 3)
        tables_row.addWidget(self.findings_table, 2)

        layout.addLayout(top_row)
        layout.addLayout(tables_row)

        return panel

    def create_history_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        section_title = QLabel("История")
        section_title.setObjectName("sectionTitle")

        clear_button = QPushButton("Очистить")
        clear_button.setObjectName("clearButton")
        clear_button.clicked.connect(self.clear_history)

        top_row.addWidget(section_title)
        top_row.addStretch(1)
        top_row.addWidget(clear_button)

        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["Дата", "Файл", "Тип", "Результат", "Точность"])
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.history_table.itemSelectionChanged.connect(self.on_history_selection_changed)

        layout.addLayout(top_row)
        layout.addWidget(self.history_table, 1)

        return panel

    def choose_file(self, media_type):
        if self.is_analysis_running():
            QMessageBox.information(self, "Статус", "Анализ выполняется.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, media_label(media_type), "", dialog_filter(media_type))
        if not file_path:
            return

        validation = validate_media_file(file_path, media_type)
        if not validation.is_valid:
            result = self.analysis_service.error_result(media_type, file_path, validation.message)
            self.repository.add(result)
            self.show_result(result)
            self.load_history()
            QMessageBox.warning(self, "Файл", validation.message)
            return

        self.start_analysis(media_type, file_path)

    def start_analysis(self, media_type, file_path):
        self.selected_file_path = file_path
        self.set_controls_enabled(False)
        self.set_busy_state(file_path)

        thread = QThread(self)
        worker = AnalysisWorker(media_type, file_path)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_analysis_finished)
        worker.finished.connect(lambda *_: thread.quit())
        worker.finished.connect(lambda *_: worker.deleteLater())
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.on_worker_thread_finished)

        self.current_thread = thread
        self.current_worker = worker
        thread.start()

    def on_analysis_finished(self, result):
        self.repository.add(result)
        self.set_controls_enabled(True)
        self.show_result(result)
        self.load_history()

    def on_worker_thread_finished(self):
        self.current_thread = None
        self.current_worker = None

    def set_busy_state(self, file_path):
        self.current_result = None
        self.status_label.setText("Анализ")
        self.status_label.setObjectName("statusBusy")
        self.file_name_label.setText(file_path)
        self.result_label.setText("В работе")
        self.result_label.setObjectName("resultNeutral")
        self.result_meta_label.setText("-")
        self.clear_report()

        self.refresh_widget_style(self.status_label)
        self.refresh_widget_style(self.result_label)

    def show_result(self, result):
        self.current_result = result
        self.status_label.setText(status_label(result.status))
        self.file_name_label.setText(result.file_path or "-")

        if result.status == STATUS_ERROR:
            self.status_label.setObjectName("statusError")
            self.result_label.setObjectName("resultDanger")
            self.result_label.setText("Ошибка")
            self.result_meta_label.setText(result.error_message or result.raw_result)
        elif result.verdict == VERDICT_DEEPFAKE:
            self.status_label.setObjectName("statusReady")
            self.result_label.setObjectName("resultDanger")
            self.result_label.setText(verdict_label(result.verdict))
            self.result_meta_label.setText(self.build_result_meta(result))
        elif result.verdict == VERDICT_ORIGINAL:
            self.status_label.setObjectName("statusReady")
            self.result_label.setObjectName("resultSuccess")
            self.result_label.setText(verdict_label(result.verdict))
            self.result_meta_label.setText(self.build_result_meta(result))
        else:
            self.status_label.setObjectName("statusReady")
            self.result_label.setObjectName("resultWarning")
            self.result_label.setText(verdict_label(result.verdict))
            self.result_meta_label.setText(self.build_result_meta(result))

        self.refresh_widget_style(self.status_label)
        self.refresh_widget_style(self.result_label)
        self.show_report(result)

    def build_result_meta(self, result):
        return (
            f"{media_label(result.media_type)} | "
            f"{confidence_label(result.confidence)} | "
            f"{file_size_label(result.file_size)}"
        )

    def load_history(self):
        records = self.repository.list_recent()
        self.history_records = records
        self.history_table.blockSignals(True)
        self.history_table.setRowCount(0)

        for record in records:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            values = [
                record.created_at.replace("T", " "),
                record.file_name,
                media_label(record.media_type),
                verdict_label(record.verdict) if record.status != STATUS_ERROR else "Ошибка",
                confidence_label(record.confidence),
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(record.error_message or record.raw_result)
                self.history_table.setItem(row, column, item)

        self.history_table.blockSignals(False)

    def on_history_selection_changed(self):
        row = self.history_table.currentRow()
        if row < 0 or row >= len(self.history_records):
            return

        self.show_result(self.history_records[row])

    def clear_history(self):
        self.repository.clear()
        self.load_history()
        self.current_result = None
        self.clear_report()

    def clear_report(self):
        self.report_table.setRowCount(0)
        self.findings_table.setRowCount(0)
        self.export_pdf_button.setEnabled(False)

    def show_report(self, result):
        self.export_pdf_button.setEnabled(True)
        rows = self.report_rows(result)
        self.report_table.setRowCount(0)
        for parameter, value in rows:
            row = self.report_table.rowCount()
            self.report_table.insertRow(row)
            self.report_table.setItem(row, 0, QTableWidgetItem(parameter))
            self.report_table.setItem(row, 1, QTableWidgetItem(value or "-"))

        findings = result.findings or ["Замечаний нет."]
        self.findings_table.setRowCount(0)
        for finding in findings:
            row = self.findings_table.rowCount()
            self.findings_table.insertRow(row)
            self.findings_table.setItem(row, 0, QTableWidgetItem(str(finding)))

        self.report_table.resizeRowsToContents()
        self.findings_table.resizeRowsToContents()

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
            if value is not None and value != "":
                rows.append((self.technical_label(key), str(value)))

        return rows

    def export_report_pdf(self):
        if self.current_result is None:
            QMessageBox.information(self, "Экспорт PDF", "Нет отчета для экспорта.")
            return

        default_name = self.default_pdf_name(self.current_result)
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчет", default_name, "PDF (*.pdf)")
        if not file_path:
            return

        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"

        try:
            self.write_report_pdf(self.current_result, file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт PDF", f"Не удалось сохранить PDF: {exc}")
            return

        QMessageBox.information(self, "Экспорт PDF", "Отчет сохранен.")

    def write_report_pdf(self, result, file_path):
        writer = QPdfWriter(file_path)
        writer.setPageSize(QPageSize(QPageSize.A4))
        writer.setPageMargins(QMarginsF(14, 14, 14, 14), QPageLayout.Millimeter)

        document = QTextDocument()
        document.setHtml(self.build_report_html(result))
        document.print_(writer)

    def build_report_html(self, result):
        verdict = "Ошибка" if result.status == STATUS_ERROR else verdict_label(result.verdict)
        confidence = confidence_label(result.confidence)
        rows_html = "\n".join(
            "<tr>"
            f"<th>{html.escape(parameter)}</th>"
            f"<td>{html.escape(str(value or '-'))}</td>"
            "</tr>"
            for parameter, value in self.report_rows(result)
        )
        findings = result.findings or ["Замечаний нет."]
        findings_html = "\n".join(f"<li>{html.escape(str(finding))}</li>" for finding in findings)

        return f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    color: #111827;
                    font-family: "Segoe UI", Arial, sans-serif;
                    font-size: 11pt;
                }}
                .header {{
                    border-bottom: 2px solid #2563eb;
                    margin-bottom: 18px;
                    padding-bottom: 12px;
                }}
                h1 {{
                    color: #111827;
                    font-size: 22pt;
                    margin: 0 0 6px 0;
                }}
                .subtitle {{
                    color: #4b5563;
                    font-size: 10pt;
                }}
                .badge {{
                    background: #eff6ff;
                    border: 1px solid #bfdbfe;
                    border-radius: 8px;
                    color: #1d4ed8;
                    display: inline-block;
                    font-size: 14pt;
                    font-weight: 700;
                    margin: 10px 0 16px 0;
                    padding: 10px 14px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th {{
                    background: #f8fafc;
                    color: #334155;
                    font-weight: 700;
                    text-align: left;
                    width: 32%;
                }}
                th, td {{
                    border: 1px solid #d9e1ea;
                    padding: 8px 10px;
                    vertical-align: top;
                }}
                h2 {{
                    color: #111827;
                    font-size: 15pt;
                    margin: 20px 0 8px 0;
                }}
                ul {{
                    margin-top: 6px;
                    padding-left: 22px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Отчет о достоверности файла</h1>
                <div class="subtitle">Deepfake Detector · локальный анализ медиафайла</div>
            </div>
            <div class="badge">{html.escape(verdict)} · {html.escape(confidence)}</div>
            <h2>Параметры проверки</h2>
            <table>{rows_html}</table>
            <h2>Обнаруженные несоответствия</h2>
            <ul>{findings_html}</ul>
        </body>
        </html>
        """

    @staticmethod
    def default_pdf_name(result):
        base_name = result.file_name.rsplit(".", 1)[0] or "report"
        safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in base_name)
        return f"{safe_name}_report.pdf"

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
            "extension": "Расширение",
            "format": "Формат",
            "fps": "FPS",
            "frame_count": "Кадры",
            "height": "Высота",
            "preprocess_warning": "Предобработка",
            "resolution": "Разрешение",
            "sample_rate": "Sample rate",
            "subtype": "Subtype",
            "width": "Ширина",
        }
        return labels.get(key, key)

    def set_controls_enabled(self, enabled):
        self.audio_button.setEnabled(enabled)
        self.video_button.setEnabled(enabled)

    def is_analysis_running(self):
        return self.current_thread is not None and self.current_thread.isRunning()

    @staticmethod
    def refresh_widget_style(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
