import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QVBoxLayout, QWidget, QFileDialog, 
                             QLabel, QTextEdit)
from PySide6.QtCore import Qt
# Импортируем твои детекторы
from video_detection.video_analyzer import analyze_video
from audio_detection.audio_analyzer import analyze_audio

class DeepfakeDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система обнаружения дипфейков v1.0")
        self.resize(550, 450)

        # Главный виджет и слой
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Заголовок
        self.label = QLabel("Выберите медиафайл для проверки на Deepfake")
        self.label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        # Кнопки
        self.btn_video = QPushButton("🎬 Проверить Видео (.mp4, .avi, .mov)")
        self.btn_video.setHeight = 40
        self.btn_video.clicked.connect(self.process_video)
        self.layout.addWidget(self.btn_video)

        self.btn_audio = QPushButton("🎵 Проверить Аудио (.flac, .wav, .mp3)")
        self.btn_audio.setHeight = 40
        self.btn_audio.clicked.connect(self.process_audio)
        self.layout.addWidget(self.btn_audio)

        # Поле вывода результатов
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setPlaceholderText("Результаты анализа появятся здесь...")
        self.layout.addWidget(self.result_output)

        # Кнопка очистки
        self.btn_clear = QPushButton("Очистить логи")
        self.btn_clear.clicked.connect(lambda: self.result_output.clear())
        self.layout.addWidget(self.btn_clear)

    def process_video(self):
        # Добавили поддержку разных форматов видео
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть видео", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.result_output.append(f"🔍 Начинаю анализ видео: {os.path.basename(file_path)}")
            QApplication.processEvents() 
            try:
                res = analyze_video(file_path)
                self.result_output.append(f"📊 {res}\n" + "-"*30)
            except Exception as e:
                self.result_output.append(f"❌ Ошибка анализа видео: {str(e)}\n")

    def process_audio(self):
        # Добавили .flac в фильтр
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть аудио", "", "Audio Files (*.flac *.wav *.mp3 *.m4a)"
        )
        if file_path:
            self.result_output.append(f"🔍 Начинаю анализ аудио: {os.path.basename(file_path)}")
            QApplication.processEvents()
            try:
                res = analyze_audio(file_path)
                self.result_output.append(f"📊 {res}\n" + "-"*30)
            except Exception as e:
                self.result_output.append(f"❌ Ошибка анализа аудио: {str(e)}\n")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Сделаем оформление чуть приятнее (опционально)
    app.setStyle("Fusion") 
    window = DeepfakeDetectorApp()
    window.show()
    sys.exit(app.exec())