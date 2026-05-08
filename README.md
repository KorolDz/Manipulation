# Deepfake Detection System (Video & Audio)

Программный комплекс для обнаружения манипуляций в видео и аудио потоках.
Точность детекции: **~99%**.

## Структура проекта
- `video_detection/` — анализ видео (EfficientNetV2B0)
- `audio_detection/` — анализ аудио (CNN + MFCC)
- `weights/` — обученные веса моделей
- `main.py` — запуск системы

## Установка и запуск
1. Создать окружение: `python -m venv venv`
2. Активировать: `venv\Scripts\activate` (Windows)
3. Установить зависимости: `pip install -r requirements.txt`
4. Запуск: `python main.py`