import os
import threading

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


VIDEO_WEIGHTS_PATH = os.path.join("weights", "efficient_weights.weights.h5")
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

_VIDEO_MODEL = None
_FACE_CASCADE = None
_VIDEO_MODEL_LOCK = threading.Lock()
_FACE_CASCADE_LOCK = threading.Lock()

def build_model_architecture():
    # Создаем базу БЕЗ встроенных весов
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None 
    )
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        # ВАЖНО: Убираем ручную нормализацию, так как EfficientNetV2 
        # имеет встроенные слои Rescaling(1/255) и Normalization.
        # Оставляем только Lambda, если она была в Kaggle-коде.
        layers.Lambda(lambda x: tf.cast(x, tf.float32)), 
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def get_video_model():
    global _VIDEO_MODEL

    if _VIDEO_MODEL is not None:
        return _VIDEO_MODEL

    with _VIDEO_MODEL_LOCK:
        if _VIDEO_MODEL is not None:
            return _VIDEO_MODEL

        if not os.path.exists(VIDEO_WEIGHTS_PATH):
            raise FileNotFoundError("Веса видео-модели не найдены")

        model = build_model_architecture()
        model.load_weights(VIDEO_WEIGHTS_PATH)
        _VIDEO_MODEL = model
        return _VIDEO_MODEL


def get_face_cascade():
    global _FACE_CASCADE

    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    with _FACE_CASCADE_LOCK:
        if _FACE_CASCADE is not None:
            return _FACE_CASCADE

        if not os.path.exists(FACE_CASCADE_PATH):
            raise FileNotFoundError("Каскад распознавания лиц не найден")

        cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        if cascade.empty():
            raise RuntimeError("Каскад распознавания лиц не загружен")

        _FACE_CASCADE = cascade
        return _FACE_CASCADE


def clear_video_model_cache():
    global _VIDEO_MODEL, _FACE_CASCADE

    with _VIDEO_MODEL_LOCK:
        _VIDEO_MODEL = None

    with _FACE_CASCADE_LOCK:
        _FACE_CASCADE = None


def analyze_video(video_path):
    try:
        model = get_video_model()
        face_cascade = get_face_cascade()
    except Exception as e:
        return f"Ошибка модели: {e}"

    cap = cv2.VideoCapture(video_path)
    predictions = []
    
    while cap.isOpened() and len(predictions) < 30:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        for (x, y, w, h) in faces:
            # 1. Вырезаем лицо
            face_img = frame[y:y+h, x:x+w]
            # 2. Приводим к размеру 224x224
            face_img = cv2.resize(face_img, (224, 224))
            # 3. Конвертируем BGR (OpenCV) в RGB (как училась модель)
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # 4. Добавляем размерность батча
            face_img = np.expand_dims(face_img, axis=0)
            
            # Предсказание
            res = model.predict(face_img, verbose=0)
            score = float(res[0][0])
            predictions.append(score)
            break 
            
    cap.release()
    
    if not predictions: return "Лица не найдены"
    
    avg_score = np.mean(predictions)
    
    # Инвертируем логику, если модель на Kaggle выдавала 0 для фейка
    # Но обычно 1 = Fake, 0 = Real
    # Исправленная логика:
    if avg_score < 0.5:
        # Если число близко к 0, значит это DEEPFAKE
        return f"DEEPFAKE ({(1 - avg_score) * 100:.1f}%)"
    else:
        # Если число близко к 1, значит это ORIGINAL
        return f"ORIGINAL ({avg_score * 100:.1f}%)"
