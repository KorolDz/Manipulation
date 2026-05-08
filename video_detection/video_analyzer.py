import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, models

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

def analyze_video(video_path):
    weights_path = os.path.join('weights', 'efficient_weights.weights.h5')
    cascade_path = 'haarcascade_frontalface_default.xml'
    
    # Загружаем модель один раз
    try:
        model = build_model_architecture()
        model.load_weights(weights_path)
    except Exception as e:
        return f"Ошибка модели: {e}"

    face_cascade = cv2.CascadeClassifier(cascade_path)
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