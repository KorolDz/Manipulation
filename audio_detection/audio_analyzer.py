import os
import threading

import librosa
import numpy as np
import tensorflow as tf


AUDIO_MODEL_PATH = os.path.join("weights", "audio_deepfake_v1.h5")
_AUDIO_MODEL = None
_AUDIO_MODEL_LOCK = threading.Lock()


def get_audio_model():
    global _AUDIO_MODEL

    if _AUDIO_MODEL is not None:
        return _AUDIO_MODEL

    with _AUDIO_MODEL_LOCK:
        if _AUDIO_MODEL is not None:
            return _AUDIO_MODEL

        if not os.path.exists(AUDIO_MODEL_PATH):
            raise FileNotFoundError("Веса аудио-модели не найдены")

        _AUDIO_MODEL = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
        return _AUDIO_MODEL


def clear_audio_model_cache():
    global _AUDIO_MODEL

    with _AUDIO_MODEL_LOCK:
        _AUDIO_MODEL = None


def analyze_audio(audio_path):
    try:
        model = get_audio_model()
        
        # Загружаем аудио (до 4 секунд)
        y, sr = librosa.load(audio_path, duration=4)
        
        # Извлекаем MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Подгоняем размер под 157 (стандарт для твоей модели)
        if mfcc.shape[1] < 157:
            mfcc = np.pad(mfcc, ((0, 0), (0, 157 - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :157]
        
        # Подготовка тензора (1, 40, 157, 1)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        # Получаем предсказание
        prediction = model.predict(mfcc, verbose=0)[0][0]
        avg_score = float(prediction)

        # Вывод в стиле видео-анализатора
        if avg_score > 0.5:
            return f"DEEPFAKE (Вероятность: {avg_score * 100:.1f}%)"
        else:
            return f"ORIGINAL (Вероятность: {(1 - avg_score) * 100:.1f}%)"
            
    except FileNotFoundError as e:
        return f"Ошибка: {e}"
    except Exception as e:
        return f"Ошибка анализа аудио: {e}"
