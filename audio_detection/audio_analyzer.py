import librosa
import numpy as np
import tensorflow as tf
import os

def analyze_audio(audio_path):
    model_path = os.path.join('weights', 'audio_deepfake_v1.h5')
    
    if not os.path.exists(model_path):
        return "Ошибка: Веса аудио-модели не найдены"

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        
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
            
    except Exception as e:
        return f"Ошибка анализа аудио: {e}"