import hashlib
import os
import threading
from dataclasses import dataclass, field

import cv2
import librosa
import numpy as np
import tensorflow as tf

from app.core.constants import EVIDENCE_FRAMES_DIR, PROJECT_ROOT


AUDIO_MODEL_PATH = os.path.join("weights", "audio_deepfake_v1.h5")
_AUDIO_MODEL = None
_AUDIO_MODEL_LOCK = threading.Lock()

AUDIO_DURATION_SECONDS = 4
MODEL_MFCC_FRAMES = 157
MODEL_MFCC_COUNT = 40
AUDIO_EVIDENCE_WIDTH = 720
AUDIO_EVIDENCE_HEIGHT = 420
AUDIO_EVIDENCE_JPEG_QUALITY = 90
NOISE_SCORE_THRESHOLD = 0.42
SPECTRAL_FLATNESS_THRESHOLD = 0.32
TIMBRE_INSTABILITY_THRESHOLD = 0.48
SPECTRAL_JUMP_THRESHOLD = 0.42
CLIPPING_THRESHOLD = 0.02


@dataclass(frozen=True)
class AudioAnalysisOutput:
    raw_result: str
    technical_info: dict[str, object] = field(default_factory=dict)


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
    return analyze_audio_with_details(audio_path).raw_result


def analyze_audio_with_details(audio_path):
    try:
        model = get_audio_model()

        y, sr = librosa.load(audio_path, duration=AUDIO_DURATION_SECONDS)
        mfcc = prepare_model_mfcc(y, sr)
        prediction = model.predict(mfcc[np.newaxis, ..., np.newaxis], verbose=0)[0][0]
        avg_score = float(prediction)

        raw_result = format_audio_result(avg_score)
        technical_info = build_audio_technical_info(y, sr)
        try:
            technical_info.update(save_audio_evidence_image(y, sr, technical_info))
        except Exception as exc:
            technical_info["audio_evidence_frame_warning"] = f"Не удалось сохранить фрагмент проверки: {exc}"

        return AudioAnalysisOutput(raw_result, technical_info)
    except FileNotFoundError as e:
        return AudioAnalysisOutput(f"Ошибка: {e}")
    except Exception as e:
        return AudioAnalysisOutput(f"Ошибка анализа аудио: {e}")


def prepare_model_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MODEL_MFCC_COUNT)
    if mfcc.shape[1] < MODEL_MFCC_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MODEL_MFCC_FRAMES - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MODEL_MFCC_FRAMES]
    return mfcc


def format_audio_result(score):
    if score > 0.5:
        return f"DEEPFAKE (Вероятность: {score * 100:.1f}%)"
    return f"ORIGINAL (Вероятность: {(1 - score) * 100:.1f}%)"


def build_audio_technical_info(y, sr):
    metrics = calculate_audio_metrics(y, sr)
    issue_details = interpret_audio_metrics(**metrics)
    return {
        "audio_noise_score": round(metrics["noise_score"], 4),
        "audio_spectral_flatness_score": round(metrics["spectral_flatness_score"], 4),
        "audio_timbre_instability_score": round(metrics["timbre_instability_score"], 4),
        "audio_spectral_jump_score": round(metrics["spectral_jump_score"], 4),
        "audio_clipping_score": round(metrics["clipping_score"], 4),
        "audio_suspicious_time": metrics["suspicious_time"],
        **issue_details,
    }


def calculate_audio_metrics(y, sr):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return {
            "noise_score": 0.0,
            "spectral_flatness_score": 0.0,
            "timbre_instability_score": 0.0,
            "spectral_jump_score": 0.0,
            "clipping_score": 0.0,
            "suspicious_time": None,
        }

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    noise_floor = float(np.percentile(rms, 20)) if rms.size else 0.0
    loud_level = float(np.percentile(rms, 95)) if rms.size else 0.0
    noise_score = clamp01(noise_floor / max(loud_level * 0.55, 1e-6))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    spectral_flatness_score = clamp01(float(np.mean(flatness)) / 0.45) if flatness.size else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] > 1:
        mfcc_delta = np.abs(np.diff(mfcc, axis=1))
        timbre_instability_score = clamp01(float(np.mean(mfcc_delta)) / 18.0)
    else:
        timbre_instability_score = 0.0

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max) if mel.size else np.zeros((1, 1))
    if mel_db.shape[1] > 1:
        frame_jumps = np.mean(np.abs(np.diff(mel_db, axis=1)), axis=0)
        spectral_jump_score = clamp01(float(np.percentile(frame_jumps, 95)) / 28.0)
        suspicious_frame = int(np.argmax(frame_jumps)) + 1
        suspicious_time = round(float(librosa.frames_to_time(suspicious_frame, sr=sr, hop_length=512)), 2)
    else:
        spectral_jump_score = 0.0
        suspicious_time = 0.0

    clipping_score = clamp01(float(np.mean(np.abs(y) >= 0.98)) / 0.08)

    return {
        "noise_score": noise_score,
        "spectral_flatness_score": spectral_flatness_score,
        "timbre_instability_score": timbre_instability_score,
        "spectral_jump_score": spectral_jump_score,
        "clipping_score": clipping_score,
        "suspicious_time": suspicious_time,
    }


def interpret_audio_metrics(
    noise_score,
    spectral_flatness_score,
    timbre_instability_score,
    spectral_jump_score,
    clipping_score,
    suspicious_time=None,
):
    findings = []
    categories = []

    if noise_score >= NOISE_SCORE_THRESHOLD:
        categories.append("шумовая обработка")
        findings.append("В аудио возможны следы шумовой обработки или нестабильного фонового шума.")

    if spectral_flatness_score >= SPECTRAL_FLATNESS_THRESHOLD:
        categories.append("спектральная плоскость")
        findings.append("Спектральная структура выглядит слишком шумной или сглаженной, возможны признаки синтетической обработки.")

    if timbre_instability_score >= TIMBRE_INSTABILITY_THRESHOLD:
        categories.append("нестабильность тембра")
        findings.append("Зафиксирована нестабильность тембра голоса, возможны следы генерации или редактирования.")

    if spectral_jump_score >= SPECTRAL_JUMP_THRESHOLD:
        categories.append("спектральные скачки")
        findings.append("Обнаружены резкие спектральные переходы, возможны монтажные склейки или цифровая обработка.")

    if clipping_score >= CLIPPING_THRESHOLD:
        categories.append("перегрузка сигнала")
        findings.append("Возможна перегрузка или обрезка аудиосигнала.")

    if not findings:
        findings.append("Дополнительные аудио-признаки явных аномалий шума, тембра или спектра не выявили.")

    return {
        "audio_issue_categories": categories,
        "audio_issue_findings": findings,
    }


def save_audio_evidence_image(y, sr, technical_info):
    EVIDENCE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    image = build_audio_evidence_image(y, sr, technical_info.get("audio_suspicious_time"))
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), AUDIO_EVIDENCE_JPEG_QUALITY],
    )
    if not success:
        raise ValueError("OpenCV не смог закодировать JPEG")

    payload = encoded.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    output_path = EVIDENCE_FRAMES_DIR / f"audio_{digest[:20]}.jpg"
    if not output_path.exists():
        output_path.write_bytes(payload)

    return {
        "audio_evidence_frame_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "audio_evidence_frame_label": evidence_fragment_label(technical_info.get("audio_issue_categories", [])),
        "audio_evidence_frame_time": technical_info.get("audio_suspicious_time"),
        "audio_evidence_frame_sha256": digest,
    }


def build_audio_evidence_image(y, sr, suspicious_time=None):
    canvas = np.full((AUDIO_EVIDENCE_HEIGHT, AUDIO_EVIDENCE_WIDTH, 3), 255, dtype=np.uint8)
    draw_waveform(canvas, y)
    draw_spectrogram(canvas, y, sr)
    draw_suspicious_time_marker(canvas, y, sr, suspicious_time)
    draw_label(canvas, "Waveform", (18, 28))
    draw_label(canvas, "Log-mel spectrogram", (18, 220))
    return canvas


def draw_waveform(canvas, y):
    top, bottom = 36, 180
    left, right = 28, AUDIO_EVIDENCE_WIDTH - 18
    cv2.line(canvas, (left, (top + bottom) // 2), (right, (top + bottom) // 2), (210, 220, 232), 1)
    if y.size == 0:
        return

    samples = np.asarray(y, dtype=np.float32)
    max_abs = float(np.max(np.abs(samples))) or 1.0
    x_values = np.linspace(left, right, num=min(samples.size, right - left + 1)).astype(int)
    indexes = np.linspace(0, samples.size - 1, num=x_values.size).astype(int)
    y_values = ((top + bottom) / 2 - (samples[indexes] / max_abs) * ((bottom - top) / 2 - 4)).astype(int)
    points = np.column_stack([x_values, y_values])
    cv2.polylines(canvas, [points], False, (37, 99, 235), 1)


def draw_spectrogram(canvas, y, sr):
    top, bottom = 232, AUDIO_EVIDENCE_HEIGHT - 28
    left, right = 28, AUDIO_EVIDENCE_WIDTH - 18
    if y.size == 0:
        return

    spectrogram = simple_log_spectrogram(y)
    mel_norm = normalize_to_uint8(spectrogram)
    mel_color = cv2.applyColorMap(np.flipud(mel_norm), cv2.COLORMAP_VIRIDIS)
    mel_color = cv2.resize(mel_color, (right - left, bottom - top), interpolation=cv2.INTER_AREA)
    canvas[top:bottom, left:right] = mel_color
    cv2.rectangle(canvas, (left, top), (right, bottom), (203, 213, 225), 1)


def simple_log_spectrogram(y, frame_size=512, hop_length=256):
    samples = np.asarray(y, dtype=np.float32)
    if samples.size < frame_size:
        samples = np.pad(samples, (0, frame_size - samples.size), mode="constant")

    frames = []
    window = np.hanning(frame_size).astype(np.float32)
    for start in range(0, max(samples.size - frame_size + 1, 1), hop_length):
        frame = samples[start:start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size), mode="constant")
        spectrum = np.abs(np.fft.rfft(frame * window))
        frames.append(np.log1p(spectrum))

    if not frames:
        frames.append(np.zeros(frame_size // 2 + 1, dtype=np.float32))

    return np.asarray(frames, dtype=np.float32).T


def draw_suspicious_time_marker(canvas, y, sr, suspicious_time):
    if suspicious_time is None or y.size == 0 or sr <= 0:
        return

    duration = max(y.size / float(sr), 1e-6)
    x = int(28 + clamp01(float(suspicious_time) / duration) * (AUDIO_EVIDENCE_WIDTH - 46))
    cv2.line(canvas, (x, 36), (x, AUDIO_EVIDENCE_HEIGHT - 28), (0, 0, 220), 2)


def draw_label(canvas, text, origin):
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 41, 59), 1, cv2.LINE_AA)


def normalize_to_uint8(values):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum:
        return np.zeros(values.shape, dtype=np.uint8)
    return ((values - minimum) / (maximum - minimum) * 255).astype(np.uint8)


def evidence_fragment_label(issue_categories):
    categories = [str(category) for category in (issue_categories or []) if category]
    if categories:
        return "Фрагмент проверки: " + ", ".join(categories)
    return "Фрагмент проверки: явные аудио-аномалии не выявлены."


def clamp01(value):
    return max(0.0, min(1.0, float(value)))
