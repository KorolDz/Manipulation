import hashlib
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from app.core.constants import EVIDENCE_FRAMES_DIR, PROJECT_ROOT


VIDEO_WEIGHTS_PATH = os.path.join("weights", "efficient_weights.weights.h5")
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

_VIDEO_MODEL = None
_FACE_CASCADE = None
_VIDEO_MODEL_LOCK = threading.Lock()
_FACE_CASCADE_LOCK = threading.Lock()

MAX_ANALYZED_FACES = 30
SCORE_VARIANCE_THRESHOLD = 0.06
FACE_MOTION_THRESHOLD = 0.12
LIP_MOTION_THRESHOLD = 0.12
ARTIFACT_SCORE_THRESHOLD = 0.55
EVIDENCE_FRAME_MAX_WIDTH = 720
EVIDENCE_FRAME_JPEG_QUALITY = 88


@dataclass(frozen=True)
class VideoAnalysisOutput:
    raw_result: str
    technical_info: dict[str, object] = field(default_factory=dict)


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
    return analyze_video_with_details(video_path).raw_result


def analyze_video_with_details(video_path):
    try:
        model = get_video_model()
        face_cascade = get_face_cascade()
    except Exception as e:
        return VideoAnalysisOutput(f"Ошибка модели: {e}")

    cap = cv2.VideoCapture(video_path)
    predictions = []
    face_motions = []
    lip_motions = []
    artifact_scores = []
    frames_analyzed = 0
    frame_index = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    best_evidence_candidate = None
    previous_box = None
    previous_lower_face = None
    previous_score = None
    
    while cap.isOpened() and len(predictions) < MAX_ANALYZED_FACES:
        ret, frame = cap.read()
        if not ret:
            break
        frames_analyzed += 1
        frame_index += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            continue

        box = select_main_face(faces)
        x, y, w, h = clamp_box(box, frame.shape)
        if w <= 0 or h <= 0:
            continue

        face_img = frame[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]
        if face_img.size == 0 or face_gray.size == 0:
            continue

        face_tensor = prepare_face_tensor(face_img)
        res = model.predict(face_tensor, verbose=0)
        score = float(res[0][0])
        predictions.append(score)

        normalized_box = normalize_box((x, y, w, h), frame.shape)
        face_motion_score = 0.0
        if previous_box is not None:
            face_motion_score = face_box_motion(previous_box, normalized_box)
            face_motions.append(face_motion_score)
        previous_box = normalized_box

        lower_face = prepare_lower_face(gray, (x, y, w, h))
        lip_motion_score = 0.0
        if previous_lower_face is not None and lower_face is not None:
            lip_motion_score = frame_difference(previous_lower_face, lower_face)
            lip_motions.append(lip_motion_score)
        if lower_face is not None:
            previous_lower_face = lower_face

        artifact_score = estimate_visual_artifact_score(face_gray)
        artifact_scores.append(artifact_score)

        score_delta = abs(score - previous_score) if previous_score is not None else 0.0
        previous_score = score
        evidence_candidate = build_evidence_candidate(
            frame=frame,
            box=(x, y, w, h),
            frame_index=frame_index,
            fps=fps,
            score_delta=score_delta,
            face_motion_score=face_motion_score,
            lip_motion_score=lip_motion_score,
            artifact_score=artifact_score,
        )
        best_evidence_candidate = select_evidence_candidate(best_evidence_candidate, evidence_candidate)
            
    cap.release()

    raw_result = format_video_result(predictions)
    technical_info = build_video_technical_info(
        frames_analyzed=frames_analyzed,
        predictions=predictions,
        face_motions=face_motions,
        lip_motions=lip_motions,
        artifact_scores=artifact_scores,
    )
    if best_evidence_candidate is not None:
        try:
            technical_info.update(
                save_evidence_frame(
                    best_evidence_candidate,
                    technical_info.get("video_issue_categories", []),
                )
            )
        except Exception as exc:
            technical_info["video_evidence_frame_warning"] = f"Не удалось сохранить кадр проверки: {exc}"

    return VideoAnalysisOutput(raw_result, technical_info)


def format_video_result(predictions):
    if not predictions:
        return "Лица не найдены"

    avg_score = float(np.mean(predictions))

    if avg_score < 0.5:
        return f"DEEPFAKE ({(1 - avg_score) * 100:.1f}%)"
    return f"ORIGINAL ({avg_score * 100:.1f}%)"


def build_video_technical_info(frames_analyzed, predictions, face_motions, lip_motions, artifact_scores):
    score_variance = float(np.var(predictions)) if len(predictions) > 1 else 0.0
    face_motion_score = max_or_zero(face_motions)
    lip_motion_score = max_or_zero(lip_motions)
    artifact_score = max_or_zero(artifact_scores)
    issue_details = interpret_video_metrics(
        faces_analyzed=len(predictions),
        score_variance=score_variance,
        face_motion_score=face_motion_score,
        lip_motion_score=lip_motion_score,
        artifact_score=artifact_score,
    )

    return {
        "video_frames_analyzed": frames_analyzed,
        "video_faces_analyzed": len(predictions),
        "video_score_variance": round(score_variance, 4),
        "video_face_motion_score": round(face_motion_score, 4),
        "video_lip_motion_score": round(lip_motion_score, 4),
        "video_artifact_score": round(artifact_score, 4),
        **issue_details,
    }


def build_evidence_candidate(
    frame,
    box,
    frame_index,
    fps,
    score_delta,
    face_motion_score,
    lip_motion_score,
    artifact_score,
):
    return {
        "frame": frame.copy(),
        "box": box,
        "frame_index": frame_index,
        "frame_time": frame_time_seconds(frame_index, fps),
        "combined_score": evidence_combined_score(
            score_delta=score_delta,
            face_motion_score=face_motion_score,
            lip_motion_score=lip_motion_score,
            artifact_score=artifact_score,
        ),
    }


def select_evidence_candidate(current_candidate, new_candidate):
    if current_candidate is None:
        return new_candidate
    if new_candidate["combined_score"] > current_candidate["combined_score"]:
        return new_candidate
    return current_candidate


def evidence_combined_score(score_delta, face_motion_score, lip_motion_score, artifact_score):
    return max(
        clamp01(score_delta / 0.5),
        clamp01(face_motion_score / FACE_MOTION_THRESHOLD),
        clamp01(lip_motion_score / LIP_MOTION_THRESHOLD),
        clamp01(artifact_score),
    )


def save_evidence_frame(candidate, issue_categories):
    EVIDENCE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    annotated_frame = annotate_evidence_frame(
        candidate["frame"],
        candidate["box"],
        has_issues=bool(issue_categories),
    )
    resized_frame = resize_evidence_frame(annotated_frame)
    success, encoded = cv2.imencode(
        ".jpg",
        resized_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), EVIDENCE_FRAME_JPEG_QUALITY],
    )
    if not success:
        raise ValueError("OpenCV не смог закодировать JPEG")

    payload = encoded.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    output_path = EVIDENCE_FRAMES_DIR / f"{digest[:20]}.jpg"
    if not output_path.exists():
        output_path.write_bytes(payload)

    return {
        "video_evidence_frame_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "video_evidence_frame_label": evidence_frame_label(issue_categories),
        "video_evidence_frame_index": candidate["frame_index"],
        "video_evidence_frame_time": candidate["frame_time"],
        "video_evidence_frame_sha256": digest,
    }


def annotate_evidence_frame(frame, box, has_issues):
    annotated = frame.copy()
    x, y, w, h = box
    color = (0, 0, 255) if has_issues else (0, 180, 0)
    thickness = max(2, int(min(frame.shape[:2]) / 160))
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
    return annotated


def resize_evidence_frame(frame):
    height, width = frame.shape[:2]
    if width <= EVIDENCE_FRAME_MAX_WIDTH:
        return frame
    scale = EVIDENCE_FRAME_MAX_WIDTH / float(width)
    new_size = (EVIDENCE_FRAME_MAX_WIDTH, max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def evidence_frame_label(issue_categories):
    categories = [str(category) for category in (issue_categories or []) if category]
    if categories:
        return "Кадр проверки: " + ", ".join(categories)
    return "Кадр проверки: лицо найдено, явные видео-аномалии не выявлены."


def frame_time_seconds(frame_index, fps):
    if not fps or fps <= 0:
        return None
    return round(max(frame_index - 1, 0) / fps, 2)


def interpret_video_metrics(
    faces_analyzed,
    score_variance,
    face_motion_score,
    lip_motion_score,
    artifact_score,
):
    findings = []
    categories = []

    if faces_analyzed <= 0:
        return {
            "video_issue_categories": ["лицо не найдено"],
            "video_issue_findings": ["В видео не удалось найти лицо для проверки подмены, губ и визуальных артефактов."],
        }

    if score_variance >= SCORE_VARIANCE_THRESHOLD or face_motion_score >= FACE_MOTION_THRESHOLD:
        categories.append("подмена лица")
        findings.append("Обнаружена нестабильность признаков лица между кадрами, возможна подмена лица.")

    if lip_motion_score >= LIP_MOTION_THRESHOLD:
        categories.append("движение губ")
        findings.append("Зафиксированы аномалии движения нижней части лица, возможна несогласованность губ.")

    if artifact_score >= ARTIFACT_SCORE_THRESHOLD:
        categories.append("визуальные артефакты")
        findings.append("Обнаружены визуальные артефакты в области лица: размытие, резкие переходы или потеря детализации.")

    if not findings:
        findings.append("Дополнительные видео-признаки явных аномалий лица, губ или артефактов не выявили.")

    return {
        "video_issue_categories": categories,
        "video_issue_findings": findings,
    }


def select_main_face(faces):
    return max(faces, key=lambda face: int(face[2]) * int(face[3]))


def clamp_box(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x, y, w, h = (int(value) for value in box)
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(0, min(w, frame_width - x))
    h = max(0, min(h, frame_height - y))
    return x, y, w, h


def prepare_face_tensor(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(face_img, axis=0)


def normalize_box(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x, y, w, h = box
    return (
        (x + w / 2) / max(frame_width, 1),
        (y + h / 2) / max(frame_height, 1),
        w / max(frame_width, 1),
        h / max(frame_height, 1),
    )


def face_box_motion(previous_box, current_box):
    center_motion = abs(current_box[0] - previous_box[0]) + abs(current_box[1] - previous_box[1])
    size_motion = abs(current_box[2] - previous_box[2]) + abs(current_box[3] - previous_box[3])
    return float(center_motion + size_motion * 0.5)


def prepare_lower_face(gray, box):
    x, y, w, h = box
    lower_y = y + int(h * 0.55)
    lower_face = gray[lower_y:y+h, x:x+w]
    if lower_face.size == 0:
        return None
    return cv2.resize(lower_face, (64, 32))


def frame_difference(previous_frame, current_frame):
    difference = cv2.absdiff(previous_frame, current_frame)
    return float(np.mean(difference) / 255.0)


def estimate_visual_artifact_score(face_gray):
    if face_gray.size == 0:
        return 0.0

    face_gray = cv2.resize(face_gray, (96, 96))
    blur_variance = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    blur_score = clamp01((80.0 - blur_variance) / 80.0)

    edges = cv2.Canny(face_gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    edge_score = clamp01((edge_density - 0.22) / 0.18)

    return max(blur_score, edge_score)


def max_or_zero(values):
    return float(max(values)) if values else 0.0


def clamp01(value):
    return max(0.0, min(1.0, float(value)))
