from app.core.constants import (
    AUDIO,
    MEDIA_LABELS,
    STATUS_COMPLETED,
    STATUS_ERROR,
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
)


def media_label(media_type):
    return MEDIA_LABELS.get(media_type, media_type)


def verdict_label(verdict):
    labels = {
        VERDICT_DEEPFAKE: "Подделка",
        VERDICT_ORIGINAL: "Оригинал",
    }
    return labels.get(verdict, "Не определено")


def status_label(status):
    labels = {
        STATUS_COMPLETED: "Готово",
        STATUS_ERROR: "Ошибка",
    }
    return labels.get(status, status)


def confidence_label(confidence):
    if confidence is None:
        return "-"
    return f"{confidence:.1f}%"


def file_size_label(file_size):
    if file_size is None:
        return "-"

    units = ("Б", "КБ", "МБ", "ГБ")
    value = float(file_size)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "Б":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024


def dialog_filter(media_type):
    if media_type == AUDIO:
        return "Аудио (*.wav *.mp3)"
    return "Видео (*.mp4 *.avi *.mov)"
