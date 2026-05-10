from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "app.sqlite3"
EVIDENCE_FRAMES_DIR = DATA_DIR / "evidence_frames"

AUDIO = "audio"
VIDEO = "video"

AUDIO_EXTENSIONS = {".wav", ".mp3"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

SUPPORTED_EXTENSIONS = {
    AUDIO: AUDIO_EXTENSIONS,
    VIDEO: VIDEO_EXTENSIONS,
}

MEDIA_LABELS = {
    AUDIO: "Аудио",
    VIDEO: "Видео",
}

STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"

VERDICT_DEEPFAKE = "deepfake"
VERDICT_ORIGINAL = "original"
VERDICT_UNKNOWN = "unknown"
