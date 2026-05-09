from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    media_type: str | None = None
    message: str = ""
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedResult:
    verdict: str
    confidence: float | None
    is_error: bool = False


@dataclass(frozen=True)
class MediaInfo:
    file_path: str
    file_name: str
    media_type: str
    file_size: int | None
    extension: str
    duration: float | None = None
    technical_info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    file_path: str
    file_name: str
    media_type: str
    file_size: int | None
    status: str
    verdict: str
    confidence: float | None
    raw_result: str
    error_message: str | None = None
    duration: float | None = None
    technical_info: dict[str, object] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HistoryRecord:
    id: int
    created_at: str
    file_path: str
    file_name: str
    media_type: str
    file_size: int | None
    status: str
    verdict: str
    confidence: float | None
    raw_result: str
    error_message: str | None
    duration: float | None = None
    technical_info: dict[str, object] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
