import re

from app.core.constants import (
    VERDICT_DEEPFAKE,
    VERDICT_ORIGINAL,
    VERDICT_UNKNOWN,
)
from app.core.models import ParsedResult


PERCENT_PATTERN = re.compile(r"(\d+(?:[\.,]\d+)?)\s*%")


def parse_model_result(raw_result):
    text = (raw_result or "").strip()
    upper_text = text.upper()

    if upper_text.startswith("ОШИБКА") or upper_text.startswith("ERROR"):
        return ParsedResult(VERDICT_UNKNOWN, None, is_error=True)

    confidence = parse_confidence(text)

    if "DEEPFAKE" in upper_text:
        return ParsedResult(VERDICT_DEEPFAKE, confidence)

    if "ORIGINAL" in upper_text:
        return ParsedResult(VERDICT_ORIGINAL, confidence)

    return ParsedResult(VERDICT_UNKNOWN, confidence)


def parse_confidence(text):
    match = PERCENT_PATTERN.search(text or "")
    if not match:
        return None

    value = match.group(1).replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None
