import re

from app.models.intent import RobotIntent, build_normalized_intent, clean_label


_SPACES = re.compile(r"\s+")
_TRAILING_PUNCTUATION = " \t\r\n.,!?;:"
_COMMAND_PATTERNS = (
    re.compile(r"^(please\s+)?(build|make|create|construct|assemble)\s+(me\s+)?(a|an|the|some)?\s*", re.I),
    re.compile(r"^(i\s+want|i'd\s+like|i\s+would\s+like)\s+(to\s+)?(build|make|create|construct|assemble)?\s*(a|an|the|some)?\s*", re.I),
)


def _clean_user_text(text: str) -> str:
    cleaned = _SPACES.sub(" ", text.strip()).strip(_TRAILING_PUNCTUATION)
    if not cleaned:
        raise ValueError("text must not be empty")
    return cleaned


def extract_label_from_text(text: str) -> str:
    cleaned = _clean_user_text(text)
    label_source = cleaned
    for pattern in _COMMAND_PATTERNS:
        label_source = pattern.sub("", label_source, count=1).strip(_TRAILING_PUNCTUATION)
        if label_source != cleaned:
            break
    label = clean_label(label_source)
    if not label:
        raise ValueError("could not infer a build target from the text")
    return label


def normalize_text_intent(text: str) -> RobotIntent:
    cleaned = _clean_user_text(text)
    label = extract_label_from_text(cleaned)
    lower = cleaned.lower()
    if lower.startswith(("build", "make", "create", "construct", "assemble", "please ")):
        normalized_intent = cleaned[0].upper() + cleaned[1:]
    else:
        normalized_intent = build_normalized_intent(label)

    return RobotIntent(
        input_type="text",
        verified_label=label,
        normalized_intent=normalized_intent,
        human_verified=True,
    )
