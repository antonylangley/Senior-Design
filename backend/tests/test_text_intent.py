from app.services.text_intent import extract_label_from_text, normalize_text_intent


def test_extract_label_from_common_build_phrase() -> None:
    assert extract_label_from_text("Build me a duck") == "duck"
    assert extract_label_from_text("Make a small house") == "small house"
    assert extract_label_from_text("Build a bridge.") == "bridge"


def test_normalize_text_intent_preserves_command_intent() -> None:
    intent = normalize_text_intent("Build me a duck")

    assert intent.input_type == "text"
    assert intent.verified_label == "duck"
    assert intent.normalized_intent == "Build me a duck"
    assert intent.human_verified is True


def test_normalize_text_intent_wraps_bare_object() -> None:
    intent = normalize_text_intent("duck")

    assert intent.verified_label == "duck"
    assert intent.normalized_intent == "Build a duck"
