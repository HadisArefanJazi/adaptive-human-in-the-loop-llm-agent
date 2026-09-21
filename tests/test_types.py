import pytest

from adaptive_hitl_agent.types import task_record, is_correct, normalize_answer


@pytest.mark.parametrize(("answer", "reference", "expected"), [
    ("-5", "5", False), (".5", "5", False),
    ("-5", "-5", True), ("Paris.", "paris", True),
    ("0°C", "0c", True), ("", "", False),
])
def test_short_answer_scoring_preserves_numeric_meaning(answer, reference, expected) -> None:
    assert is_correct(answer, (reference,)) is expected


def test_task_requires_valid_answers_and_split() -> None:
    payload = {
        "id": "case", "question": "a question?", "answer": "yes",
        "split": "test", "category": "direct",
    }
    assert task_record.from_dict(payload).reference_answer == "yes"
    for change in (
        {"acceptable_answers": "yes"}, {"acceptable_answers": []},
        {"split": "typo"}, {"id": ""}, {"category": ""},
    ):
        with pytest.raises(ValueError):
            task_record.from_dict({**payload, **change})
    assert normalize_answer("  Albany, New York. ") == "albany new york"


@pytest.mark.parametrize("change", [
    {"answer": None}, {"answer": 42}, {"acceptable_answers": None},
    {"acceptable_answers": [None]}, {"id": None}, {"question": 123},
    {"category": None},
])
def test_task_loader_rejects_invalid_values_instead_of_stringifying(change) -> None:
    payload = dict(id="case", question="question?", answer="yes", split="test", category="direct")
    with pytest.raises(ValueError):
        task_record.from_dict({**payload, **change})


def test_task_constructor_rejects_string_answer_container() -> None:
    with pytest.raises(ValueError):
        task_record("case", "question?", "yes", "test", "direct")


@pytest.mark.parametrize(("text", "expected"), [
    ("Café_東京 #C. -5 .5!", "café_東京 #c. -5 .5"),
    ("Straße / İ", "strasse i"),
    ("alpha,beta\t gamma", "alpha beta gamma"),
    ("² ١", "² ١"),
    ("...", ""),
])
def test_normalization_preserves_unicode_and_token_boundaries(text, expected):
    assert normalize_answer(text) == expected
