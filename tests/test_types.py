import pytest

from adaptive_hitl_agent.types import Task, is_correct, normalize_answer


@pytest.mark.parametrize(("answer", "reference", "expected"), [
    ("-5", "5", False), (".5", "5", False),
    ("-5", "-5", True), ("Paris.", "paris", True),
    ("0°C", "0c", True), ("", "", False),
])
def test_short_answer_scoring_preserves_numeric_meaning(answer, reference, expected) -> None:
    assert is_correct(answer, (reference,)) is expected


def test_task_requires_valid_answers_and_split() -> None:
    payload = {
        "id": "case", "question": "A question?", "answer": "yes",
        "split": "test", "category": "direct",
    }
    assert Task.from_dict(payload).reference_answer == "yes"
    for change in (
        {"acceptable_answers": "yes"}, {"acceptable_answers": []},
        {"split": "typo"}, {"id": ""}, {"category": ""},
    ):
        with pytest.raises(ValueError):
            Task.from_dict({**payload, **change})
    assert normalize_answer("  Albany, New York. ") == "albany new york"


@pytest.mark.parametrize("change", [
    {"answer": None}, {"answer": 42}, {"acceptable_answers": None},
    {"acceptable_answers": [None]}, {"id": None}, {"question": 123},
    {"category": None},
])
def test_task_loader_rejects_invalid_values_instead_of_stringifying(change) -> None:
    payload = dict(id="case", question="Question?", answer="yes", split="test", category="direct")
    with pytest.raises(ValueError):
        Task.from_dict({**payload, **change})


def test_task_constructor_rejects_string_answer_container() -> None:
    with pytest.raises(ValueError):
        Task("case", "Question?", "yes", "test", "direct")
