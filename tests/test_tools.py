import ast

import pytest

from adaptive_hitl_agent.tools import calculator_error, safe_calculator


@pytest.mark.parametrize(("question", "expected"), [
    ("calculate 7 * 9.", "63"),
    ("what is (18 / 3) + 4?", "10"),
    ("calculate 6 ** 3.", "216"),
    ("calculate -5 + 2.", "-3"),
    ("calculate 1 / 4.", "0.25"),
    ("calculate 7 // 2.", "3"),
    ("calculate 7 % 3.", "1"),
])
def test_calculator_handles_supported_arithmetic(question: str, expected: str) -> None:
    assert safe_calculator().calculate(question) == expected


@pytest.mark.parametrize("question", [
    "run __import__('os').system('id')",
    "calculate 2 ** 99",
    "calculate 1 / 0",
    "calculate (-1) ** 0.5",
    "calculate (2 + )",
    "calculate " + "9" * 400 + "+1",
    "calculate " + "1+" * 600 + "1",
    "calculate " + "1+" * 100 + "1",
    "calculate " + "-" * 40 + "1",
    "calculate (1000000000000 * 1000000000000) / 1000000000000",
])
def test_calculator_reports_invalid_or_unbounded_inputs(question: str) -> None:
    with pytest.raises(calculator_error):
        safe_calculator().calculate(question)


@pytest.mark.parametrize("expression", ["True", "abs(-1)", "~1", "1 << 2"])
def test_ast_evaluator_rejects_non_arithmetic_nodes(expression: str) -> None:
    with pytest.raises(calculator_error):
        safe_calculator()._evaluate(ast.parse(expression, mode="eval").body)


def test_calculator_rejects_non_finite_values() -> None:
    with pytest.raises(calculator_error):
        safe_calculator()._checked(float("inf"))



@pytest.mark.parametrize("question", [
    "calculate 1e3 + 2.",
    "calculate 2 ^ 3 + 1.",
    "calculate 1,000 + 2.",
    "calculate sqrt(9) + 1.",
    "calculate 2 + 3 and 4 + 5.",
])
def test_calculator_does_not_evaluate_fragments_of_unsupported_input(question) -> None:
    with pytest.raises(calculator_error):
        safe_calculator().calculate(question)


def test_calculator_accepts_a_complete_bare_expression() -> None:
    assert safe_calculator().calculate(" (2 + 3) * 4 ") == "20"


@pytest.mark.parametrize(("question", "expected"), [
    ("calculate 123456789 + 0.1", "123456789.1"),
    ("calculate 1 / 3", "0.3333333333333333"),
])
def test_calculator_preserves_float_precision_in_output(question, expected) -> None:
    assert safe_calculator().calculate(question) == expected


@pytest.mark.parametrize(("question", "expected"), [
    ("cAlCuLaTe\t+5 + 2??", "7"),
    ("Compute(2 + 3).", "5"),
    ("WHAT İS 2 + 3?", "5"),
    ("What ıſ 2 + 3?", "5"),
    ("2\t-\t3", "-1"),
])
def test_calculator_prefixes_and_unary_plus(question, expected):
    assert safe_calculator().calculate(question) == expected


@pytest.mark.parametrize("question", [
    "calculate2 + 3", "compute_ 2 + 3", "what island 2 + 3",
    "what  is 2 + 3", "calculate ² + 3", "calculate", "",
])
def test_calculator_preserves_prefix_boundaries_and_character_rules(question):
    with pytest.raises(calculator_error):
        safe_calculator().extract_expression(question)


def test_expression_extraction_accepts_unicode_decimal_digits():
    assert safe_calculator().extract_expression("compute ١ + ٢") == "١ + ٢"
