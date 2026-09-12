import ast

import pytest

from adaptive_hitl_agent.tools import CalculatorError, SafeCalculator


@pytest.mark.parametrize(("question", "expected"), [
    ("Calculate 7 * 9.", "63"),
    ("What is (18 / 3) + 4?", "10"),
    ("Calculate 6 ** 3.", "216"),
    ("Calculate -5 + 2.", "-3"),
    ("Calculate 1 / 4.", "0.25"),
    ("Calculate 7 // 2.", "3"),
    ("Calculate 7 % 3.", "1"),
])
def test_calculator_handles_supported_arithmetic(question: str, expected: str) -> None:
    assert SafeCalculator().calculate(question) == expected


@pytest.mark.parametrize("question", [
    "Run __import__('os').system('id')",
    "Calculate 2 ** 99",
    "Calculate 1 / 0",
    "Calculate (-1) ** 0.5",
    "Calculate (2 + )",
    "Calculate " + "9" * 400 + "+1",
    "Calculate " + "1+" * 600 + "1",
    "Calculate " + "1+" * 100 + "1",
    "Calculate " + "-" * 40 + "1",
    "Calculate (1000000000000 * 1000000000000) / 1000000000000",
])
def test_calculator_reports_invalid_or_unbounded_inputs(question: str) -> None:
    with pytest.raises(CalculatorError):
        SafeCalculator().calculate(question)


@pytest.mark.parametrize("expression", ["True", "abs(-1)", "~1", "1 << 2"])
def test_ast_evaluator_rejects_non_arithmetic_nodes(expression: str) -> None:
    with pytest.raises(CalculatorError):
        SafeCalculator()._evaluate(ast.parse(expression, mode="eval").body)


def test_calculator_rejects_non_finite_values() -> None:
    with pytest.raises(CalculatorError):
        SafeCalculator()._checked(float("inf"))



@pytest.mark.parametrize("question", [
    "Calculate 1e3 + 2.",
    "Calculate 2 ^ 3 + 1.",
    "Calculate 1,000 + 2.",
    "Calculate sqrt(9) + 1.",
    "Calculate 2 + 3 and 4 + 5.",
])
def test_calculator_does_not_evaluate_fragments_of_unsupported_input(question) -> None:
    with pytest.raises(CalculatorError):
        SafeCalculator().calculate(question)


def test_calculator_accepts_a_complete_bare_expression() -> None:
    assert SafeCalculator().calculate(" (2 + 3) * 4 ") == "20"


@pytest.mark.parametrize(("question", "expected"), [
    ("Calculate 123456789 + 0.1", "123456789.1"),
    ("Calculate 1 / 3", "0.3333333333333333"),
])
def test_calculator_preserves_float_precision_in_output(question, expected) -> None:
    assert SafeCalculator().calculate(question) == expected
