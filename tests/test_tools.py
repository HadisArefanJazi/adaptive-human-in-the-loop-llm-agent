import pytest

from adaptive_hitl_agent.tools import CalculatorError, SafeCalculator


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Calculate 7 * 9.", "63"),
        ("What is (18 / 3) + 4?", "10"),
        ("Calculate 6 ** 3.", "216"),
    ],
)
def test_calculator_handles_supported_arithmetic(question: str, expected: str) -> None:
    assert SafeCalculator().calculate(question) == expected


def test_calculator_rejects_code_and_large_exponents() -> None:
    calculator = SafeCalculator()
    with pytest.raises(CalculatorError):
        calculator.calculate("Run __import__('os').system('id')")
    with pytest.raises(CalculatorError):
        calculator.calculate("Calculate 2 ** 99")
