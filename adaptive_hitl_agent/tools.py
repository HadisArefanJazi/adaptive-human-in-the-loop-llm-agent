from __future__ import annotations

import ast
import math
import operator
import re


class CalculatorError(ValueError):
    pass


class SafeCalculator:
    """Evaluate bounded arithmetic expressions without executing Python code."""

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operators = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    _expression_pattern = re.compile(r"[\d\s()+\-*/%.]+")
    _max_input_length = 1024
    _max_nodes = 128
    _max_depth = 32
    _max_magnitude = 1e12

    def extract_expression(self, question: str) -> str:
        if len(question) > self._max_input_length:
            raise CalculatorError("Input is too long")
        # Accept a complete expression, optionally introduced by a supported
        # question prefix. Extracting a numeric substring can silently change
        # the meaning of scientific notation, grouped numbers, or functions.
        expression = re.sub(
            r"^(?:calculate|compute|what is)\b\s*",
            "",
            question.strip(),
            flags=re.IGNORECASE,
        ).rstrip("?").strip().rstrip(".").strip()
        if (
            not self._expression_pattern.fullmatch(expression)
            or not any(character.isdigit() for character in expression)
            or not any(symbol in expression for symbol in "+-*/%")
        ):
            raise CalculatorError("Expected a complete supported arithmetic expression")
        return expression

    def calculate(self, question: str) -> str:
        expression = self.extract_expression(question)
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > self._max_nodes:
                raise CalculatorError("Expression has too many operations")
            value = self._evaluate(tree.body)
        except CalculatorError:
            raise
        except (SyntaxError, ArithmeticError, RecursionError, ValueError) as error:
            raise CalculatorError(str(error)) from error
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else f"{value:.8g}"
        return str(value)

    def _checked(self, value: int | float) -> int | float:
        if type(value) not in {int, float}:
            raise CalculatorError("Only real numbers are supported")
        if isinstance(value, float) and not math.isfinite(value):
            raise CalculatorError("Result must be finite")
        if abs(value) > self._max_magnitude:
            raise CalculatorError("Value exceeds the configured magnitude limit")
        return value

    def _evaluate(self, node: ast.AST, depth: int = 0) -> int | float:
        if depth > self._max_depth:
            raise CalculatorError("Expression is nested too deeply")
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return self._checked(node.value)
        if isinstance(node, ast.UnaryOp):
            function = self._unary_operators.get(type(node.op))
            if function is None:
                raise CalculatorError("Unsupported unary operator")
            return self._checked(function(self._evaluate(node.operand, depth + 1)))
        if isinstance(node, ast.BinOp):
            function = self._binary_operators.get(type(node.op))
            if function is None:
                raise CalculatorError("Unsupported binary operator")
            left = self._evaluate(node.left, depth + 1)
            right = self._evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise CalculatorError("Exponent exceeds the configured limit")
            # Check each intermediate value, not just the final result.
            return self._checked(function(left, right))
        raise CalculatorError(f"Unsupported expression type: {type(node).__name__}")
