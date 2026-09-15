from __future__ import annotations

import ast
import math


class calculator_error(ValueError):
    pass


class safe_calculator:
    """evaluate bounded arithmetic expressions without executing python code."""

    _binary_operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }
    _unary_operators = {ast.UAdd: lambda value: +value, ast.USub: lambda value: -value}
    _max_input_length = 1024
    _max_nodes = 128
    _max_depth = 32
    _max_magnitude = 1e12

    def extract_expression(self, question: str) -> str:
        if len(question) > self._max_input_length:
            raise calculator_error("input is too long")
        # accept a complete expression, optionally introduced by a supported
        # question prefix. extracting a numeric substring can silently change
        # the meaning of scientific notation, grouped numbers, or functions.
        expression = question.strip()
        # match case-insensitively, including unicode i/s variants, while
        # preserving character positions for slicing the expression.
        prefix_text = "".join(
            "i" if character in "İı" else "s" if character == "ſ" else character.lower()
            for character in expression[:9]
        )
        for prefix in ("calculate", "compute", "what is"):
            if prefix_text.startswith(prefix) and (
                len(expression) == len(prefix)
                or not (expression[len(prefix)].isalnum() or expression[len(prefix)] == "_")
            ):
                expression = expression[len(prefix):].lstrip()
                break
        expression = expression.rstrip("?").strip().rstrip(".").strip()
        if (
            not expression
            or not all(
                character.isdecimal() or character.isspace() or character in "()+-*/%."
                for character in expression
            )
            or not any(character.isdigit() for character in expression)
            or not any(symbol in expression for symbol in "+-*/%")
        ):
            raise calculator_error("expected a complete supported arithmetic expression")
        return expression

    def calculate(self, question: str) -> str:
        expression = self.extract_expression(question)
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > self._max_nodes:
                raise calculator_error("expression has too many operations")
            value = self._evaluate(tree.body)
        except calculator_error:
            raise
        except (SyntaxError, ArithmeticError, RecursionError, ValueError) as error:
            raise calculator_error(str(error)) from error
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else str(value)
        return str(value)

    def _checked(self, value: int | float) -> int | float:
        if type(value) not in {int, float}:
            raise calculator_error("only real numbers are supported")
        if isinstance(value, float) and not math.isfinite(value):
            raise calculator_error("result must be finite")
        if abs(value) > self._max_magnitude:
            raise calculator_error("value exceeds the configured magnitude limit")
        return value

    def _evaluate(self, node: ast.AST, depth: int = 0) -> int | float:
        if depth > self._max_depth:
            raise calculator_error("expression is nested too deeply")
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return self._checked(node.value)
        if isinstance(node, ast.UnaryOp):
            function = self._unary_operators.get(type(node.op))
            if function is None:
                raise calculator_error("unsupported unary operator")
            return self._checked(function(self._evaluate(node.operand, depth + 1)))
        if isinstance(node, ast.BinOp):
            function = self._binary_operators.get(type(node.op))
            if function is None:
                raise calculator_error("unsupported binary operator")
            left = self._evaluate(node.left, depth + 1)
            right = self._evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise calculator_error("exponent exceeds the configured limit")
            # check each intermediate value, not just the final result.
            return self._checked(function(left, right))
        raise calculator_error(f"unsupported expression type: {type(node).__name__}")
