from __future__ import annotations

import ast
import operator
import re


class CalculatorError(ValueError):
    pass


class SafeCalculator:
    """Evaluate arithmetic expressions without eval or arbitrary code execution."""

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
    _candidate_pattern = re.compile(r"[\d\s()+\-*/%.]+")

    def extract_expression(self, question: str) -> str:
        candidates = []
        for match in self._candidate_pattern.findall(question):
            candidate = match.strip().rstrip(".")
            has_number = any(character.isdigit() for character in candidate)
            has_operator = any(symbol in candidate for symbol in "+-*/%")
            if has_number and has_operator:
                candidates.append(candidate)

        if not candidates:
            raise CalculatorError("No arithmetic expression found")
        return max(candidates, key=len)

    def calculate(self, question: str) -> str:
        expression = self.extract_expression(question)
        try:
            tree = ast.parse(expression, mode="eval")
            value = self._evaluate(tree.body)
        except (SyntaxError, ZeroDivisionError, OverflowError) as error:
            raise CalculatorError(str(error)) from error
        if abs(float(value)) > 1e12:
            raise CalculatorError("Result exceeds the configured magnitude limit")
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, float):
            return f"{value:.8g}"
        return str(value)

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return node.value

        if isinstance(node, ast.UnaryOp):
            operator_function = self._unary_operators.get(type(node.op))
            if operator_function is None:
                raise CalculatorError("Unsupported unary operator")
            value = self._evaluate(node.operand)
            return operator_function(value)

        if isinstance(node, ast.BinOp):
            operator_function = self._binary_operators.get(type(node.op))
            if operator_function is None:
                raise CalculatorError("Unsupported binary operator")
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 10:
                raise CalculatorError("Exponent exceeds the configured limit")
            return operator_function(left, right)

        expression_type = type(node).__name__
        raise CalculatorError(f"Unsupported expression type: {expression_type}")
