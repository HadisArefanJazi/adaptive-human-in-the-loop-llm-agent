"""Records, resource backends, and the routing environment."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass, field
from enum import IntEnum as int_enum
from importlib.resources import files
from typing import (
    Any as any_type,
    Iterable as iterable,
    Protocol as protocol,
    Sequence as sequence,
)


class action_type(int_enum):
    """actions available to the routing policy."""

    answer_directly = 0
    retrieve = 1
    use_tool = 2
    ask_human = 3


@dataclass(frozen=True)
class task_record:
    task_id: str
    question: str
    acceptable_answers: tuple[str, ...]
    split: str
    category: str

    def __post_init__(self) -> None:
        for name in ("task_id", "question", "category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task_record {name} must be a non-empty string")
        if (
            not isinstance(self.acceptable_answers, tuple)
            or not self.acceptable_answers
            or any(
                not isinstance(answer, str) or not answer.strip()
                for answer in self.acceptable_answers
            )
        ):
            raise ValueError("a task needs a tuple of non-empty reference strings")
        if self.split not in ("train", "test"):
            raise ValueError("task_record split must be 'train' or 'test'")

    @classmethod
    def from_dict(cls, payload: dict[str, any_type]) -> "task_record":
        answers = payload.get("acceptable_answers")
        if "acceptable_answers" not in payload:
            answers = [payload["answer"]]
        if not isinstance(answers, (list, tuple)):
            raise ValueError("acceptable_answers must be a list of answers")

        return cls(
            task_id=payload["id"],
            question=payload["question"],
            acceptable_answers=tuple(answers),
            split=payload["split"],
            category=payload["category"],
        )

    @property
    def reference_answer(self) -> str:
        return self.acceptable_answers[0]


@dataclass
class resource_usage:
    tokens: int = 0
    retrieval_calls: int = 0
    tool_calls: int = 0
    human_calls: int = 0
    latency_units: float = 0.0

    def copy(self) -> "resource_usage":
        return resource_usage(**self.as_dict())

    def delta(self, earlier: "resource_usage") -> "resource_usage":
        return resource_usage(
            tokens=self.tokens - earlier.tokens,
            retrieval_calls=self.retrieval_calls - earlier.retrieval_calls,
            tool_calls=self.tool_calls - earlier.tool_calls,
            human_calls=self.human_calls - earlier.human_calls,
            latency_units=self.latency_units - earlier.latency_units,
        )

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class model_answer:
    text: str
    confidence: float


@dataclass(frozen=True)
class observation_record:
    features: tuple[float, ...]
    feature_names: tuple[str, ...]
    direct_confidence: float
    math_signal: float
    ambiguity_signal: float
    has_retrieval: bool
    has_tool_output: bool
    step_index: int

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.features, strict=True))


@dataclass(frozen=True)
class step_result:
    observation: observation_record | None
    reward: float
    done: bool
    answer: str | None
    correct: bool | None
    action: action_type
    info: dict[str, any_type] = field(default_factory=dict)


def normalize_answer(text: str) -> str:
    """normalize short answers for deterministic benchmark scoring."""

    lowered = text.casefold().replace("°", "")
    cleaned = "".join(
        character if character.isalnum() or character in "_#.-" else " "
        for character in lowered
    )
    normalized = " ".join(cleaned.split())
    # keep leading signs and decimal points: -5 and .5 are not 5.
    return normalized.strip().rstrip(".")


def is_correct(answer: str, acceptable_answers: tuple[str, ...]) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and any(
        normalized == normalize_answer(candidate) for candidate in acceptable_answers
    )


stopwords = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "for",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


def tokenize(text: str) -> list[str]:
    cleaned = "".join(
        character
        if ("a" <= character <= "z" or "0" <= character <= "9" or character == "#")
        else " "
        for character in text.casefold()
    )
    return [token for token in cleaned.split() if token not in stopwords]


@dataclass(frozen=True)
class document_record:
    doc_id: str
    title: str
    text: str
    answer: str


@dataclass(frozen=True)
class retrieved_document:
    document: document_record
    score: float


class bm25_retriever:
    """compact local bm25 retriever used only when retrieve is selected."""

    def __init__(
        self,
        documents: iterable[document_record],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._tokens: list[list[str]] = []
        self._term_frequencies: list[dict[str, int]] = []

        total_length = 0
        for document in self.documents:
            tokens = tokenize(f"{document.title} {document.text}")
            self._tokens.append(tokens)
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self._term_frequencies.append(frequencies)
            total_length += len(tokens)
        self._average_length = total_length / len(self.documents)

        document_frequency: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        document_count = len(self.documents)
        self._idf = {
            term: math.log(1 + (document_count - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @classmethod
    def from_package_data(cls) -> "bm25_retriever":
        path = files("adaptive_hitl_agent.data").joinpath("knowledge_base.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(document_record(**item) for item in payload)

    def _score(self, query_tokens: list[str], index: int) -> float:
        frequencies = self._term_frequencies[index]
        length = len(self._tokens[index])
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / self._average_length
            )
            score += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def retrieve(self, query: str, top_k: int = 2) -> list[retrieved_document]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = tokenize(query)
        scored = [
            retrieved_document(
                document=document, score=self._score(query_tokens, index)
            )
            for index, document in enumerate(self.documents)
        ]
        scored.sort(key=lambda item: (-item.score, item.document.doc_id))
        return scored[:top_k]


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
        ast.Pow: lambda a, b: a**b,
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
                or not (
                    expression[len(prefix)].isalnum() or expression[len(prefix)] == "_"
                )
            ):
                expression = expression[len(prefix) :].lstrip()
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
            raise calculator_error(
                "expected a complete supported arithmetic expression"
            )
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


class language_model(protocol):
    """interface required by the routing environment."""

    def confidence(
        self,
        question: str,
        documents: sequence[retrieved_document] = (),
        tool_output: str | None = None,
    ) -> float: ...

    def answer(
        self,
        question: str,
        documents: sequence[retrieved_document] = (),
        tool_output: str | None = None,
    ) -> model_answer: ...


class rule_based_language_model:
    """small deterministic model used for reproducible experiments."""

    _memory = {
        "what is the capital of france?": "paris",
        "who wrote hamlet?": "william shakespeare",
        "at what temperature does water freeze in celsius?": "0°C",
        "what is the largest planet in our solar system?": "jupiter",
        "which planet is known as the red planet?": "mars",
        "what is the capital of japan?": "tokyo",
        "what is the chemical symbol for gold?": "Au",
        "at what temperature does water boil in celsius?": "100°C",
    }

    def confidence(
        self,
        question: str,
        documents: sequence[retrieved_document] = (),
        tool_output: str | None = None,
    ) -> float:
        if tool_output and tool_output.casefold() != "tool_error":
            return 0.99
        if documents and documents[0].score > 0:
            return min(0.98, 0.72 + documents[0].score / 20)
        if question.casefold().strip() in self._memory:
            return 0.96
        return 0.08

    def answer(
        self,
        question: str,
        documents: sequence[retrieved_document] = (),
        tool_output: str | None = None,
    ) -> model_answer:
        if tool_output and tool_output.casefold() != "tool_error":
            text = tool_output
        elif documents and documents[0].score > 0:
            text = documents[0].document.answer
        else:
            text = self._memory.get(
                question.casefold().strip(),
                "i don't know",
            )

        return model_answer(
            text=text,
            confidence=self.confidence(question, documents, tool_output),
        )


class hugging_face_language_model:
    """optional adapter for a local hugging face causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 48,
        device: str = "cpu",
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        try:
            from transformers import (
                AutoModelForCausalLM as auto_model_for_causal_lm,
                AutoTokenizer as auto_tokenizer,
            )
        except ImportError as error:
            raise ImportError(
                'install hugging face support with: pip install -e ".[hf]"'
            ) from error

        self.tokenizer = auto_tokenizer.from_pretrained(model_name)
        self.model = auto_model_for_causal_lm.from_pretrained(model_name).to(device)
        self.model.eval()

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device

    def confidence(
        self,
        question: str,
        documents: sequence[retrieved_document] = (),
        tool_output: str | None = None,
    ) -> float:
        if tool_output and tool_output.casefold() != "tool_error":
            return 0.9
        if documents and documents[0].score > 0:
            return min(0.85, 0.55 + documents[0].score / 20)
        return 0.35

    def answer(
        self,
        question: str,
        documents: sequence[retrieved_document] = (),
        tool_output: str | None = None,
    ) -> model_answer:
        if tool_output and tool_output.casefold() == "tool_error":
            tool_output = None

        evidence = "\n".join(
            f"[{item.document.title}] {item.document.text}" for item in documents
        )

        prompt = (
            "answer with only a short final answer. "
            "use the provided evidence or tool result when available. "
            "if unknown, say 'i don't know'.\n\n"
            f"evidence:\n{evidence or 'none'}\n"
            f"tool result: {tool_output or 'none'}\n"
            f"question: {question}\n"
            "answer:"
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        output = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = output[
            0,
            encoded["input_ids"].shape[1] :,
        ]

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        return model_answer(
            text=text or "i don't know",
            confidence=self.confidence(
                question,
                documents,
                tool_output,
            ),
        )


class human_reviewer(protocol):
    """backend called when the router requests a human answer."""

    def answer(self, task: task_record) -> str: ...


class oracle_human_reviewer:
    """idealized benchmark reviewer; reads the reference answer, not the question."""

    def answer(self, task: task_record) -> str:
        return task.reference_answer


feature_names = (
    "direct_confidence",
    "math_signal",
    "ambiguity_signal",
    "query_length",
    "has_retrieval",
    "has_tool_output",
    "step_progress",
)


@dataclass(frozen=True)
class reward_config_type:
    success_reward: float = 1.0
    failure_penalty: float = -0.25
    token_cost: float = 0.002
    retrieval_cost: float = 0.08
    tool_cost: float = 0.04
    human_cost: float = 0.45
    latency_cost: float = 0.02

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith("_cost") and value < 0:
                raise ValueError(f"{name} cannot be negative")


class assistance_environment:
    """a routing episode: gather evidence, then answer or request review.

    each resource action can run once. the last step is reserved for an answer
    or human review, so even an empty retrieval result cannot strand an episode.
    """

    _ambiguity_terms = (
        "ambiguous",
        "approve",
        "choose for me",
        "missing",
        "no criteria",
        "no preference",
        "personal judgment",
        "sensitive",
        "should i",
        "which one should",
    )

    def __init__(
        self,
        task: task_record,
        retriever: bm25_retriever,
        language_model: language_model,
        calculator: safe_calculator,
        human_reviewer: human_reviewer,
        reward_config: reward_config_type | None = None,
        max_steps: int = 3,
        retrieval_top_k: int = 2,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")

        self.task = task
        self.retriever = retriever
        self.language_model = language_model
        self.calculator = calculator
        self.human_reviewer = human_reviewer
        self.reward_config = reward_config or reward_config_type()
        self.max_steps = max_steps
        self.retrieval_top_k = retrieval_top_k

        self.documents: list[retrieved_document] = []
        self.tool_output: str | None = None
        self.usage = resource_usage()
        self.steps = 0
        self.done = False
        self.answer: str | None = None
        self.action_history: list[action_type] = []

    def observe(self) -> observation_record:
        direct_confidence = self.language_model.confidence(
            self.task.question,
            self.documents,
            self.tool_output,
        )
        lowered = self.task.question.casefold()
        # whitespace is allowed between the digits and a subtraction sign.
        compact = "".join(character for character in lowered if not character.isspace())
        has_subtraction = any(
            compact[index] == "-"
            and compact[index - 1].isdecimal()
            and compact[index + 1].isdecimal()
            for index in range(1, len(compact) - 1)
        )
        math_signal = float(
            any(character.isdecimal() for character in lowered)
            and (any(symbol in lowered for symbol in "+*/%") or has_subtraction)
        )
        ambiguity_signal = float(any(term in lowered for term in self._ambiguity_terms))
        query_length = min(1.0, len(self.task.question.split()) / 20.0)
        step_progress = self.steps / self.max_steps
        features = (
            direct_confidence,
            math_signal,
            ambiguity_signal,
            query_length,
            float(bool(self.documents)),
            float(self.tool_output is not None),
            step_progress,
        )
        return observation_record(
            features=features,
            feature_names=feature_names,
            direct_confidence=direct_confidence,
            math_signal=math_signal,
            ambiguity_signal=ambiguity_signal,
            has_retrieval=bool(self.documents),
            has_tool_output=self.tool_output is not None,
            step_index=self.steps,
        )

    def available_actions(self) -> tuple[action_type, ...]:
        if self.done:
            return ()

        actions = [action_type.answer_directly, action_type.ask_human]
        if self.steps < self.max_steps - 1:
            if self.usage.retrieval_calls == 0:
                actions.append(action_type.retrieve)
            if self.usage.tool_calls == 0:
                actions.append(action_type.use_tool)
        return tuple(sorted(actions, key=int))

    def step(self, action: action_type) -> step_result:
        if self.done:
            raise RuntimeError("the episode has already terminated")
        action = action_type(action)
        if action not in self.available_actions():
            raise ValueError(
                f"action_type {action.name} is not available in the current state"
            )

        before = self.usage.copy()
        self.steps += 1
        self.action_history.append(action)
        info: dict[str, object] = {}

        if action is action_type.retrieve:
            self.documents = self.retriever.retrieve(
                self.task.question,
                top_k=self.retrieval_top_k,
            )
            self.usage.tokens += sum(
                len(item.document.text.split()) for item in self.documents
            )
            self.usage.retrieval_calls += 1
            self.usage.latency_units += 1.0
            info["documents"] = [item.document.doc_id for item in self.documents]

        elif action is action_type.use_tool:
            try:
                self.tool_output = self.calculator.calculate(self.task.question)
            except calculator_error as error:
                self.tool_output = "tool_error"
                info["tool_error"] = str(error)
            self.usage.tokens += 4
            self.usage.tool_calls += 1
            self.usage.latency_units += 0.5
            info["tool_output"] = self.tool_output

        elif action is action_type.ask_human:
            self.answer = self.human_reviewer.answer(self.task)
            self.usage.tokens += 5
            self.usage.human_calls += 1
            self.usage.latency_units += 5.0
            self.done = True

        elif action is action_type.answer_directly:
            model_answer = self.language_model.answer(
                self.task.question,
                self.documents,
                self.tool_output,
            )
            self.answer = model_answer.text
            self.usage.tokens += (
                len(self.task.question.split())
                + max(1, len(model_answer.text.split()))
                + 2
            )
            self.usage.latency_units += 0.5
            info["model_confidence"] = model_answer.confidence
            self.done = True

        if not self.done and self.steps >= self.max_steps:
            raise RuntimeError(
                "the maximum step count was reached without a terminal action"
            )

        delta = self.usage.delta(before)
        reward = -self._resource_penalty(delta)
        correct: bool | None = None
        if self.done:
            assert self.answer is not None
            correct = is_correct(self.answer, self.task.acceptable_answers)
            reward += (
                self.reward_config.success_reward
                if correct
                else self.reward_config.failure_penalty
            )

        return step_result(
            observation=None if self.done else self.observe(),
            reward=reward,
            done=self.done,
            answer=self.answer,
            correct=correct,
            action=action,
            info=info,
        )

    def _resource_penalty(self, usage: resource_usage) -> float:
        config = self.reward_config
        return (
            config.token_cost * usage.tokens
            + config.retrieval_cost * usage.retrieval_calls
            + config.tool_cost * usage.tool_calls
            + config.human_cost * usage.human_calls
            + config.latency_cost * usage.latency_units
        )

    @property
    def total_resource_penalty(self) -> float:
        return self._resource_penalty(self.usage)
