from __future__ import annotations

from typing import Protocol as protocol, Sequence as sequence

from .retrieval import retrieved_document
from .types import model_answer


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
            from transformers import AutoModelForCausalLM as auto_model_for_causal_lm, AutoTokenizer as auto_tokenizer
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
            f"[{item.document.title}] {item.document.text}"
            for item in documents
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
