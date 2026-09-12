from __future__ import annotations

from typing import Sequence

from .retrieval import RetrievedDocument
from .types import ModelAnswer


class LanguageModel:
    """Common language-model interface used by the routing environment."""

    def confidence(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> float:
        raise NotImplementedError

    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> ModelAnswer:
        raise NotImplementedError


class RuleBasedLanguageModel(LanguageModel):
    """Deterministic stand-in that keeps the benchmark fast and reproducible."""

    _memory = {
        "what is the capital of france?": "Paris",
        "who wrote hamlet?": "William Shakespeare",
        "at what temperature does water freeze in celsius?": "0°C",
        "what is the largest planet in our solar system?": "Jupiter",
        "which planet is known as the red planet?": "Mars",
        "what is the capital of japan?": "Tokyo",
        "what is the chemical symbol for gold?": "Au",
        "at what temperature does water boil in celsius?": "100°C",
    }

    def confidence(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> float:
        if tool_output is not None and tool_output != "TOOL_ERROR":
            return 0.99
        if documents and documents[0].score > 0:
            return min(0.98, 0.72 + documents[0].score / 20)
        if question.casefold().strip() in self._memory:
            return 0.96
        return 0.08

    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> ModelAnswer:
        if tool_output is not None and tool_output != "TOOL_ERROR":
            return ModelAnswer(
                text=tool_output,
                confidence=self.confidence(question, documents, tool_output),
            )

        if documents and documents[0].score > 0:
            return ModelAnswer(
                text=documents[0].document.answer,
                confidence=self.confidence(question, documents, tool_output),
            )

        answer = self._memory.get(question.casefold().strip(), "I don't know")
        return ModelAnswer(
            text=answer,
            confidence=self.confidence(question, documents, tool_output),
        )


class HuggingFaceLanguageModel(LanguageModel):
    """Optional local Hugging Face causal-LM adapter.

    The benchmark defaults to RuleBasedLanguageModel so core installation and
    CI do not download model weights.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 48,
        device: str = "cpu",
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                'Install the optional Hugging Face dependencies with: '
                'pip install -e ".[hf]"'
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device

    def confidence(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> float:
        # This transparent prior avoids an extra generation during routing. A
        # real deployment should replace it with calibrated uncertainty.
        if tool_output is not None and tool_output != "TOOL_ERROR":
            return 0.9
        if documents and documents[0].score > 0:
            return min(0.85, 0.55 + documents[0].score / 20)
        return 0.35

    def answer(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> ModelAnswer:
        evidence = "\n".join(
            f"[{item.document.title}] {item.document.text}" for item in documents
        )
        if tool_output == "TOOL_ERROR":
            tool_output = None
        prompt = (
            "Answer with only a short final answer. If evidence or a tool result is "
            "provided, use it. If the answer is unknown, say 'I don't know'.\n\n"
            f"Evidence:\n{evidence or 'None'}\n"
            f"Tool result: {tool_output or 'None'}\n"
            f"Question: {question}\nAnswer:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return ModelAnswer(
            text=text or "I don't know",
            confidence=self.confidence(question, documents, tool_output),
        )
