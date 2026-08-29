from __future__ import annotations

from typing import Sequence

from .retrieval import RetrievedDocument
from .types import ModelAnswer


class LanguageModel:
    """Common interface used by the environment."""

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
    """Deterministic stand-in used to make experiments fast and reproducible."""

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
        if tool_output is not None:
            confidence = self.confidence(question, documents, tool_output)
            return ModelAnswer(tool_output, confidence)

        if documents and documents[0].score > 0:
            answer = documents[0].document.answer
            confidence = self.confidence(question, documents, tool_output)
            return ModelAnswer(answer, confidence)

        question_key = question.casefold().strip()
        answer = self._memory.get(question_key, "I don't know")
        confidence = self.confidence(question, documents, tool_output)
        return ModelAnswer(answer, confidence)


class HuggingFaceLanguageModel(LanguageModel):
    """Optional local Hugging Face causal-LM adapter.

    The benchmark defaults to the deterministic model above so CI does not need
    to download weights. Instantiating this class loads the requested model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 48,
        device: str = "cpu",
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Install project dependencies before using HuggingFaceLanguageModel"
            ) from error

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = device

    def confidence(
        self,
        question: str,
        documents: Sequence[RetrievedDocument] = (),
        tool_output: str | None = None,
    ) -> float:
        # Replace this transparent prior with a calibrated confidence model in
        # production. It avoids an extra, unaccounted generation during routing.
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
        evidence_lines = []
        for item in documents:
            line = f"[{item.document.title}] {item.document.text}"
            evidence_lines.append(line)
        evidence = "\n".join(evidence_lines)

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
        confidence = self.confidence(question, documents, tool_output)
        return ModelAnswer(text or "I don't know", confidence)
