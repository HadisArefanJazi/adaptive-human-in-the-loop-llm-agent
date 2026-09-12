from adaptive_hitl_agent.llm import RuleBasedLanguageModel
from adaptive_hitl_agent.retrieval import BM25Retriever


def test_rule_based_model_handles_direct_tool_and_retrieval_context() -> None:
    model = RuleBasedLanguageModel()

    direct = model.answer("What is the capital of France?")
    assert direct.text == "Paris"
    assert direct.confidence == 0.96

    tool = model.answer("Calculate 2 + 2", tool_output="4")
    assert tool.text == "4"
    assert tool.confidence == 0.99

    documents = BM25Retriever.from_package_data().retrieve(
        "What is the Orion latency target?",
        top_k=1,
    )
    retrieved = model.answer("What is the Orion latency target?", documents)
    assert retrieved.text == "120 milliseconds"
    assert retrieved.confidence > 0.72


def test_tool_failure_does_not_override_valid_model_evidence() -> None:
    model = RuleBasedLanguageModel()
    assert model.answer("What is the capital of France?", tool_output="TOOL_ERROR").text == "Paris"
    documents = BM25Retriever.from_package_data().retrieve("What is the Orion latency target?")
    answer = model.answer("What is the Orion latency target?", documents, "TOOL_ERROR")
    assert answer.text == "120 milliseconds"


def test_huggingface_dependency_is_optional(monkeypatch) -> None:
    import sys
    import pytest
    from adaptive_hitl_agent.llm import HuggingFaceLanguageModel

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="hf"):
        HuggingFaceLanguageModel()
    with pytest.raises(ValueError):
        HuggingFaceLanguageModel(max_new_tokens=0)


def test_huggingface_adapter_with_local_fakes(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    import torch
    from adaptive_hitl_agent.llm import HuggingFaceLanguageModel

    prompts = []

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors):
            prompts.append(prompt)
            return Encoded(input_ids=torch.tensor([[1, 2]]))

        def decode(self, generated, skip_special_tokens):
            assert generated.tolist() == [3]
            return "Paris"

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda name: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda name: Model()),
    ))
    model = HuggingFaceLanguageModel(model_name="test-model")
    assert model.answer("A question?", tool_output="TOOL_ERROR").text == "Paris"
    assert "TOOL_ERROR" not in prompts[0]
    assert model.confidence("A question?", tool_output="4") == 0.9
    documents = BM25Retriever.from_package_data().retrieve("Orion latency")
    assert model.confidence("Orion latency", documents) > 0.55
