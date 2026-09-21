from adaptive_hitl_agent.llm import rule_based_language_model
from adaptive_hitl_agent.retrieval import bm25_retriever


def test_rule_based_model_handles_direct_tool_and_retrieval_context() -> None:
    model = rule_based_language_model()

    direct = model.answer("what is the capital of france?")
    assert direct.text == "paris"
    assert direct.confidence == 0.96

    tool = model.answer("calculate 2 + 2", tool_output="4")
    assert tool.text == "4"
    assert tool.confidence == 0.99

    documents = bm25_retriever.from_package_data().retrieve(
        "what is the orion latency target?",
        top_k=1,
    )
    retrieved = model.answer("what is the orion latency target?", documents)
    assert retrieved.text == "120 milliseconds"
    assert retrieved.confidence > 0.72


def test_tool_failure_does_not_override_valid_model_evidence() -> None:
    model = rule_based_language_model()
    assert model.answer("what is the capital of france?", tool_output="tool_error").text == "paris"
    documents = bm25_retriever.from_package_data().retrieve("what is the orion latency target?")
    answer = model.answer("what is the orion latency target?", documents, "tool_error")
    assert answer.text == "120 milliseconds"


def test_huggingface_dependency_is_optional(monkeypatch) -> None:
    import sys
    import pytest
    from adaptive_hitl_agent.llm import hugging_face_language_model

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="hf"):
        hugging_face_language_model()
    with pytest.raises(ValueError):
        hugging_face_language_model(max_new_tokens=0)


def test_huggingface_adapter_with_local_fakes(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace as simple_namespace
    import torch
    from adaptive_hitl_agent.llm import hugging_face_language_model

    prompts = []

    class encoded_batch(dict):
        def to(self, device):
            return self

    class tokenizer_type:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors):
            prompts.append(prompt)
            return encoded_batch(input_ids=torch.tensor([[1, 2]]))

        def decode(self, generated, skip_special_tokens):
            assert generated.tolist() == [3]
            return "paris"

    class model_type:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setitem(sys.modules, "transformers", simple_namespace(
        AutoTokenizer=simple_namespace(from_pretrained=lambda name: tokenizer_type()),
        AutoModelForCausalLM=simple_namespace(from_pretrained=lambda name: model_type()),
    ))
    model = hugging_face_language_model(model_name="test-model")
    assert model.answer("a question?", tool_output="tool_error").text == "paris"
    assert "tool_error" not in prompts[0]
    assert model.confidence("a question?", tool_output="4") == 0.9
    documents = bm25_retriever.from_package_data().retrieve("orion latency")
    assert model.confidence("orion latency", documents) > 0.55
