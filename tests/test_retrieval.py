import pytest

from adaptive_hitl_agent.retrieval import BM25Retriever


def test_bm25_returns_relevant_project_document() -> None:
    retriever = BM25Retriever.from_package_data()
    result = retriever.retrieve("What is the Orion latency target?", top_k=1)[0]
    assert result.document.doc_id == "orion"
    assert result.document.answer == "120 milliseconds"
    assert result.score > 0


def test_bm25_rejects_invalid_top_k() -> None:
    retriever = BM25Retriever.from_package_data()
    with pytest.raises(ValueError):
        retriever.retrieve("Orion", top_k=0)


def test_query_is_tokenized_once_per_retrieval(monkeypatch) -> None:
    import adaptive_hitl_agent.retrieval as module

    retriever = BM25Retriever.from_package_data()
    original = module.tokenize
    calls = []

    def counted(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(module, "tokenize", counted)
    query = "What is the Orion latency target?"
    result = retriever.retrieve(query, top_k=1)
    assert calls == [query]
    assert result[0].document.doc_id == "orion"
