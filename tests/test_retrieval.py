import pytest

from adaptive_hitl_agent.retrieval import bm25_retriever


def test_bm25_returns_relevant_project_document() -> None:
    retriever = bm25_retriever.from_package_data()
    result = retriever.retrieve("what is the orion latency target?", top_k=1)[0]
    assert result.document.doc_id == "orion"
    assert result.document.answer == "120 milliseconds"
    assert result.score > 0


def test_bm25_rejects_invalid_top_k() -> None:
    retriever = bm25_retriever.from_package_data()
    with pytest.raises(ValueError):
        retriever.retrieve("orion", top_k=0)


def test_query_is_tokenized_once_per_retrieval(monkeypatch) -> None:
    import adaptive_hitl_agent.retrieval as module

    retriever = bm25_retriever.from_package_data()
    original = module.tokenize
    calls = []

    def counted(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(module, "tokenize", counted)
    query = "what is the orion latency target?"
    result = retriever.retrieve(query, top_k=1)
    assert calls == [query]
    assert result[0].document.doc_id == "orion"


@pytest.mark.parametrize(("text", "expected"), [
    ("The C# API_v2: 120ms!", ["c#", "api", "v2", "120ms"]),
    ("Straße Café 東京 ١٢ ²", ["strasse", "caf"]),
    ("İstanbul", ["i", "stanbul"]),
    ("--- \t", []),
])
def test_tokenization_preserves_ascii_rules(text, expected):
    from adaptive_hitl_agent.retrieval import tokenize

    assert tokenize(text) == expected


def test_repeated_terms_affect_frequency_without_inflating_document_frequency():
    from adaptive_hitl_agent.retrieval import document_record

    retriever = bm25_retriever([
        document_record("a", "", "echo echo echo", "first"),
        document_record("b", "", "echo other other", "second"),
    ])
    # equal document lengths isolate term frequency; echo occurs in both documents.
    results = retriever.retrieve("echo")
    assert [item.document.doc_id for item in results] == ["a", "b"]
    idf = 0.1823215567939546  # log(1 + 0.5 / 2.5)
    assert results[0].score == pytest.approx(idf * 3 * 2.5 / 4.5)
    assert results[1].score == pytest.approx(idf)
