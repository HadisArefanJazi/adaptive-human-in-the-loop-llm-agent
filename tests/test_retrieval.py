from adaptive_hitl_agent.retrieval import BM25Retriever


def test_bm25_returns_relevant_project_document() -> None:
    retriever = BM25Retriever.from_package_data()
    result = retriever.retrieve("What is the Orion latency target?", top_k=1)[0]
    assert result.document.doc_id == "orion"
    assert result.document.answer == "120 milliseconds"
    assert result.score > 0
