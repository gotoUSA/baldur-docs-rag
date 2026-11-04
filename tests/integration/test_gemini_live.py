"""Calls the real Gemini API. Each test spends a few requests of the free quota."""

import pytest

from baldur_docs_rag.config import ENV_FILE, load_settings
from baldur_docs_rag.embedding import GeminiEmbedder
from baldur_docs_rag.errors import EmbeddingQuotaError

pytestmark = pytest.mark.gemini


class _SkipOnQuota:
    """An exhausted quota says nothing about the code, so it is a visible skip."""

    def __init__(self, embedder):
        self._embedder = embedder

    def embed_documents(self, texts):
        try:
            return self._embedder.embed_documents(texts)
        except EmbeddingQuotaError as e:
            pytest.skip(f"Gemini quota exhausted: {e}")

    def embed_query(self, question):
        try:
            return self._embedder.embed_query(question)
        except EmbeddingQuotaError as e:
            pytest.skip(f"Gemini quota exhausted: {e}")


@pytest.fixture
def embedder():
    if not ENV_FILE.exists():
        pytest.skip(".env with GEMINI_API_KEY is not present")
    return _SkipOnQuota(GeminiEmbedder(load_settings()))


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_each_document_gets_its_own_normalized_embedding(embedder):
    vectors = embedder.embed_documents(
        [
            "title: Bulkhead | text: A bulkhead limits concurrent calls to one dependency.",
            "title: Packaging | text: Install the package with pip.",
        ]
    )

    assert len(vectors) == 2
    assert vectors[0] != vectors[1]
    assert cosine(vectors[0], vectors[0]) == pytest.approx(1.0, abs=1e-3)


def test_query_is_closer_to_the_related_document(embedder):
    related, unrelated = embedder.embed_documents(
        [
            "title: Bulkhead | text: A bulkhead limits concurrent calls to one dependency.",
            "title: License | text: This project is released under an open source license.",
        ]
    )
    query = embedder.embed_query(
        "How do I stop one slow dependency using every worker?"
    )

    assert cosine(query, related) > cosine(query, unrelated)
