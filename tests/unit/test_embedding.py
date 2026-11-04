"""The fakes here copy responses seen from the real API."""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from baldur_docs_rag.chunking import Chunk
from baldur_docs_rag.config import Settings
from baldur_docs_rag.embedding import GeminiEmbedder, format_document, format_query
from baldur_docs_rag.errors import EmbeddingError, EmbeddingQuotaError

DIM = 128


class FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def embedder_with(outcome):
    settings = Settings(_env_file=None, gemini_api_key="k", embedding_dimensions=DIM)
    models = FakeModels(outcome)
    return GeminiEmbedder(settings, client=SimpleNamespace(models=models)), models


def response(count, dimensions=DIM):
    return types.EmbedContentResponse(
        embeddings=[
            types.ContentEmbedding(values=[0.5] * dimensions) for _ in range(count)
        ]
    )


def test_each_text_is_sent_as_its_own_content():
    embedder, models = embedder_with(response(2))

    assert len(embedder.embed_documents(["one", "two"])) == 2

    sent = models.calls[0]["contents"]
    assert all(isinstance(item, types.Content) for item in sent)
    assert [item.parts[0].text for item in sent] == ["one", "two"]


def test_requested_dimensions_are_sent():
    embedder, models = embedder_with(response(1))

    embedder.embed_query("question")

    assert models.calls[0]["config"].output_dimensionality == DIM


def test_aggregated_single_embedding_for_many_texts_is_rejected():
    # A real call with a plain list of three strings returned one embedding.
    embedder, _ = embedder_with(response(1))

    with pytest.raises(EmbeddingError, match="Expected 3 embeddings, got 1"):
        embedder.embed_documents(["a", "b", "c"])


def test_wrong_dimension_is_rejected():
    embedder, _ = embedder_with(response(1, dimensions=DIM + 1))

    with pytest.raises(EmbeddingError, match="dimensions"):
        embedder.embed_documents(["a"])


def test_empty_response_is_rejected():
    embedder, _ = embedder_with(types.EmbedContentResponse(embeddings=None))

    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])


def test_quota_error_stops_as_a_quota_failure():
    # Shape and wording copied from the real 429 of a full ingest on the free tier.
    message = (
        "You exceeded your current quota. Embed content requests per day limit"
        " exceeded on Free Tier. Please retry in 17h13m59s."
    )
    error = errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message}},
    )
    embedder, models = embedder_with(error)

    with pytest.raises(EmbeddingQuotaError) as caught:
        embedder.embed_documents(["a"])

    assert caught.value.__cause__ is error
    assert "Please retry in 17h13m59s" in str(caught.value)
    assert len(models.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        # Shapes seen from the real API during a probe.
        errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}),
        errors.ServerError(
            504, {"error": {"code": 504, "status": "DEADLINE_EXCEEDED"}}
        ),
        httpx.ConnectTimeout("timed out"),
    ],
)
def test_other_failures_are_not_reported_as_quota(error):
    embedder, _ = embedder_with(error)

    with pytest.raises(EmbeddingError) as caught:
        embedder.embed_documents(["a"])

    assert not isinstance(caught.value, EmbeddingQuotaError)
    assert caught.value.__cause__ is error


def test_document_and_query_use_the_documented_prefixes():
    chunk = Chunk("docs/a.md", 0, "Bulkhead > Why", "body")

    assert format_document(chunk) == "title: Bulkhead > Why | text: body"
    assert format_query("what?") == "task: search result | query: what?"


def test_document_without_a_heading_uses_the_documented_placeholder():
    assert (
        format_document(Chunk("docs/a.md", 0, "", "body")) == "title: none | text: body"
    )
