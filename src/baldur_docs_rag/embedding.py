"""Turn text into embeddings with the Gemini API."""

import logging
from collections.abc import Sequence

import httpx
from google import genai
from google.genai import errors, types

from baldur_docs_rag.chunking import Chunk
from baldur_docs_rag.config import Settings
from baldur_docs_rag.errors import EmbeddingError, EmbeddingQuotaError

logger = logging.getLogger(__name__)

HTTP_TOO_MANY_REQUESTS = 429

# gemini-embedding-2 has no task_type field. The official docs ask for these
# text prefixes instead, and documents and queries use different ones.
DOCUMENT_TEMPLATE = "title: {title} | text: {text}"
QUERY_TEMPLATE = "task: search result | query: {text}"
NO_TITLE = "none"


def format_document(chunk: Chunk) -> str:
    return DOCUMENT_TEMPLATE.format(title=chunk.heading or NO_TITLE, text=chunk.content)


def format_query(question: str) -> str:
    return QUERY_TEMPLATE.format(text=question)


class GeminiEmbedder:
    def __init__(self, settings: Settings, client: genai.Client | None = None) -> None:
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._client = client or genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=settings.embedding_timeout_ms),
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._embed([format_query(question)])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Each text needs its own Content. A real call with a plain list of three
        # strings returned one aggregated embedding, not three.
        contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except errors.APIError as e:
            if e.code == HTTP_TOO_MANY_REQUESTS:
                logger.warning("embedding.quota_exhausted", extra={"status": e.status})
                raise EmbeddingQuotaError(f"Gemini quota reached: {e.message}") from e
            logger.warning(
                "embedding.request_failed", extra={"code": e.code, "status": e.status}
            )
            raise EmbeddingError(f"Gemini returned {e.code} {e.status}") from e
        except httpx.HTTPError as e:
            logger.warning(
                "embedding.request_failed", extra={"error": type(e).__name__}
            )
            raise EmbeddingError(f"Gemini request failed: {type(e).__name__}") from e

        return self._validated(response, expected_count=len(texts))

    def _validated(
        self,
        response: types.EmbedContentResponse,
        expected_count: int,
    ) -> list[list[float]]:
        embeddings = response.embeddings or []
        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} embeddings, got {len(embeddings)}"
            )

        vectors = [list(embedding.values or []) for embedding in embeddings]
        for vector in vectors:
            if len(vector) != self._dimensions:
                raise EmbeddingError(
                    f"Expected {self._dimensions} dimensions, got {len(vector)}"
                )
        return vectors
