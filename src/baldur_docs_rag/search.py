"""Find the stored chunks closest to a question."""

from dataclasses import dataclass
from typing import Protocol

from baldur_docs_rag.config import Settings
from baldur_docs_rag.errors import BaldurDocsRagError, CorpusNotReadyError
from baldur_docs_rag.store import SearchHit, Store


class QueryEmbedder(Protocol):
    def embed_query(self, question: str) -> list[float]: ...


@dataclass(frozen=True)
class SearchResult:
    source_commit: str
    hits: list[SearchHit]


def run_search(
    question: str,
    settings: Settings,
    embedder: QueryEmbedder,
    store: Store,
) -> SearchResult:
    if not question.strip():
        raise BaldurDocsRagError("The question is empty")

    state = store.read_state()
    if state is None or state.completed_at is None:
        raise CorpusNotReadyError(
            "No complete ingest found. Run the ingest command first"
        )
    if (
        state.embedding_model != settings.embedding_model
        or state.embedding_dimensions != settings.embedding_dimensions
    ):
        # Embeddings from different models or sizes cannot be compared.
        raise CorpusNotReadyError(
            f"Stored embeddings use {state.embedding_model}/{state.embedding_dimensions},"
            f" settings use {settings.embedding_model}/{settings.embedding_dimensions}."
            " Run the ingest command again"
        )

    hits = store.search(embedder.embed_query(question), settings.search_top_k)
    return SearchResult(source_commit=state.source_commit, hits=hits)
