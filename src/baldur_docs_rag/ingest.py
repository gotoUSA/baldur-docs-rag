"""Read the documents, embed their chunks and store them, resuming if stopped."""

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from baldur_docs_rag.chunking import Chunk, split_markdown
from baldur_docs_rag.config import Settings
from baldur_docs_rag.documents import load_documents, resolve_commit
from baldur_docs_rag.embedding import format_document
from baldur_docs_rag.store import Store

logger = logging.getLogger(__name__)

# Deliberately low, so the estimate errs towards too many tokens.
# A real sample of dense technical text measured about 3.2 characters per token.
ESTIMATED_CHARS_PER_TOKEN = 3
SECONDS_PER_MINUTE = 60


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestResult:
    source_commit: str
    total_chunks: int
    embedded_chunks: int


def build_chunks(settings: Settings, source_commit: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents(settings.baldur_repo_path, source_commit):
        chunks.extend(
            split_markdown(document.path, document.text, settings.chunk_max_chars)
        )
    return chunks


def corpus_fingerprint(
    settings: Settings, source_commit: str, chunks: Sequence[Chunk]
) -> str:
    """Hash everything that decides the stored rows, including the exact embedded text."""
    digest = hashlib.sha256()
    parts = [
        source_commit,
        settings.embedding_model,
        str(settings.embedding_dimensions),
    ]
    for part in parts:
        digest.update(part.encode("utf-8") + b"\0")
    for chunk in chunks:
        key = f"{chunk.source_path}\0{chunk.chunk_index}\0{format_document(chunk)}\0"
        digest.update(key.encode("utf-8"))
    return digest.hexdigest()


def pacing_delay_seconds(
    texts: Sequence[str],
    tokens_per_minute: int,
    texts_per_minute: int,
) -> float:
    """Return the wait that keeps both per-minute limits, whichever is stricter."""
    estimated_tokens = sum(len(text) for text in texts) / ESTIMATED_CHARS_PER_TOKEN
    return SECONDS_PER_MINUTE * max(
        estimated_tokens / tokens_per_minute,
        len(texts) / texts_per_minute,
    )


def run_ingest(
    settings: Settings,
    embedder: DocumentEmbedder,
    store: Store,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestResult:
    source_commit = resolve_commit(settings.baldur_repo_path, settings.source_ref)
    chunks = build_chunks(settings, source_commit)
    stored = store.prepare(
        fingerprint=corpus_fingerprint(settings, source_commit, chunks),
        source_commit=source_commit,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        expected_chunks=len(chunks),
    )
    pending = [c for c in chunks if (c.source_path, c.chunk_index) not in stored]
    logger.info(
        "ingest.run_started",
        extra={"total": len(chunks), "stored": len(stored), "pending": len(pending)},
    )

    done = 0
    size = settings.embedding_batch_size
    for start in range(0, len(pending), size):
        batch = pending[start : start + size]
        texts = [format_document(chunk) for chunk in batch]
        store.add_chunks(batch, embedder.embed_documents(texts))
        done += len(batch)
        logger.info(
            "ingest.progress_updated", extra={"done": done, "pending": len(pending)}
        )
        if done < len(pending):
            sleep(
                pacing_delay_seconds(
                    texts,
                    settings.embedding_tokens_per_minute,
                    settings.embedding_texts_per_minute,
                )
            )

    store.mark_complete()
    return IngestResult(
        source_commit=source_commit,
        total_chunks=len(chunks),
        embedded_chunks=done,
    )
