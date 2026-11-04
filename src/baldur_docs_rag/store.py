"""Store chunks and embeddings in PostgreSQL, and search them with pgvector."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Self

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

from baldur_docs_rag.chunking import Chunk
from baldur_docs_rag.errors import StoreError

CONNECT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class CorpusState:
    fingerprint: str
    source_commit: str
    embedding_model: str
    embedding_dimensions: int
    expected_chunks: int
    completed_at: datetime | None


@dataclass(frozen=True)
class SearchHit:
    source_path: str
    chunk_index: int
    heading: str
    content: str
    score: float


@contextmanager
def _store_errors(action: str) -> Iterator[None]:
    try:
        yield
    except psycopg.Error as e:
        raise StoreError(f"PostgreSQL failed to {action}: {type(e).__name__}") from e


class Store:
    def __init__(self, database_url: str) -> None:
        with _store_errors("connect"):
            self._conn = psycopg.connect(
                database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS
            )
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()
            register_vector(self._conn)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._conn.close()

    def read_state(self) -> CorpusState | None:
        with _store_errors("read the corpus state"):
            exists = self._conn.execute("SELECT to_regclass('corpus_state')").fetchone()
            if exists is None or exists[0] is None:
                self._conn.rollback()
                return None
            row = self._conn.execute(
                "SELECT fingerprint, source_commit, embedding_model, embedding_dimensions,"
                " expected_chunks, completed_at FROM corpus_state"
            ).fetchone()
            self._conn.rollback()
        return CorpusState(*row) if row else None

    def prepare(
        self,
        fingerprint: str,
        source_commit: str,
        embedding_model: str,
        embedding_dimensions: int,
        expected_chunks: int,
    ) -> set[tuple[str, int]]:
        """Return the stored chunk keys, resetting first if the corpus changed.

        Rows are only reusable for the exact same chunks, commit and model, so a
        stopped ingest can resume but different inputs never mix in one table.
        """
        state = self.read_state()
        with _store_errors("prepare the tables"):
            if state is None or state.fingerprint != fingerprint:
                self._recreate_tables(embedding_dimensions)
                self._conn.execute(
                    "INSERT INTO corpus_state (fingerprint, source_commit, embedding_model,"
                    " embedding_dimensions, expected_chunks) VALUES (%s, %s, %s, %s, %s)",
                    (
                        fingerprint,
                        source_commit,
                        embedding_model,
                        embedding_dimensions,
                        expected_chunks,
                    ),
                )
                self._conn.commit()
                return set()

            rows = self._conn.execute(
                "SELECT source_path, chunk_index FROM chunks"
            ).fetchall()
            self._conn.rollback()
        return {(path, index) for path, index in rows}

    def _recreate_tables(self, embedding_dimensions: int) -> None:
        self._conn.execute("DROP TABLE IF EXISTS chunks")
        self._conn.execute("DROP TABLE IF EXISTS corpus_state")
        self._conn.execute(
            "CREATE TABLE corpus_state ("
            " id boolean PRIMARY KEY DEFAULT true CHECK (id),"
            " fingerprint text NOT NULL,"
            " source_commit text NOT NULL,"
            " embedding_model text NOT NULL,"
            " embedding_dimensions integer NOT NULL,"
            " expected_chunks integer NOT NULL,"
            " completed_at timestamptz)"
        )
        self._conn.execute(
            sql.SQL(
                "CREATE TABLE chunks ("
                " source_path text NOT NULL,"
                " chunk_index integer NOT NULL,"
                " heading text NOT NULL,"
                " content text NOT NULL,"
                " embedding vector({}) NOT NULL,"
                " PRIMARY KEY (source_path, chunk_index))"
            ).format(sql.Literal(embedding_dimensions))
        )

    def add_chunks(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Commit one batch, so a later failure cannot lose it."""
        rows = [
            (c.source_path, c.chunk_index, c.heading, c.content, Vector(list(e)))
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with _store_errors("store chunks"):
            with self._conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO chunks (source_path, chunk_index, heading, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    rows,
                )
            self._conn.commit()

    def mark_complete(self) -> None:
        with _store_errors("mark the corpus complete"):
            row = self._conn.execute(
                "UPDATE corpus_state SET completed_at = now()"
                " WHERE expected_chunks = (SELECT count(*) FROM chunks)"
                " RETURNING id"
            ).fetchone()
            self._conn.commit()
        if row is None:
            raise StoreError("Stored chunk count does not match the expected count")

    def search(self, embedding: Sequence[float], top_k: int) -> list[SearchHit]:
        """Exact cosine search. No index is used, so every chunk is compared."""
        with _store_errors("search"):
            rows = self._conn.execute(
                "SELECT source_path, chunk_index, heading, content,"
                " 1 - (embedding <=> %(query)s) AS score"
                " FROM chunks ORDER BY embedding <=> %(query)s LIMIT %(top_k)s",
                {"query": Vector(list(embedding)), "top_k": top_k},
            ).fetchall()
            self._conn.rollback()
        return [SearchHit(*row) for row in rows]
