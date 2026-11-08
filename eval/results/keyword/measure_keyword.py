"""Measure top-5 hit rate for vector, keyword (PostgreSQL full text) and RRF fusion."""

import json
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from baldur_docs_rag.config import Settings
from baldur_docs_rag.embedding import GeminiEmbedder
from baldur_docs_rag.evaluation import load_questions
from baldur_docs_rag.store import Store

DEPTH = 50
RRF_K = 60
TOP = 5

KEYWORD_SQL = """
WITH q AS (
  SELECT NULLIF(replace(plainto_tsquery('english', %(question)s)::text, '&', '|'), '')
         AS text
)
SELECT source_path, chunk_index
FROM chunks, q
WHERE q.text IS NOT NULL
  AND to_tsvector('english', heading || ' ' || content) @@ q.text::tsquery
ORDER BY ts_rank(to_tsvector('english', heading || ' ' || content), q.text::tsquery) DESC,
         source_path, chunk_index
LIMIT %(depth)s
"""


def rrf(*ranked: list[tuple[str, int]]) -> list[tuple[str, int]]:
    scores: dict[tuple[str, int], float] = {}
    for ranking in ranked:
        for rank, ref in enumerate(ranking, start=1):
            scores[ref] = scores.get(ref, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda ref: (-scores[ref], ref))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    settings = Settings()
    embedder = GeminiEmbedder(settings)
    out = {}
    with (
        Store(settings.database_url) as store,
        psycopg.connect(settings.database_url) as conn,
    ):
        register_vector(conn)
        for label, path in (
            ("set1", "eval/questions.json"),
            ("set2", "eval/questions-2.json"),
        ):
            questions, _ = load_questions(Path(path))
            rows = []
            for q in questions.questions:
                if q.group != "answerable":
                    continue
                expected = {(c.source_path, c.chunk_index) for c in q.expected_chunks}
                vec = [
                    (h.source_path, h.chunk_index)
                    for h in store.search(embedder.embed_query(q.question), DEPTH)
                ]
                kw = [
                    tuple(r)
                    for r in conn.execute(
                        KEYWORD_SQL, {"question": q.question, "depth": DEPTH}
                    ).fetchall()
                ]
                fused = rrf(vec, kw)

                def best(ranking: list[tuple[str, int]]) -> int | None:
                    ranks = [
                        i for i, ref in enumerate(ranking, start=1) if ref in expected
                    ]
                    return min(ranks) if ranks else None

                rows.append(
                    {
                        "id": q.id,
                        "vector": best(vec),
                        "keyword": best(kw),
                        "fused": best(fused),
                        "keyword_results": len(kw),
                    }
                )
            out[label] = rows
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
