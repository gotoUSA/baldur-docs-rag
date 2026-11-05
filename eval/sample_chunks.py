"""Pick evaluation chunks at random, so the author cannot favor easy passages.

Prints the chunks in a fixed order for the seed. Chunks are taken in that order,
and any chunk skipped while writing questions is recorded with its reason.
"""

import random

import psycopg

from baldur_docs_rag.config import load_settings

SEED = 20261105
MIN_CHARS = 200
SAMPLE_SIZE = 44


def main() -> None:
    with psycopg.connect(load_settings().database_url) as conn:
        rows = conn.execute(
            "SELECT source_path, chunk_index, heading, content FROM chunks"
            " WHERE length(content) >= %s ORDER BY source_path, chunk_index",
            (MIN_CHARS,),
        ).fetchall()

    random.Random(SEED).shuffle(rows)

    seen_paths: set[str] = set()
    taken = 0
    for path, index, heading, content in rows:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        taken += 1
        print(f"=== {taken} {path}#{index} | {heading}\n{content}\n")
        if taken == SAMPLE_SIZE:
            break


if __name__ == "__main__":
    main()
