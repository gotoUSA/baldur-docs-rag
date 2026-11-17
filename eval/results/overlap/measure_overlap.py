"""Measure how many question words also appear in the expected chunks (no API calls).

Run with the database up: .venv/Scripts/python.exe eval/results/overlap/measure_overlap.py
Words are PostgreSQL english lexemes, the same setting as the keyword measurement.
"""

import json
import re
import sys
from pathlib import Path
from statistics import mean, median

import psycopg

from baldur_docs_rag.config import Settings
from baldur_docs_rag.evaluation import load_questions

SETS = (("set1", "eval/questions.json"), ("set2", "eval/questions-2.json"))
KEYWORD_RESULT = Path("eval/results/keyword/keyword-result.json")
OUT_FILE = Path(__file__).resolve().parent / "result.json"
TOP = 5
HANGUL = re.compile(r"[가-힣]")

LEXEMES_SQL = "SELECT array(SELECT lexeme FROM unnest(to_tsvector('english', %s)))"
CHUNK_LEXEMES_SQL = """
SELECT array(SELECT lexeme FROM unnest(to_tsvector('english', heading || ' ' || content)))
FROM chunks WHERE source_path = %s AND chunk_index = %s
"""


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    settings = Settings()
    vector_rank = {
        label: {row["id"]: row["vector"] for row in rows}
        for label, rows in json.loads(
            KEYWORD_RESULT.read_text(encoding="utf-8")
        ).items()
    }
    out: dict = {"sets": {}}
    with psycopg.connect(settings.database_url) as conn:
        for label, path in SETS:
            questions, _ = load_questions(Path(path))
            rows = []
            for q in questions.questions:
                if q.group != "answerable":
                    continue
                q_lex = set(conn.execute(LEXEMES_SQL, (q.question,)).fetchone()[0])
                best = 0.0
                for ref in q.expected_chunks:
                    found = conn.execute(
                        CHUNK_LEXEMES_SQL, (ref.source_path, ref.chunk_index)
                    ).fetchone()
                    if found is None:
                        raise SystemExit(f"{q.id}: expected chunk missing from the DB")
                    if q_lex:
                        best = max(best, len(q_lex & set(found[0])) / len(q_lex))
                rank = vector_rank[label].get(q.id)
                rows.append(
                    {
                        "id": q.id,
                        "language": "ko" if HANGUL.search(q.question) else "en",
                        "question_lexemes": len(q_lex),
                        "overlap": round(best, 3),
                        "vector_rank": rank,
                        "vector_hit": rank is not None and rank <= TOP,
                    }
                )
            out["sets"][label] = {
                "rows": rows,
                "en": summary([r["overlap"] for r in rows if r["language"] == "en"]),
                "ko": summary([r["overlap"] for r in rows if r["language"] == "ko"]),
            }
    set2 = out["sets"]["set2"]["rows"]
    out["set2_en_by_hit"] = {
        "hit": summary(
            [r["overlap"] for r in set2 if r["language"] == "en" and r["vector_hit"]]
        ),
        "miss": summary(
            [
                r["overlap"]
                for r in set2
                if r["language"] == "en" and not r["vector_hit"]
            ]
        ),
    }
    OUT_FILE.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for label in ("set1", "set2"):
        s = out["sets"][label]
        print(label, "en", s["en"], "ko", s["ko"])
        print(
            "  en rows:",
            [
                (r["id"], r["overlap"], r["question_lexemes"])
                for r in s["rows"]
                if r["language"] == "en"
            ],
        )
    print("set2 en by vector hit:", out["set2_en_by_hit"])
    print("saved", OUT_FILE)


if __name__ == "__main__":
    main()
