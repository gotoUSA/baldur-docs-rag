"""Study only: ask Gemini about r01 once more and keep the reply text it wrote.

evaluate drops the answer text of a refusal, so the reason was never seen. This sends
the five chunks evaluate saved for r01, with the same instruction and schema, and saves
the whole reply. It is a new reply, not the one counted in the evaluation, and it is
kept out of every evaluation number. The plan is in eval/RECORD.md.
Run from the repository root: python eval/results/refusal/r01_reason.py
"""

import json
import sys
from pathlib import Path

from baldur_docs_rag.answer import GeminiAnswerer
from baldur_docs_rag.config import Settings
from baldur_docs_rag.store import SearchHit

QUESTION_ID = "r01"
QUESTIONS_FILE = Path("eval/questions.json")
RESULTS_FILE = Path("eval/results/20a2f7f3-gemini-3.6-flash-1e9a160e.jsonl")
OUT_FILE = Path("eval/results/refusal/r01-reply.json")


def main() -> None:
    if OUT_FILE.exists():
        raise SystemExit(f"{OUT_FILE} already exists; the plan allows one call")
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    question = next(q["question"] for q in questions if q["id"] == QUESTION_ID)
    rows = [
        json.loads(line)
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run = next(r for r in rows if r["kind"] == "run")
    saved = next(r for r in rows if r["kind"] == "retrieval" and r["id"] == QUESTION_ID)
    hits = [SearchHit(**hit) for hit in saved["hits"]]

    settings = Settings()
    if settings.answer_model != run["answer_model"]:
        raise SystemExit(
            f"answer model {settings.answer_model} differs from the evaluation's "
            f"{run['answer_model']}"
        )
    reply = GeminiAnswerer(settings).answer(question, hits)
    OUT_FILE.write_text(
        json.dumps(
            {
                "id": QUESTION_ID,
                "model": settings.answer_model,
                "source_commit": run["source_commit"],
                "chunks": [[h.source_path, h.chunk_index] for h in hits],
                "reply": reply.model_dump(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(json.dumps(reply.model_dump(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
