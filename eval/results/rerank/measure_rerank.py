"""Compare vector top-5 with cross-encoder reranking of the vector top 50, on sets 1 and 2 only.

Step 1 (candidates): embed each answerable question once and save its top 50 chunks.
Step 2 (rerank MODEL_ID [BATCH] [DEVICE] [DTYPE]): rescore those saved candidates locally
and report top-5 hits. DEVICE defaults to cpu; DTYPE (e.g. float32) defaults to the model's own.
The sealed third set is never read here.
"""

import json
import sys
import time
from pathlib import Path

from baldur_docs_rag.config import Settings
from baldur_docs_rag.embedding import GeminiEmbedder
from baldur_docs_rag.evaluation import QuestionGroup, load_questions
from baldur_docs_rag.store import Store

CANDIDATES = 50
TOP = 5
# sentence-transformers' own default for CrossEncoder.predict.
DEFAULT_BATCH_SIZE = 32
SETS = {"set1": Path("eval/questions.json"), "set2": Path("eval/questions-2.json")}
OUT_DIR = Path("eval/results/rerank")
CANDIDATES_FILE = OUT_DIR / "candidates.json"


def document_text(heading: str, content: str) -> str:
    return f"{heading}\n\n{content}"


def collect_candidates() -> None:
    settings = Settings()
    embedder = GeminiEmbedder(settings)
    out = {}
    with Store(settings.database_url) as store:
        for label, path in SETS.items():
            questions, digest = load_questions(path)
            rows = []
            for q in questions.questions:
                if q.group is not QuestionGroup.ANSWERABLE:
                    continue
                hits = store.search(embedder.embed_query(q.question), CANDIDATES)
                rows.append(
                    {
                        "id": q.id,
                        "question": q.question,
                        "expected": [
                            [c.source_path, c.chunk_index] for c in q.expected_chunks
                        ],
                        "candidates": [
                            {
                                "ref": [h.source_path, h.chunk_index],
                                "text": document_text(h.heading, h.content),
                            }
                            for h in hits
                        ],
                    }
                )
            out[label] = {"questions_sha256": digest, "rows": rows}
    CANDIDATES_FILE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("saved", CANDIDATES_FILE, {k: len(v["rows"]) for k, v in out.items()})


def best_rank(
    refs: list[tuple[str, int]], expected: set[tuple[str, int]]
) -> int | None:
    ranks = [i for i, ref in enumerate(refs, start=1) if ref in expected]
    return min(ranks) if ranks else None


def rerank(model_id: str, batch_size: int, device: str, dtype: str | None) -> None:
    import torch
    from sentence_transformers import CrossEncoder

    model_kwargs = {"dtype": getattr(torch, dtype)} if dtype else None
    loaded_at = time.monotonic()
    if model_id == "nlpai-lab/KURE-Reranker-nano":
        # The model card constructs it with this max_length.
        model = CrossEncoder(
            model_id, max_length=8192, device=device, model_kwargs=model_kwargs
        )
    else:
        model = CrossEncoder(model_id, device=device, model_kwargs=model_kwargs)
    load_seconds = time.monotonic() - loaded_at

    data = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
    report = {
        "model": model_id,
        "batch_size": batch_size,
        "device": str(model.device),
        "dtype": str(model.dtype),
        "load_seconds": round(load_seconds, 1),
        "sets": {},
    }
    for label, block in data.items():
        rows = []
        for row in block["rows"]:
            expected = {tuple(e) for e in row["expected"]}
            refs = [tuple(c["ref"]) for c in row["candidates"]]
            started = time.monotonic()
            scores = model.predict(
                [(row["question"], c["text"]) for c in row["candidates"]],
                batch_size=batch_size,
            )
            seconds = time.monotonic() - started
            order = sorted(range(len(refs)), key=lambda i: -float(scores[i]))
            reranked = [refs[i] for i in order]
            vector_best = best_rank(refs, expected)
            rerank_best = best_rank(reranked, expected)
            rows.append(
                {
                    "id": row["id"],
                    "vector_best": vector_best,
                    "rerank_best": rerank_best,
                    "vector_hit": vector_best is not None and vector_best <= TOP,
                    "rerank_hit": rerank_best is not None and rerank_best <= TOP,
                    "seconds": round(seconds, 2),
                }
            )
            print(label, row["id"], f"{seconds:.1f}s", flush=True)
        report["sets"][label] = rows
    name = model_id.split("/")[-1]
    out = OUT_DIR / f"result-{name}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for label, rows in report["sets"].items():
        n = len(rows)
        vector = sum(r["vector_hit"] for r in rows)
        reranked = sum(r["rerank_hit"] for r in rows)
        gained = [r["id"] for r in rows if r["rerank_hit"] and not r["vector_hit"]]
        lost = [r["id"] for r in rows if r["vector_hit"] and not r["rerank_hit"]]
        mean = sum(r["seconds"] for r in rows) / n
        print(
            f"{name} {label}: vector {vector}/{n} rerank {reranked}/{n} gained {gained} lost {lost} mean {mean:.1f}s"
        )
    print("saved", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "candidates":
        collect_candidates()
    else:
        args = sys.argv[1:] + [None] * 3
        rerank(
            args[0],
            int(args[1]) if args[1] else DEFAULT_BATCH_SIZE,
            args[2] or "cpu",
            args[3],
        )
