"""Print a per-question breakdown and 95% Wilson intervals for the saved results.

Reads only files already in the repository, so it makes no API call.
Run from the repository root: python eval/results/intervals.py
"""

import json
import math

Z_95 = 1.959964
SETS = [
    (
        "set1",
        "eval/questions.json",
        "eval/results/20a2f7f3-gemini-3.6-flash-1e9a160e.jsonl",
        "eval/results/20a2f7f3-gemini-3.6-flash-1e9a160e.grades.json",
    ),
    (
        "set2",
        "eval/questions-2.json",
        "eval/results/20a2f7f3-gemini-3.6-flash-50e2f547.jsonl",
        "eval/results/20a2f7f3-gemini-3.6-flash-50e2f547.grades.json",
    ),
]


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    denominator = 1 + Z_95**2 / n
    center = (p + Z_95**2 / (2 * n)) / denominator
    margin = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def ratio(name: str, k: int, n: int) -> None:
    low, high = wilson(k, n)
    se = math.sqrt((k / n) * (1 - k / n) / n)
    print(f"  {name:28} {k}/{n} = {k / n:.0%}  SE {se:.3f}  95% {low:.0%}-{high:.0%}")


for name, questions_path, results_path, grades_path in SETS:
    questions = json.load(open(questions_path, encoding="utf-8"))
    questions = questions["questions"] if isinstance(questions, dict) else questions
    retrievals, answers = {}, {}
    for line in open(results_path, encoding="utf-8"):
        record = json.loads(line)
        if record["kind"] == "retrieval":
            retrievals[record["id"]] = record["hits"]
        elif record["kind"] == "answer":
            answers[record["id"]] = record
    grades = json.load(open(grades_path, encoding="utf-8"))

    print(f"== {name}")
    counts = {"hit": 0, "rank1": 0, "answered": 0, "match": 0, "wrong_refusal": 0}
    refusals = {"related_unanswerable": [0, 0], "unrelated": [0, 0]}
    answerable = 0
    for question in questions:
        answer = answers[question["id"]]
        group = question["group"]
        if group != "answerable":
            refusals[group][0] += answer["outcome"] == "refused"
            refusals[group][1] += 1
            continue
        answerable += 1
        expected = {
            (c["source_path"], c["chunk_index"]) for c in question["expected_chunks"]
        }
        hits = [
            (h["source_path"], h["chunk_index"]) for h in retrievals[question["id"]]
        ]
        rank = next((i + 1 for i, hit in enumerate(hits) if hit in expected), None)
        counts["hit"] += rank is not None
        counts["rank1"] += rank == 1
        match = None
        if answer["outcome"] == "answered":
            counts["answered"] += 1
            cited = {hits[number - 1] for number in answer["citations"]}
            match = bool(cited & expected)
            counts["match"] += match
        else:
            counts["wrong_refusal"] += 1
        print(
            f"  {question['id']} rank={rank} outcome={answer['outcome']} "
            f"citation_match={match} grade={grades.get(question['id'])}"
        )
    correct = sum(grade == "correct" for grade in grades.values())
    ratio("retrieval hit (top 5)", counts["hit"], answerable)
    ratio("expected chunk at rank 1", counts["rank1"], answerable)
    ratio("citation match", counts["match"], counts["answered"])
    ratio("wrong refusal", counts["wrong_refusal"], answerable)
    ratio("graded correct (AI)", correct, len(grades))
    for group, (refused, total) in refusals.items():
        ratio(f"refused ({group})", refused, total)
