"""Build grading-input.md for answered, answerable questions from an evaluation results file."""

import json
import sys
from pathlib import Path


def build(questions_path: Path, results_path: Path, ids: set[str] | None = None) -> str:
    """ids limits the output to those questions; None keeps every answered one."""
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hits = {r["id"]: r["hits"] for r in rows if r["kind"] == "retrieval"}
    answers = {r["id"]: r for r in rows if r["kind"] == "answer"}

    parts = ["# 채점 자료\n"]
    for q in questions:
        a = answers.get(q["id"])
        if q["group"] != "answerable" or a is None or a["outcome"] != "answered":
            continue
        if ids is not None and q["id"] not in ids:
            continue
        parts.append(f"## {q['id']}\n")
        parts.append(f"### 질문\n\n{q['question']}\n")
        parts.append(f"### 답변\n\n{a['answer']}\n")
        parts.append("### 인용 문단\n")
        for number in a["citations"]:
            hit = hits[q["id"]][number - 1]
            parts.append(
                f"#### [{number}]\n\n<passage>\n제목: {hit['heading']}\n\n"
                f"{hit['content']}\n</passage>\n"
            )
    return "\n".join(parts)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(build(Path(sys.argv[1]), Path(sys.argv[2])))
