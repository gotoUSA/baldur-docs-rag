"""Count the tokens of every indexed document at one commit with one countTokens call.

Run: .venv/Scripts/python.exe eval/results/tokens/count_tokens.py
The documents are the same set ingest reads (load_documents). The call is made once,
with SDK retries off, because the docs do not say whether it counts toward the
daily request limit.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from google import genai
from google.genai import types

from baldur_docs_rag.config import load_settings
from baldur_docs_rag.documents import load_documents, resolve_commit

SOURCE_REF = "20a2f7f3"
EXPECTED_DOCUMENTS = 117
EXPECTED_CHARS = 1_020_253
TIMEOUT_MS = 120_000
OUT_FILE = Path(__file__).resolve().parent / "result.json"


def main() -> int:
    settings = load_settings()
    commit = resolve_commit(settings.baldur_repo_path, SOURCE_REF)
    documents = load_documents(settings.baldur_repo_path, commit)
    chars = sum(len(d.text) for d in documents)
    print(f"commit {commit[:8]}, documents {len(documents)}, chars {chars}")
    if len(documents) != EXPECTED_DOCUMENTS or chars != EXPECTED_CHARS:
        print("document set differs from the recorded one; not calling the API")
        return 1

    client = genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(
            timeout=TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    # Raw document text only, one part per document; no prompt or file headers.
    response = client.models.count_tokens(
        model=settings.answer_model,
        contents=[types.Part.from_text(text=d.text) for d in documents],
    )
    result = {
        "measured_at": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "model": settings.answer_model,
        "documents": len(documents),
        "chars": chars,
        "total_tokens": response.total_tokens,
        "chars_per_token": round(chars / response.total_tokens, 3),
    }
    OUT_FILE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
