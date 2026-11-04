import io
import sys

from baldur_docs_rag import cli
from baldur_docs_rag.cli import PREVIEW_CHARS, format_search_result
from baldur_docs_rag.ingest import pacing_delay_seconds
from baldur_docs_rag.search import SearchResult
from baldur_docs_rag.store import SearchHit


def test_output_survives_a_cp949_console(monkeypatch):
    # Printing a Baldur document crashed with UnicodeEncodeError on this machine.
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp949"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp949"))
    monkeypatch.setattr(cli.logging.getLogger("baldur_docs_rag"), "handlers", [])

    cli.configure_output()
    sys.stdout.write("slice — end")
    sys.stdout.flush()

    assert raw.getvalue().decode("utf-8") == "slice — end"


def test_search_output_shows_source_commit_score_and_heading():
    hit = SearchHit("docs/a.md", 3, "A > B", "short body", 0.71234)

    output = format_search_result(SearchResult("abc123", [hit]))

    assert "abc123" in output
    assert "score=0.7123 docs/a.md#3" in output
    assert "A > B" in output
    assert "short body" in output


def test_long_content_is_cut_and_marked():
    hit = SearchHit("docs/a.md", 0, "A", "x" * (PREVIEW_CHARS + 1), 0.5)

    output = format_search_result(SearchResult("abc", [hit]))

    assert "x" * PREVIEW_CHARS + " ..." in output
    assert "x" * (PREVIEW_CHARS + 1) not in output


def test_pacing_follows_the_token_limit_for_long_texts():
    # 6000 characters is estimated as 2000 tokens, a tenth of a 20000 budget.
    delay = pacing_delay_seconds(["a" * 6000], tokens_per_minute=20_000, texts_per_minute=80)

    assert delay == 6.0


def test_pacing_follows_the_text_limit_for_short_texts():
    delay = pacing_delay_seconds(["a"] * 20, tokens_per_minute=20_000, texts_per_minute=80)

    assert delay == 15.0
