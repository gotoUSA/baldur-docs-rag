import io
import sys
from types import SimpleNamespace

from baldur_docs_rag import cli
from baldur_docs_rag.answer import AnswerResult, CitedSource
from baldur_docs_rag.cli import PREVIEW_CHARS, format_search_result
from baldur_docs_rag.errors import EmbeddingQuotaError
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
    delay = pacing_delay_seconds(
        ["a" * 6000], tokens_per_minute=20_000, texts_per_minute=80
    )

    assert delay == 6.0


def test_pacing_follows_the_text_limit_for_short_texts():
    delay = pacing_delay_seconds(
        ["a"] * 20, tokens_per_minute=20_000, texts_per_minute=80
    )

    assert delay == 15.0


class _FakeStore:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _run_with_search_failure(monkeypatch, error, argv):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(cli, "configure_output", lambda: None)
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(database_url="x"))
    monkeypatch.setattr(cli, "GeminiEmbedder", lambda _settings: object())
    monkeypatch.setattr(cli, "Store", lambda _url: _FakeStore())
    monkeypatch.setattr(cli, "run_search", fail)
    return cli.main(argv)


def test_quota_stop_during_search_does_not_promise_resuming_an_ingest(
    monkeypatch, capsys
):
    code = _run_with_search_failure(
        monkeypatch, EmbeddingQuotaError("quota reached"), ["search", "question"]
    )

    assert code == cli.EXIT_FAILED
    error_output = capsys.readouterr().err
    assert "quota reached" in error_output
    assert "Finished batches" not in error_output


def test_answer_output_shows_the_answer_commit_and_cited_source():
    hit = SearchHit("docs/a.md", 3, "A > B", "body", 0.7)

    output = cli.format_answer(
        AnswerResult("abc123", "It limits calls.", [CitedSource(2, hit)])
    )

    assert output.startswith("It limits calls.")
    assert "abc123" in output
    assert "[2] docs/a.md#3 A > B" in output


def test_refusal_output_says_so_and_names_the_commit():
    output = cli.format_answer(AnswerResult("abc123", None, []))

    assert output.startswith(cli.NOT_IN_DOCUMENTS)
    assert "abc123" in output


def test_ask_prints_the_answer_and_exits_ok(monkeypatch, capsys):
    monkeypatch.setattr(cli, "configure_output", lambda: None)
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(database_url="x"))
    monkeypatch.setattr(cli, "GeminiEmbedder", lambda _settings: object())
    monkeypatch.setattr(cli, "GeminiAnswerer", lambda _settings: object())
    monkeypatch.setattr(cli, "Store", lambda _url: _FakeStore())
    monkeypatch.setattr(cli, "run_answer", lambda *_args: AnswerResult("abc", None, []))

    assert cli.main(["ask", "question"]) == cli.EXIT_OK
    assert cli.NOT_IN_DOCUMENTS in capsys.readouterr().out


def test_quota_stop_during_ingest_still_explains_resuming(monkeypatch, capsys):
    monkeypatch.setattr(cli, "configure_output", lambda: None)
    monkeypatch.setattr(cli, "load_settings", lambda: SimpleNamespace(database_url="x"))
    monkeypatch.setattr(cli, "GeminiEmbedder", lambda _settings: object())
    monkeypatch.setattr(cli, "Store", lambda _url: _FakeStore())

    def fail(*_args):
        raise EmbeddingQuotaError("quota reached")

    monkeypatch.setattr(cli, "run_ingest", fail)

    assert cli.main(["ingest"]) == cli.EXIT_FAILED
    assert "Finished batches" in capsys.readouterr().err
