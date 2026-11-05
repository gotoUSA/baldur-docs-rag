"""The fakes copy responses and error classes seen from the real API.

The rate limit case is the exception: a real 429 has not been seen yet, so that
test only checks the mapping of the SDK's own RateLimitError class.
"""

from types import SimpleNamespace

import httpx
import pytest
from google.genai import _interactions

from baldur_docs_rag.answer import (
    SYSTEM_INSTRUCTION,
    GeminiAnswerer,
    ModelAnswer,
    build_prompt,
    run_answer,
)
from baldur_docs_rag.config import Settings
from baldur_docs_rag.errors import (
    AnswerError,
    AnswerQuotaError,
    AnswerRejectedError,
)
from baldur_docs_rag.search import SearchResult
from baldur_docs_rag.store import SearchHit

COMMIT = "20a2f7f3"
HITS = [
    SearchHit("docs/a.md", 0, "A", "alpha body", 0.74),
    SearchHit("docs/b.md", 2, "B > C", "bravo body", 0.72),
]


class FakeInteractions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def settings():
    return Settings(_env_file=None, gemini_api_key="k")


def answerer_with(outcome):
    interactions = FakeInteractions(outcome)
    client = SimpleNamespace(interactions=interactions)
    return GeminiAnswerer(settings(), client=client), interactions


def interaction(output_text, status="completed"):
    return SimpleNamespace(status=status, output_text=output_text)


def status_error(error_class, code):
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(code, request=request)
    return error_class("failed", response=response, body=None)


def test_prompt_numbers_evidence_from_one_with_its_heading():
    prompt = build_prompt("What is it?", HITS)

    assert "[1] A\nalpha body" in prompt
    assert "[2] B > C\nbravo body" in prompt
    assert prompt.endswith("Question: What is it?")


def test_request_uses_the_settings_schema_and_no_server_storage():
    answerer, interactions = answerer_with(
        interaction('{"answerable":true,"answer":"x","citations":[1]}')
    )

    answerer.answer("q", HITS)

    sent = interactions.calls[0]
    assert sent["model"] == settings().answer_model
    assert sent["store"] is False
    assert sent["timeout"] == settings().answer_timeout_seconds
    assert sent["system_instruction"] == SYSTEM_INSTRUCTION
    assert sent["response_format"]["mime_type"] == "application/json"
    assert set(sent["response_format"]["schema"]["properties"]) == {
        "answerable",
        "answer",
        "citations",
    }


def test_pretty_printed_json_from_the_real_api_is_accepted():
    # A real refusal arrived with line breaks and indentation.
    answerer, _ = answerer_with(
        interaction(
            '{\n  "answerable": false,\n  "answer": "no",\n  "citations": []\n}'
        )
    )

    assert answerer.answer("q", HITS).answerable is False


@pytest.mark.parametrize("output_text", ["not json", "", None, '{"answer":"only"}'])
def test_wrong_format_is_a_failure(output_text):
    answerer, _ = answerer_with(interaction(output_text))

    with pytest.raises(AnswerError, match="wrong format"):
        answerer.answer("q", HITS)


def test_incomplete_status_is_a_failure_even_with_text():
    answerer, _ = answerer_with(
        interaction(
            '{"answerable":true,"answer":"x","citations":[1]}', status="incomplete"
        )
    )

    with pytest.raises(AnswerError, match="incomplete"):
        answerer.answer("q", HITS)


def test_rate_limit_becomes_a_quota_error():
    answerer, _ = answerer_with(status_error(_interactions.RateLimitError, 429))

    with pytest.raises(AnswerQuotaError, match="quota"):
        answerer.answer("q", HITS)


@pytest.mark.parametrize(
    "error_class, code",
    [
        (_interactions.NotFoundError, 404),
        (_interactions.BadRequestError, 400),
        (_interactions.InternalServerError, 503),
    ],
)
def test_status_errors_seen_from_the_real_api_become_answer_errors(error_class, code):
    answerer, _ = answerer_with(status_error(error_class, code))

    with pytest.raises(AnswerError, match=str(code)) as caught:
        answerer.answer("q", HITS)

    assert not isinstance(caught.value, AnswerQuotaError | AnswerRejectedError)
    assert isinstance(caught.value.__cause__, error_class)


def test_timeout_seen_from_the_real_api_becomes_an_answer_error():
    request = httpx.Request("POST", "https://example.test")
    answerer, _ = answerer_with(_interactions.APITimeoutError(request=request))

    with pytest.raises(AnswerError, match="APITimeoutError"):
        answerer.answer("q", HITS)


class FixedAnswerer:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def answer(self, question, hits):
        self.calls += 1
        return self.reply


def run(monkeypatch, reply, hits=HITS):
    monkeypatch.setattr(
        "baldur_docs_rag.answer.run_search",
        lambda *_args: SearchResult(COMMIT, list(hits)),
    )
    answerer = FixedAnswerer(reply)
    return run_answer("q", settings(), object(), answerer, object()), answerer


def test_answer_lists_only_the_cited_sources(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="It limits calls.", citations=[2])

    result, _ = run(monkeypatch, reply)

    assert result.answer == "It limits calls."
    assert result.source_commit == COMMIT
    assert [(s.number, s.hit.source_path) for s in result.sources] == [(2, "docs/b.md")]


def test_repeated_citations_are_listed_once_in_order(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="x", citations=[2, 1, 2])

    result, _ = run(monkeypatch, reply)

    assert [s.number for s in result.sources] == [2, 1]


def test_refusal_hides_the_model_wording_and_has_no_sources(monkeypatch):
    reply = ModelAnswer(answerable=False, answer="Maybe it is 42.", citations=[1])

    result, _ = run(monkeypatch, reply)

    assert result.answer is None
    assert result.sources == []


@pytest.mark.parametrize("number", [0, 3, -1])
def test_citation_outside_the_evidence_is_a_failure(monkeypatch, number):
    reply = ModelAnswer(answerable=True, answer="x", citations=[1, number])

    with pytest.raises(AnswerError, match="not retrieved"):
        run(monkeypatch, reply)


def test_answer_without_citations_is_a_failure(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="x", citations=[])

    with pytest.raises(AnswerRejectedError, match="without citing"):
        run(monkeypatch, reply)


def test_blank_answer_is_a_failure(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="  ", citations=[1])

    with pytest.raises(AnswerError, match="no answer"):
        run(monkeypatch, reply)


def test_no_retrieved_chunks_refuses_without_calling_the_model(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="x", citations=[1])

    result, answerer = run(monkeypatch, reply, hits=[])

    assert result.answer is None
    assert answerer.calls == 0
