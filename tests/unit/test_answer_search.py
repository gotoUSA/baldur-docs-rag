"""The search tool in ask. The fakes copy responses seen from the real API on 2026-11-09.

A real request came back with status requires_action and two steps, a thought with a
signature and a function_call with id, name and arguments. Resending those steps with
a function_result gave status completed and a JSON answer. Wrong names, extra
arguments and parallel calls have not been seen from the real API; those fakes only
check that the code refuses them.
"""

from types import SimpleNamespace

import pytest

from baldur_docs_rag.answer import (
    NO_NEW_EVIDENCE,
    SEARCH_QUERY_MAX_CHARS,
    search_system_instruction,
    SEARCH_TOOL,
    SYSTEM_INSTRUCTION,
    GeminiAnswerer,
    ModelAnswer,
    SearchedAnswer,
    run_answer_with_search,
)
from baldur_docs_rag.config import Settings
from baldur_docs_rag.errors import AnswerRejectedError
from baldur_docs_rag.search import SearchResult
from baldur_docs_rag.store import SearchHit

COMMIT = "20a2f7f3"
HITS = [
    SearchHit("docs/a.md", 0, "A", "alpha body", 0.74),
    SearchHit("docs/b.md", 2, "B > C", "bravo body", 0.72),
]
# The same chunk as HITS[1], found again by the second search with another score.
REPEATED = SearchHit("docs/b.md", 2, "B > C", "bravo body", 0.61)
NEW = SearchHit("docs/c.md", 4, "Policies", "per-service body", 0.66)
CALL_ID = "call_9908724702816832641"


class Step(SimpleNamespace):
    def model_dump(self):
        return dict(vars(self))


def thought():
    return Step(type="thought", signature="Eu8JCuwJAWkU")


def function_call(name="search_docs", arguments=None, call_id=CALL_ID):
    if arguments is None:
        arguments = {"query": "per-service policy"}
    return Step(type="function_call", id=call_id, name=name, arguments=arguments)


def asks_for(*calls, status="requires_action"):
    return SimpleNamespace(status=status, output_text="", steps=[thought(), *calls])


def answers(output_text):
    return SimpleNamespace(
        status="completed",
        output_text=output_text,
        steps=[thought(), Step(type="model_output")],
    )


class FakeInteractions:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        # Copy the history, since the code keeps appending to the same list.
        sent = dict(kwargs)
        if isinstance(sent["input"], list):
            sent["input"] = list(sent["input"])
        self.calls.append(sent)
        return self.outcomes.pop(0)


class FakeSearch:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def __call__(self, query):
        self.queries.append(query)
        return list(self.hits)


def settings(**values):
    return Settings(_env_file=None, gemini_api_key="k", **values)


def answerer_with(*outcomes, **setting_values):
    interactions = FakeInteractions(*outcomes)
    client = SimpleNamespace(interactions=interactions)
    return GeminiAnswerer(settings(**setting_values), client=client), interactions


def test_requested_search_runs_and_its_hits_continue_the_numbering():
    answerer, interactions = answerer_with(
        asks_for(function_call()),
        answers('{"answerable":true,"answer":"x","citations":[3]}'),
    )
    search = FakeSearch([REPEATED, NEW])

    searched = answerer.answer_with_search("q", HITS, search)

    assert search.queries == ["per-service policy"]
    assert searched.searches == ["per-service policy"]
    # The repeated chunk is not numbered twice, so the new one becomes 3.
    assert searched.evidence == [*HITS, NEW]
    result = interactions.calls[1]["input"][-1]
    assert result["type"] == "function_result"
    assert result["call_id"] == CALL_ID
    assert result["name"] == "search_docs"
    assert result["result"][0]["text"] == "[3] Policies\nper-service body"


def test_second_call_resends_the_question_and_the_model_steps_as_received():
    answerer, interactions = answerer_with(
        asks_for(function_call()),
        answers('{"answerable":false,"answer":"no","citations":[]}'),
    )

    answerer.answer_with_search("q", HITS, FakeSearch([NEW]))

    first, second = interactions.calls
    assert first["input"][0]["type"] == "user_input"
    assert second["input"][0] == first["input"][0]
    assert second["input"][1] == thought().model_dump()
    assert second["input"][2] == function_call().model_dump()
    for sent in (first, second):
        assert sent["tools"] == [SEARCH_TOOL]
        assert sent["store"] is False
        assert sent["system_instruction"] == search_system_instruction(1)


def test_a_search_with_nothing_new_says_so_to_the_model():
    answerer, interactions = answerer_with(
        asks_for(function_call()),
        answers('{"answerable":false,"answer":"no","citations":[]}'),
    )

    searched = answerer.answer_with_search("q", HITS, FakeSearch([REPEATED]))

    assert searched.evidence == HITS
    assert interactions.calls[1]["input"][-1]["result"][0]["text"] == NO_NEW_EVIDENCE


def test_an_answer_without_a_request_makes_one_call_and_no_search():
    answerer, interactions = answerer_with(
        answers('{"answerable":true,"answer":"x","citations":[1]}')
    )
    search = FakeSearch([NEW])

    searched = answerer.answer_with_search("q", HITS, search)

    assert searched.reply.citations == [1]
    assert searched.searches == []
    assert search.queries == []
    assert len(interactions.calls) == 1


def test_plain_answer_still_sends_no_tools_and_the_old_instruction():
    # The evaluation calls answer(), so its request must not change.
    answerer, interactions = answerer_with(
        answers('{"answerable":true,"answer":"x","citations":[1]}')
    )

    answerer.answer("q", HITS)

    assert "tools" not in interactions.calls[0]
    assert interactions.calls[0]["system_instruction"] == SYSTEM_INSTRUCTION


@pytest.mark.parametrize(
    "response, reason",
    [
        (asks_for(function_call(name="delete_docs")), "unknown tool"),
        (
            asks_for(function_call(arguments={"query": "x", "top_k": 50})),
            "single query",
        ),
        (asks_for(function_call(arguments={})), "single query"),
        (asks_for(function_call(arguments={"query": 7})), "empty or not text"),
        (asks_for(function_call(arguments={"query": "  "})), "empty or not text"),
        (
            asks_for(
                function_call(arguments={"query": "x" * (SEARCH_QUERY_MAX_CHARS + 1)})
            ),
            "longer than",
        ),
        (
            asks_for(function_call(), function_call(call_id="call_2")),
            "2 calls at once",
        ),
        (asks_for(function_call(), status="completed"), "status completed"),
    ],
)
def test_a_request_the_code_did_not_offer_is_refused_and_not_run(response, reason):
    answerer, interactions = answerer_with(response)
    search = FakeSearch([NEW])

    with pytest.raises(AnswerRejectedError, match=reason):
        answerer.answer_with_search("q", HITS, search)

    assert search.queries == []
    assert len(interactions.calls) == 1


def test_query_at_the_length_limit_is_run():
    query = "x" * SEARCH_QUERY_MAX_CHARS
    answerer, _ = answerer_with(
        asks_for(function_call(arguments={"query": query})),
        answers('{"answerable":false,"answer":"no","citations":[]}'),
    )
    search = FakeSearch([NEW])

    answerer.answer_with_search("q", HITS, search)

    assert search.queries == [query]


def test_a_request_past_the_search_limit_is_refused():
    answerer, interactions = answerer_with(
        asks_for(function_call()),
        asks_for(function_call(call_id="call_2")),
    )
    search = FakeSearch([NEW])

    with pytest.raises(AnswerRejectedError, match="more than 1 searches"):
        answerer.answer_with_search("q", HITS, search)

    assert len(search.queries) == 1
    assert len(interactions.calls) == 2


@pytest.mark.parametrize("limit", [1, 2])
def test_the_instruction_tells_the_model_the_search_limit(limit):
    # On 2026-11-10 the real model asked for a second search under a limit of one,
    # and nothing it was sent mentioned the limit.
    answerer, interactions = answerer_with(
        answers('{"answerable":true,"answer":"x","citations":[1]}'),
        search_tool_max_calls=limit,
    )

    answerer.answer_with_search("q", HITS, FakeSearch([NEW]))

    instruction = interactions.calls[0]["system_instruction"]
    assert instruction.startswith(SYSTEM_INSTRUCTION)
    assert f"at most {limit} " in instruction


def test_a_higher_limit_allows_a_second_search():
    other = SearchHit("docs/d.md", 1, "D", "delta body", 0.6)
    answerer, _ = answerer_with(
        asks_for(function_call()),
        asks_for(function_call(arguments={"query": "second"}, call_id="call_2")),
        answers('{"answerable":true,"answer":"x","citations":[4]}'),
        search_tool_max_calls=2,
    )
    searches = iter([[NEW], [other]])

    searched = answerer.answer_with_search("q", HITS, lambda _q: next(searches))

    assert searched.searches == ["per-service policy", "second"]
    assert searched.evidence == [*HITS, NEW, other]


class FixedSearchingAnswerer:
    def __init__(self, searched):
        self.searched = searched
        self.calls = 0

    def answer_with_search(self, question, hits, search):
        self.calls += 1
        return self.searched


def run(monkeypatch, searched, hits=HITS):
    monkeypatch.setattr(
        "baldur_docs_rag.answer.run_search",
        lambda *_args: SearchResult(COMMIT, list(hits)),
    )
    answerer = FixedSearchingAnswerer(searched)
    result = run_answer_with_search("q", settings(), object(), answerer, object())
    return result, answerer


def test_a_cited_hit_from_the_search_is_listed_with_its_number(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="x", citations=[3])
    searched = SearchedAnswer(reply, [*HITS, NEW], ["per-service policy"])

    result, _ = run(monkeypatch, searched)

    assert [(s.number, s.hit) for s in result.sources] == [(3, NEW)]
    assert result.searches == ["per-service policy"]
    assert result.source_commit == COMMIT


def test_a_citation_past_all_the_evidence_is_still_a_failure(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="x", citations=[4])
    searched = SearchedAnswer(reply, [*HITS, NEW], ["per-service policy"])

    with pytest.raises(AnswerRejectedError, match="not retrieved"):
        run(monkeypatch, searched)


def test_a_refusal_after_a_search_keeps_the_query(monkeypatch):
    reply = ModelAnswer(answerable=False, answer="Maybe 42.", citations=[3])
    searched = SearchedAnswer(reply, [*HITS, NEW], ["recovery off"])

    result, _ = run(monkeypatch, searched)

    assert result.answer is None
    assert result.sources == []
    assert result.searches == ["recovery off"]


def test_no_retrieved_chunks_refuses_without_calling_the_model(monkeypatch):
    reply = ModelAnswer(answerable=True, answer="x", citations=[1])

    result, answerer = run(monkeypatch, SearchedAnswer(reply, [], []), hits=[])

    assert result.answer is None
    assert answerer.calls == 0
