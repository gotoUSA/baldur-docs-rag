"""Real PostgreSQL and real git, with a fake embedder so no API quota is spent."""

import json
import subprocess
import uuid

import psycopg
import pytest
from psycopg import sql

from baldur_docs_rag.answer import ModelAnswer, run_answer
from baldur_docs_rag.config import EVAL_DIR, Settings
from baldur_docs_rag.errors import (
    AnswerQuotaError,
    CorpusNotReadyError,
    EmbeddingQuotaError,
    StoreError,
)
from baldur_docs_rag.evaluation import Ratio, load_questions, run_evaluation
from baldur_docs_rag.ingest import run_ingest
from baldur_docs_rag.search import run_search
from baldur_docs_rag.store import Store

pytestmark = pytest.mark.postgres

ADMIN_URL = "postgresql://baldur_rag@127.0.0.1:5433/baldur_rag"
DIM = 128


class FakeEmbedder:
    """Puts each text on its own axis, so the closest chunk is predictable."""

    def __init__(self, axes, fail_on_call=None):
        self.axes = axes
        self.fail_on_call = fail_on_call
        self.calls = []

    def _vector(self, text):
        vector = [0.0] * DIM
        for word, axis in self.axes.items():
            if word in text:
                vector[axis] = 1.0
                return vector
        vector[DIM - 1] = 1.0
        return vector

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise EmbeddingQuotaError("quota reached")
        return [self._vector(text) for text in texts]

    def embed_query(self, question):
        return self._vector(question)


AXES = {"alpha": 0, "bravo": 1, "charlie": 2, "delta": 3}


@pytest.fixture
def database_url():
    name = f"baldur_rag_test_{uuid.uuid4().hex[:12]}"
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL from docker-compose.yml is not running")
    with admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        yield f"postgresql://baldur_rag@127.0.0.1:5433/{name}"
        admin.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
        )


def git(repo, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


def commit_file(repo, name, text):
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / name).write_text(text, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", f"add {name}")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    for word in AXES:
        commit_file(
            tmp_path, f"{word}.md", f"# {word.title()}\nThis page is about {word}.\n"
        )
    return tmp_path


def make_settings(repo, database_url, **overrides):
    values = {
        "gemini_api_key": "k",
        "database_url": database_url,
        "baldur_repo_path": repo,
        "embedding_dimensions": DIM,
        "embedding_batch_size": 2,
        "search_top_k": 2,
    }
    return Settings(_env_file=None, **(values | overrides))


def no_sleep(_seconds):
    pass


def test_ingest_then_search_returns_the_closest_chunk_first(repo, database_url):
    settings = make_settings(repo, database_url)
    embedder = FakeEmbedder(AXES)

    with Store(database_url) as store:
        result = run_ingest(settings, embedder, store, sleep=no_sleep)
        found = run_search("tell me about charlie", settings, embedder, store)

    assert result.total_chunks == 4
    assert result.embedded_chunks == 4
    assert found.hits[0].source_path == "docs/charlie.md"
    assert found.hits[0].score == pytest.approx(1.0)
    assert found.hits[1].score == pytest.approx(0.0)
    assert found.source_commit == result.source_commit
    assert len(found.hits) == 2


def test_stored_data_survives_a_new_connection(repo, database_url):
    settings = make_settings(repo, database_url)
    embedder = FakeEmbedder(AXES)
    with Store(database_url) as store:
        run_ingest(settings, embedder, store, sleep=no_sleep)

    with Store(database_url) as store:
        found = run_search("delta", settings, embedder, store)

    assert found.hits[0].source_path == "docs/delta.md"


def test_search_is_refused_while_an_ingest_is_incomplete(repo, database_url):
    settings = make_settings(repo, database_url)

    with Store(database_url) as store:
        with pytest.raises(EmbeddingQuotaError):
            run_ingest(
                settings, FakeEmbedder(AXES, fail_on_call=2), store, sleep=no_sleep
            )

        with pytest.raises(CorpusNotReadyError):
            run_search("alpha", settings, FakeEmbedder(AXES), store)


def test_search_before_any_ingest_is_refused(repo, database_url):
    settings = make_settings(repo, database_url)

    with Store(database_url) as store:
        with pytest.raises(CorpusNotReadyError):
            run_search("alpha", settings, FakeEmbedder(AXES), store)


def test_rerun_after_a_quota_stop_embeds_only_the_remaining_chunks(repo, database_url):
    settings = make_settings(repo, database_url)
    with Store(database_url) as store:
        with pytest.raises(EmbeddingQuotaError):
            run_ingest(
                settings, FakeEmbedder(AXES, fail_on_call=2), store, sleep=no_sleep
            )

    resumed = FakeEmbedder(AXES)
    with Store(database_url) as store:
        result = run_ingest(settings, resumed, store, sleep=no_sleep)
        found = run_search("delta", settings, resumed, store)

    assert result.total_chunks == 4
    assert result.embedded_chunks == 2
    assert sum(len(call) for call in resumed.calls) == 2
    assert found.hits[0].source_path == "docs/delta.md"


def test_rerun_of_a_complete_ingest_spends_no_api_calls(repo, database_url):
    settings = make_settings(repo, database_url)
    with Store(database_url) as store:
        run_ingest(settings, FakeEmbedder(AXES), store, sleep=no_sleep)

    second = FakeEmbedder(AXES)
    with Store(database_url) as store:
        result = run_ingest(settings, second, store, sleep=no_sleep)

    assert second.calls == []
    assert result.embedded_chunks == 0


def test_new_commit_rebuilds_so_old_rows_cannot_mix_in(repo, database_url):
    settings = make_settings(repo, database_url)
    with Store(database_url) as store:
        first = run_ingest(settings, FakeEmbedder(AXES), store, sleep=no_sleep)

    commit_file(repo, "alpha.md", "# Alpha\nRewritten page about bravo.\n")
    second_embedder = FakeEmbedder(AXES)
    with Store(database_url) as store:
        second = run_ingest(settings, second_embedder, store, sleep=no_sleep)
        found = run_search("alpha", settings, second_embedder, store)

    assert second.source_commit != first.source_commit
    assert second.embedded_chunks == 4
    assert found.source_commit == second.source_commit


def test_search_with_a_different_model_is_refused(repo, database_url):
    settings = make_settings(repo, database_url)
    other = make_settings(repo, database_url, embedding_model="another-model")

    with Store(database_url) as store:
        run_ingest(settings, FakeEmbedder(AXES), store, sleep=no_sleep)
        with pytest.raises(CorpusNotReadyError, match="another-model"):
            run_search("alpha", other, FakeEmbedder(AXES), store)


def test_changed_dimensions_recreate_the_table(repo, database_url):
    with Store(database_url) as store:
        run_ingest(
            make_settings(repo, database_url), FakeEmbedder(AXES), store, sleep=no_sleep
        )

    settings = make_settings(repo, database_url, embedding_dimensions=256)

    class Wider(FakeEmbedder):
        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 255 for _ in texts]

    with Store(database_url) as store:
        assert (
            run_ingest(settings, Wider(AXES), store, sleep=no_sleep).embedded_chunks
            == 4
        )


def test_pacing_sleeps_between_batches_only(repo, database_url):
    settings = make_settings(repo, database_url)
    naps = []

    with Store(database_url) as store:
        run_ingest(settings, FakeEmbedder(AXES), store, sleep=naps.append)

    assert len(naps) == 1
    assert naps[0] > 0


def test_unreachable_database_is_reported_as_a_store_error():
    with pytest.raises(StoreError):
        Store("postgresql://baldur_rag@127.0.0.1:1/none")


def test_pacing_stays_under_the_per_minute_text_limit(repo, database_url):
    # Real case: the free tier counts each text as one request. Pacing only by tokens
    # reached 100 of 100 per minute in AI Studio, because short texts have few tokens.
    settings = make_settings(repo, database_url)
    naps = []

    with Store(database_url) as store:
        run_ingest(settings, FakeEmbedder(AXES), store, sleep=naps.append)

    free_tier_texts_per_minute = 100
    assert naps[0] >= 60 * settings.embedding_batch_size / free_tier_texts_per_minute


class FixedAnswerer:
    def __init__(self, reply):
        self.reply = reply
        self.questions = []

    def answer(self, question, hits):
        self.questions.append((question, [hit.source_path for hit in hits]))
        return self.reply


def test_answer_cites_a_chunk_stored_in_the_database(repo, database_url):
    settings = make_settings(repo, database_url)
    embedder = FakeEmbedder(AXES)
    answerer = FixedAnswerer(ModelAnswer(answerable=True, answer="x", citations=[1]))

    with Store(database_url) as store:
        ingested = run_ingest(settings, embedder, store, sleep=no_sleep)
        result = run_answer(
            "tell me about charlie", settings, embedder, answerer, store
        )

    assert result.source_commit == ingested.source_commit
    assert result.sources[0].hit.source_path == "docs/charlie.md"
    assert answerer.questions[0][1][0] == "docs/charlie.md"


def test_answer_is_refused_before_any_model_call_while_ingest_is_incomplete(
    repo, database_url
):
    settings = make_settings(repo, database_url)
    answerer = FixedAnswerer(ModelAnswer(answerable=True, answer="x", citations=[1]))

    with Store(database_url) as store:
        with pytest.raises(EmbeddingQuotaError):
            run_ingest(
                settings, FakeEmbedder(AXES, fail_on_call=2), store, sleep=no_sleep
            )
        with pytest.raises(CorpusNotReadyError):
            run_answer("alpha", settings, FakeEmbedder(AXES), answerer, store)

    assert answerer.questions == []


class ScriptedAnswerer:
    """Replies in order. Quota errors are raised, like a stop part way through a run."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def answer(self, question, hits):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def write_questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "authoring": "test",
                "questions": [
                    {
                        "id": "a1",
                        "group": "answerable",
                        "question": "tell me about charlie",
                        "expected_chunks": [
                            {"source_path": "docs/charlie.md", "chunk_index": 0}
                        ],
                    },
                    {
                        "id": "u1",
                        "group": "unrelated",
                        "question": "tell me about delta",
                        "expected_chunks": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def eval_settings(repo, database_url, tmp_path):
    return make_settings(
        repo,
        database_url,
        eval_questions_path=write_questions(tmp_path),
        eval_results_dir=tmp_path / "results",
    )


ANSWER_ONE = ModelAnswer(answerable=True, answer="x", citations=[1])
REFUSE = ModelAnswer(answerable=False, answer="", citations=[])


def test_evaluation_measures_real_retrieval_and_records_its_conditions(
    repo, database_url, tmp_path
):
    settings = eval_settings(repo, database_url, tmp_path)
    embedder = FakeEmbedder(AXES)

    with Store(database_url) as store:
        ingested = run_ingest(settings, embedder, store, sleep=no_sleep)
        report = run_evaluation(
            settings,
            embedder,
            ScriptedAnswerer(ANSWER_ONE, REFUSE),
            store,
            sleep=no_sleep,
        )

    assert report.summary.complete is True
    assert report.summary.retrieval_hit == Ratio(1, 1)
    assert report.summary.citation_match == Ratio(1, 1)
    assert report.summary.refusal_unrelated == Ratio(1, 1)
    assert report.run_info["source_commit"] == ingested.source_commit
    assert report.run_info["answer_model"] == settings.answer_model


def test_stopped_evaluation_resumes_from_the_file_after_a_new_connection(
    repo, database_url, tmp_path
):
    settings = eval_settings(repo, database_url, tmp_path)
    embedder = FakeEmbedder(AXES)

    with Store(database_url) as store:
        run_ingest(settings, embedder, store, sleep=no_sleep)
        stopped = run_evaluation(
            settings,
            embedder,
            ScriptedAnswerer(ANSWER_ONE, AnswerQuotaError("quota reached")),
            store,
            sleep=no_sleep,
        )

    assert "quota reached" in stopped.stopped_reason
    assert stopped.summary.pending_answers == 1
    assert stopped.summary.retrieval_hit == Ratio(1, 1)

    # A new connection and new objects, so only the results file can carry the state.
    resumed_embedder = FakeEmbedder(AXES)
    resumed_answerer = ScriptedAnswerer(REFUSE)
    resumed_embedder.embed_query = None
    with Store(database_url) as store:
        resumed = run_evaluation(
            settings, resumed_embedder, resumed_answerer, store, sleep=no_sleep
        )

    assert resumed_answerer.calls == 1
    assert resumed.summary.complete is True
    assert resumed.run_info == stopped.run_info


def test_evaluation_waits_between_model_calls_only(repo, database_url, tmp_path):
    settings = eval_settings(repo, database_url, tmp_path)
    embedder = FakeEmbedder(AXES)
    naps = []

    with Store(database_url) as store:
        run_ingest(settings, embedder, store, sleep=no_sleep)
        run_evaluation(
            settings,
            embedder,
            ScriptedAnswerer(ANSWER_ONE, REFUSE),
            store,
            sleep=naps.append,
        )

    assert naps == [settings.eval_answer_interval_seconds]


def test_invalid_citation_is_recorded_as_a_rejected_reply(repo, database_url, tmp_path):
    settings = eval_settings(repo, database_url, tmp_path)
    embedder = FakeEmbedder(AXES)
    invalid = ModelAnswer(answerable=True, answer="x", citations=[99])

    with Store(database_url) as store:
        run_ingest(settings, embedder, store, sleep=no_sleep)
        report = run_evaluation(
            settings,
            embedder,
            ScriptedAnswerer(invalid, REFUSE),
            store,
            sleep=no_sleep,
        )

    assert report.summary.rejected_answers == 1
    assert report.summary.citation_match == Ratio(0, 0)


def test_evaluation_is_refused_before_any_ingest(repo, database_url, tmp_path):
    settings = eval_settings(repo, database_url, tmp_path)

    with Store(database_url) as store:
        with pytest.raises(CorpusNotReadyError):
            run_evaluation(
                settings, FakeEmbedder(AXES), ScriptedAnswerer(), store, sleep=no_sleep
            )


def test_every_expected_chunk_in_the_question_file_exists_in_the_database():
    # Skips visibly when the real corpus is not loaded, since nothing can be checked.
    question_set, _ = load_questions(EVAL_DIR / "questions.json")
    try:
        conn = psycopg.connect(ADMIN_URL, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL from docker-compose.yml is not running")
    with conn:
        exists = conn.execute("SELECT to_regclass('chunks')").fetchone()[0]
        if exists is None:
            pytest.skip("The real corpus is not loaded")
        stored = set(
            conn.execute("SELECT source_path, chunk_index FROM chunks").fetchall()
        )

    missing = [
        (question.id, ref.source_path, ref.chunk_index)
        for question in question_set.questions
        for ref in question.expected_chunks
        if (ref.source_path, ref.chunk_index) not in stored
    ]
    assert missing == []
