import json

import pytest

from baldur_docs_rag.config import EVAL_DIR, Settings
from baldur_docs_rag.errors import EvaluationError
from baldur_docs_rag.evaluation import (
    AnswerRecord,
    ChunkRef,
    EvalQuestion,
    Grade,
    Outcome,
    QuestionGroup,
    QuestionSet,
    Ratio,
    load_grades,
    load_questions,
    results_path,
    summarize,
)
from baldur_docs_rag.store import SearchHit


def answerable(question_id, path="docs/a.md", index=0):
    return EvalQuestion(
        id=question_id,
        group=QuestionGroup.ANSWERABLE,
        question="q",
        expected_chunks=[ChunkRef(source_path=path, chunk_index=index)],
    )


def unanswerable(question_id, group):
    return EvalQuestion(id=question_id, group=group, question="q", expected_chunks=[])


def hit(path="docs/a.md", index=0):
    return SearchHit(path, index, "H", "body", 0.7)


def answered(*citations):
    return AnswerRecord(Outcome.ANSWERED, "text", list(citations))


REFUSED = AnswerRecord(Outcome.REFUSED, None, [])
REJECTED = AnswerRecord(Outcome.REJECTED, None, [], detail="bad number")


def test_empty_ratio_reads_as_not_measured_instead_of_zero():
    assert str(Ratio(0, 0)) == "not measured"
    assert str(Ratio(1, 3)) == "1/3 (33.3%)"


def test_retrieval_hit_counts_only_answerable_questions():
    questions = [
        answerable("a1"),
        answerable("a2"),
        unanswerable("r1", QuestionGroup.RELATED_UNANSWERABLE),
    ]
    retrievals = {
        "a1": [hit(), hit("docs/b.md")],
        "a2": [hit("docs/c.md")],
        "r1": [hit()],
    }

    summary = summarize(questions, retrievals, {}, {})

    assert summary.retrieval_hit == Ratio(1, 2)


def test_same_file_with_another_chunk_is_not_a_retrieval_hit():
    summary = summarize([answerable("a1", index=3)], {"a1": [hit(index=4)]}, {}, {})

    assert summary.retrieval_hit == Ratio(0, 1)


def test_citation_match_reads_the_chunk_behind_the_cited_number():
    questions = [answerable("a1"), answerable("a2")]
    retrievals = {
        "a1": [hit("docs/other.md"), hit()],
        "a2": [hit(), hit("docs/other.md")],
    }
    answers = {"a1": answered(2), "a2": answered(2)}

    summary = summarize(questions, retrievals, answers, {})

    assert summary.citation_match == Ratio(1, 2)


def test_refused_answerable_question_is_a_wrong_refusal_not_a_citation_miss():
    questions = [answerable("a1"), answerable("a2")]
    retrievals = {"a1": [hit()], "a2": [hit()]}
    answers = {"a1": REFUSED, "a2": answered(1)}

    summary = summarize(questions, retrievals, answers, {})

    assert summary.wrong_refusal == Ratio(1, 2)
    assert summary.citation_match == Ratio(1, 1)


def test_refusal_rates_are_kept_apart_by_group():
    related = QuestionGroup.RELATED_UNANSWERABLE
    questions = [
        unanswerable("r1", related),
        unanswerable("r2", related),
        unanswerable("u1", QuestionGroup.UNRELATED),
    ]
    answers = {"r1": REFUSED, "r2": answered(1), "u1": REFUSED}

    summary = summarize(questions, {}, answers, {})

    assert summary.refusal_related == Ratio(1, 2)
    assert summary.refusal_unrelated == Ratio(1, 1)


def test_rejected_reply_is_counted_and_is_not_a_refusal():
    questions = [unanswerable("r1", QuestionGroup.RELATED_UNANSWERABLE)]

    summary = summarize(questions, {}, {"r1": REJECTED}, {})

    assert summary.rejected_answers == 1
    assert summary.refusal_related == Ratio(0, 1)


def test_unfinished_questions_stay_visible_as_pending():
    questions = [answerable("a1"), answerable("a2")]

    summary = summarize(questions, {"a1": [hit()]}, {}, {})

    assert summary.pending_retrievals == 1
    assert summary.pending_answers == 2
    assert summary.complete is False
    assert str(summary.citation_match) == "not measured"


def test_grades_count_only_answered_questions_and_report_the_rest():
    questions = [answerable("a1"), answerable("a2"), answerable("a3")]
    retrievals = {"a1": [hit()], "a2": [hit()], "a3": [hit()]}
    answers = {"a1": answered(1), "a2": answered(1), "a3": REFUSED}

    summary = summarize(questions, retrievals, answers, {"a1": Grade.WRONG})

    assert summary.graded_correct == Ratio(0, 1)
    assert summary.ungraded_answers == 1


@pytest.mark.parametrize(
    "group, expected",
    [
        (QuestionGroup.ANSWERABLE, []),
        (
            QuestionGroup.UNRELATED,
            [ChunkRef(source_path="docs/a.md", chunk_index=0)],
        ),
    ],
)
def test_expected_chunks_must_match_the_group(group, expected):
    with pytest.raises(ValueError, match="expected chunks"):
        EvalQuestion(id="x", group=group, question="q", expected_chunks=expected)


def test_duplicate_question_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        QuestionSet(authoring="a", questions=[answerable("a1"), answerable("a1")])


def test_missing_question_file_is_reported(tmp_path):
    with pytest.raises(EvaluationError, match="Cannot read"):
        load_questions(tmp_path / "missing.json")


def test_changed_questions_get_a_new_results_file():
    settings = Settings(_env_file=None, gemini_api_key="k")

    first = results_path(settings, "20a2f7f3ecf4", "aaaaaaaaaaaa")
    second = results_path(settings, "20a2f7f3ecf4", "bbbbbbbbbbbb")

    assert first != second
    assert first.name.startswith("20a2f7f3-")


def test_grades_are_read_from_beside_the_results(tmp_path):
    results = tmp_path / "run.jsonl"
    (tmp_path / "run.grades.json").write_text(
        json.dumps({"a01": "correct", "a02": "wrong"}), encoding="utf-8"
    )

    assert load_grades(results) == {"a01": Grade.CORRECT, "a02": Grade.WRONG}


def test_no_grades_yet_is_not_an_error(tmp_path):
    assert load_grades(tmp_path / "run.jsonl") == {}


def test_unknown_grade_value_is_reported(tmp_path):
    (tmp_path / "run.grades.json").write_text(
        json.dumps({"a01": "maybe"}), encoding="utf-8"
    )

    with pytest.raises(EvaluationError, match="grades"):
        load_grades(tmp_path / "run.jsonl")


def test_the_real_question_file_has_the_agreed_shape():
    question_set, _ = load_questions(EVAL_DIR / "questions.json")
    groups = [question.group for question in question_set.questions]

    assert len(groups) == 30
    assert groups.count(QuestionGroup.ANSWERABLE) == 18
    assert groups.count(QuestionGroup.RELATED_UNANSWERABLE) == 8
    assert groups.count(QuestionGroup.UNRELATED) == 4
