"""Measure retrieval and answers against a fixed question set, resuming if stopped."""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from baldur_docs_rag.answer import Answerer, answer_from_hits
from baldur_docs_rag.config import Settings
from baldur_docs_rag.errors import (
    AnswerQuotaError,
    AnswerRejectedError,
    CorpusNotReadyError,
    EmbeddingQuotaError,
    EvaluationError,
)
from baldur_docs_rag.search import QueryEmbedder, SearchResult, run_search
from baldur_docs_rag.store import SearchHit, Store

logger = logging.getLogger(__name__)

SHORT_HASH_CHARS = 8
SHORT_COMMIT_CHARS = 8


class QuestionGroup(StrEnum):
    ANSWERABLE = "answerable"
    RELATED_UNANSWERABLE = "related_unanswerable"
    UNRELATED = "unrelated"


class Outcome(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"
    REJECTED = "rejected"


class Grade(StrEnum):
    CORRECT = "correct"
    WRONG = "wrong"


class ChunkRef(BaseModel, frozen=True):
    source_path: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)


class EvalQuestion(BaseModel):
    id: str = Field(min_length=1)
    group: QuestionGroup
    question: str = Field(min_length=1)
    expected_chunks: list[ChunkRef]

    @model_validator(mode="after")
    def _expected_chunks_match_the_group(self) -> Self:
        answerable = self.group is QuestionGroup.ANSWERABLE
        if answerable != bool(self.expected_chunks):
            raise ValueError(
                f"{self.id}: only answerable questions list expected chunks,"
                " and each of them must"
            )
        return self


class QuestionSet(BaseModel):
    authoring: str = Field(min_length=1)
    questions: list[EvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        ids = [question.id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question ids must be unique")
        return self


def load_questions(path: Path) -> tuple[QuestionSet, str]:
    """Return the questions and a hash, so changed questions never reuse old results."""
    try:
        raw = path.read_bytes()
        return QuestionSet.model_validate_json(raw), hashlib.sha256(raw).hexdigest()
    except (OSError, ValidationError) as e:
        raise EvaluationError(f"Cannot read the questions in {path}") from e


@dataclass(frozen=True)
class AnswerRecord:
    outcome: Outcome
    answer: str | None
    citations: list[int]
    detail: str | None = None


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int

    def __str__(self) -> str:
        # Zero measured questions must not read as a score of zero or a pass.
        if self.denominator == 0:
            return "not measured"
        percent = 100 * self.numerator / self.denominator
        return f"{self.numerator}/{self.denominator} ({percent:.1f}%)"


@dataclass(frozen=True)
class EvalSummary:
    retrieval_hit: Ratio
    citation_match: Ratio
    refusal_related: Ratio
    refusal_unrelated: Ratio
    wrong_refusal: Ratio
    graded_correct: Ratio
    ungraded_answers: int
    rejected_answers: int
    pending_retrievals: int
    pending_answers: int

    @property
    def complete(self) -> bool:
        return self.pending_retrievals == 0 and self.pending_answers == 0


@dataclass(frozen=True)
class EvalReport:
    results_path: Path
    run_info: dict[str, object]
    summary: EvalSummary
    stopped_reason: str | None


def _refusal_ratio(
    questions: Sequence[EvalQuestion],
    answers: dict[str, AnswerRecord],
    group: QuestionGroup,
) -> Ratio:
    measured = [q for q in questions if q.group is group and q.id in answers]
    refused = [q for q in measured if answers[q.id].outcome is Outcome.REFUSED]
    return Ratio(len(refused), len(measured))


def summarize(
    questions: Sequence[EvalQuestion],
    retrievals: dict[str, list[SearchHit]],
    answers: dict[str, AnswerRecord],
    grades: dict[str, Grade],
) -> EvalSummary:
    answerable = [q for q in questions if q.group is QuestionGroup.ANSWERABLE]

    retrieved = [q for q in answerable if q.id in retrievals]
    retrieval_hits = 0
    for question in retrieved:
        found = {
            ChunkRef(source_path=hit.source_path, chunk_index=hit.chunk_index)
            for hit in retrievals[question.id]
        }
        retrieval_hits += bool(found.intersection(question.expected_chunks))

    answered = [
        q
        for q in answerable
        if q.id in answers and answers[q.id].outcome is Outcome.ANSWERED
    ]
    citation_matches = 0
    for question in answered:
        hits = retrievals[question.id]
        cited = {
            ChunkRef(
                source_path=hits[number - 1].source_path,
                chunk_index=hits[number - 1].chunk_index,
            )
            for number in answers[question.id].citations
        }
        citation_matches += bool(cited.intersection(question.expected_chunks))

    graded = [q for q in answered if q.id in grades]
    correct = [q for q in graded if grades[q.id] is Grade.CORRECT]

    return EvalSummary(
        retrieval_hit=Ratio(retrieval_hits, len(retrieved)),
        citation_match=Ratio(citation_matches, len(answered)),
        refusal_related=_refusal_ratio(
            questions, answers, QuestionGroup.RELATED_UNANSWERABLE
        ),
        refusal_unrelated=_refusal_ratio(questions, answers, QuestionGroup.UNRELATED),
        wrong_refusal=_refusal_ratio(questions, answers, QuestionGroup.ANSWERABLE),
        graded_correct=Ratio(len(correct), len(graded)),
        ungraded_answers=len(answered) - len(graded),
        rejected_answers=sum(
            record.outcome is Outcome.REJECTED for record in answers.values()
        ),
        pending_retrievals=sum(q.id not in retrievals for q in questions),
        pending_answers=sum(q.id not in answers for q in questions),
    )


class _ResultsFile:
    """An append-only JSONL file, flushed per line so a stop keeps every finished step."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_info: dict[str, object] | None = None
        self.retrievals: dict[str, list[SearchHit]] = {}
        self.answers: dict[str, AnswerRecord] = {}
        if path.exists():
            self._read()

    def _read(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                self._apply(json.loads(line))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise EvaluationError(f"Cannot read the results in {self.path}") from e

    def _apply(self, record: dict) -> None:
        if record["kind"] == "run":
            self.run_info = {k: v for k, v in record.items() if k != "kind"}
        elif record["kind"] == "retrieval":
            self.retrievals[record["id"]] = [SearchHit(**h) for h in record["hits"]]
        elif record["kind"] == "answer":
            self.answers[record["id"]] = AnswerRecord(
                outcome=Outcome(record["outcome"]),
                answer=record["answer"],
                citations=record["citations"],
                detail=record["detail"],
            )

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            raise EvaluationError(f"Cannot write the results to {self.path}") from e

    def add_run_info(self, run_info: dict[str, object]) -> None:
        self._append({"kind": "run", **run_info})
        self.run_info = run_info

    def add_retrieval(self, question_id: str, hits: Sequence[SearchHit]) -> None:
        self._append(
            {
                "kind": "retrieval",
                "id": question_id,
                "hits": [asdict(hit) for hit in hits],
            }
        )
        self.retrievals[question_id] = list(hits)

    def add_answer(self, question_id: str, record: AnswerRecord) -> None:
        self._append({"kind": "answer", "id": question_id, **asdict(record)})
        self.answers[question_id] = record


def results_path(settings: Settings, source_commit: str, questions_hash: str) -> Path:
    name = (
        f"{source_commit[:SHORT_COMMIT_CHARS]}-{settings.answer_model}"
        f"-{questions_hash[:SHORT_HASH_CHARS]}.jsonl"
    )
    return settings.eval_results_dir / name


def load_grades(results_file: Path) -> dict[str, Grade]:
    """Read the person's grades kept beside the results they judge. None yet is fine."""
    grades_file = results_file.with_suffix(".grades.json")
    if not grades_file.exists():
        return {}
    try:
        raw = json.loads(grades_file.read_text(encoding="utf-8"))
        return {question_id: Grade(value) for question_id, value in raw.items()}
    except (OSError, ValueError, AttributeError) as e:
        raise EvaluationError(f"Cannot read the grades in {grades_file}") from e


def _answer_record(
    question: EvalQuestion,
    hits: list[SearchHit],
    source_commit: str,
    answerer: Answerer,
) -> AnswerRecord:
    found = SearchResult(source_commit, hits)
    try:
        result = answer_from_hits(question.question, found, answerer)
    except AnswerRejectedError as e:
        # An unusable reply is a measured result about the model, not a broken run.
        return AnswerRecord(Outcome.REJECTED, None, [], detail=str(e))
    if result.answer is None:
        return AnswerRecord(Outcome.REFUSED, None, [])
    return AnswerRecord(
        Outcome.ANSWERED,
        result.answer,
        [source.number for source in result.sources],
    )


def run_evaluation(
    settings: Settings,
    embedder: QueryEmbedder,
    answerer: Answerer,
    store: Store,
    sleep: Callable[[float], None] = time.sleep,
) -> EvalReport:
    question_set, questions_hash = load_questions(settings.eval_questions_path)
    questions = question_set.questions

    state = store.read_state()
    if state is None or state.completed_at is None:
        raise CorpusNotReadyError(
            "No complete ingest found. Run the ingest command first"
        )

    results = _ResultsFile(results_path(settings, state.source_commit, questions_hash))
    if results.run_info is None:
        results.add_run_info(
            {
                "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source_commit": state.source_commit,
                "embedding_model": state.embedding_model,
                "embedding_dimensions": state.embedding_dimensions,
                "answer_model": settings.answer_model,
                "search_top_k": settings.search_top_k,
                "questions_sha256": questions_hash,
                "question_count": len(questions),
                "authoring": question_set.authoring,
            }
        )

    stopped_reason = None
    try:
        # All retrievals come first: they cost no answer quota, so a stop during the
        # answers still leaves the retrieval numbers complete.
        for question in questions:
            if question.id not in results.retrievals:
                found = run_search(question.question, settings, embedder, store)
                results.add_retrieval(question.id, found.hits)

        called_model = False
        for question in questions:
            if question.id in results.answers:
                continue
            if called_model:
                sleep(settings.eval_answer_interval_seconds)
            called_model = True
            record = _answer_record(
                question,
                results.retrievals[question.id],
                state.source_commit,
                answerer,
            )
            results.add_answer(question.id, record)
            logger.info(
                "evaluation.progress_updated",
                extra={"done": len(results.answers), "total": len(questions)},
            )
    except (EmbeddingQuotaError, AnswerQuotaError) as e:
        logger.warning("evaluation.quota_exhausted", extra={"error": type(e).__name__})
        stopped_reason = str(e)

    grades = load_grades(results.path)
    known_ids = {question.id for question in questions}
    unknown = sorted(set(grades) - known_ids)
    if unknown:
        raise EvaluationError(f"Grades name unknown questions: {unknown}")

    return EvalReport(
        results_path=results.path,
        run_info=results.run_info,
        summary=summarize(questions, results.retrievals, results.answers, grades),
        stopped_reason=stopped_reason,
    )
