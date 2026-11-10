"""Command line entry point: ingest the documents, search them, or ask a question."""

import argparse
import logging
import sys

from baldur_docs_rag.answer import (
    AnswerResult,
    GeminiAnswerer,
    run_answer_with_search,
)
from baldur_docs_rag.config import load_settings
from baldur_docs_rag.embedding import GeminiEmbedder
from baldur_docs_rag.errors import BaldurDocsRagError, EmbeddingQuotaError
from baldur_docs_rag.evaluation import EvalReport, run_evaluation
from baldur_docs_rag.ingest import run_ingest
from baldur_docs_rag.search import SearchResult, run_search
from baldur_docs_rag.store import Store

PREVIEW_CHARS = 500
NOT_IN_DOCUMENTS = "The documents do not contain an answer to this question."
EXIT_OK = 0
EXIT_FAILED = 1


class _ExtraFormatter(logging.Formatter):
    """Prints the fields passed through a log call's 'extra' argument."""

    _standard = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in record.__dict__.items() if k not in self._standard}
        suffix = "".join(f" {k}={v}" for k, v in fields.items())
        return f"{record.levelname} {record.getMessage()}{suffix}"


def configure_output() -> None:
    # The Windows console default (cp949) cannot encode characters in the documents.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ExtraFormatter())
    package_logger = logging.getLogger("baldur_docs_rag")
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


def format_search_result(result: SearchResult) -> str:
    lines = [f"Documents from baldur commit {result.source_commit}"]
    for rank, hit in enumerate(result.hits, start=1):
        preview = hit.content[:PREVIEW_CHARS]
        if len(hit.content) > PREVIEW_CHARS:
            preview += " ..."
        lines += [
            "",
            f"[{rank}] score={hit.score:.4f} {hit.source_path}#{hit.chunk_index}",
            f"    {hit.heading}",
            preview,
        ]
    return "\n".join(lines)


def format_answer(result: AnswerResult) -> str:
    commit_line = f"Documents from baldur commit {result.source_commit}"
    searched = [f"Searched again for: {query}" for query in result.searches]
    if result.answer is None:
        return "\n".join([NOT_IN_DOCUMENTS, "", commit_line, *searched])
    lines = [result.answer, "", commit_line, *searched]
    for source in result.sources:
        hit = source.hit
        lines.append(
            f"[{source.number}] {hit.source_path}#{hit.chunk_index} {hit.heading}"
        )
    return "\n".join(lines)


def format_evaluation(report: EvalReport) -> str:
    info = report.run_info
    summary = report.summary
    lines = [
        f"Run started {info['started_at']}",
        f"Baldur commit {info['source_commit']}",
        f"Models {info['embedding_model']} and {info['answer_model']}",
        f"Questions {info['question_count']}: {info['authoring']}",
        "",
        f"Retrieval hit rate (top {info['search_top_k']}): {summary.retrieval_hit}",
        f"Citation match: {summary.citation_match}",
        f"Refusal rate, related but unanswerable: {summary.refusal_related}",
        f"Refusal rate, unrelated: {summary.refusal_unrelated}",
        f"Wrong refusal rate, answerable: {summary.wrong_refusal}",
        f"Answers graded correct: {summary.graded_correct}"
        f" ({summary.ungraded_answers} not graded)",
        f"Rejected replies: {summary.rejected_answers}",
        "",
        f"Pending: {summary.pending_retrievals} retrievals,"
        f" {summary.pending_answers} answers",
        f"Results: {report.results_path}",
    ]
    if report.stopped_reason is not None:
        lines.append(
            f"Stopped: {report.stopped_reason}\n"
            "Finished questions are saved. Run the same command later to resume."
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baldur_docs_rag")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="embed the Baldur documents and store them")
    search = commands.add_parser("search", help="show the chunks closest to a question")
    search.add_argument("question")
    ask = commands.add_parser("ask", help="answer a question from the documents")
    ask.add_argument("question")
    commands.add_parser("evaluate", help="measure the fixed question set")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_output()
    try:
        settings = load_settings()
        embedder = GeminiEmbedder(settings)
        with Store(settings.database_url) as store:
            if args.command == "ingest":
                result = run_ingest(settings, embedder, store)
                print(
                    f"Ingest complete: {result.total_chunks} chunks from baldur commit"
                    f" {result.source_commit} ({result.embedded_chunks} embedded in this run)"
                )
            elif args.command == "evaluate":
                report = run_evaluation(
                    settings, embedder, GeminiAnswerer(settings), store
                )
                print(format_evaluation(report))
                # Unfinished numbers must not look like a finished evaluation.
                if not report.summary.complete:
                    return EXIT_FAILED
            elif args.command == "ask":
                print(
                    format_answer(
                        run_answer_with_search(
                            args.question,
                            settings,
                            embedder,
                            GeminiAnswerer(settings),
                            store,
                        )
                    )
                )
            else:
                print(
                    format_search_result(
                        run_search(args.question, settings, embedder, store)
                    )
                )
    except EmbeddingQuotaError as e:
        if args.command != "ingest":
            print(f"Failed: {e}", file=sys.stderr)
            return EXIT_FAILED
        print(
            f"Stopped: {e}\nFinished batches are saved. Run the same command later to resume.",
            file=sys.stderr,
        )
        return EXIT_FAILED
    except BaldurDocsRagError as e:
        print(f"Failed: {e}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK
