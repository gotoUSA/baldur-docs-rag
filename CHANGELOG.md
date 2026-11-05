# Changelog

## Unreleased

### Added

- `python -m baldur_docs_rag ingest`: reads the Baldur documents from one git commit,
  splits them into chunks, embeds them with Gemini and stores them in PostgreSQL (pgvector).
  A stopped run resumes from the batches already saved. On the free tier the first run
  needs two days, because the 1,083 chunks exceed the limit of 1,000 texts per day.
- `python -m baldur_docs_rag search "<question>"`: shows the closest chunks with score,
  file path, heading and the source commit. It refuses to run on an incomplete ingest.
- `python -m baldur_docs_rag ask "<question>"`: answers from the closest chunks with Gemini
  and lists the cited sources with the source commit. It says so when the documents do not
  contain the answer, and fails instead of showing a source that was not retrieved.
- `python -m baldur_docs_rag evaluate`: runs the fixed question set in `eval/questions.json`
  and reports retrieval hit rate, citation match, refusal rates and graded answers, with the
  date, models and source commit. A run stopped by a quota resumes from the saved results.
- `docker-compose.yml`: local PostgreSQL with pgvector, reachable only from this machine.

### Fixed

- A quota stop during `search` no longer says that ingest batches were saved.
