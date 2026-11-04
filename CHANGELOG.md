# Changelog

## Unreleased

### Added

- `python -m baldur_docs_rag ingest`: reads the Baldur documents from one git commit,
  splits them into chunks, embeds them with Gemini and stores them in PostgreSQL (pgvector).
  A stopped run resumes from the batches already saved. On the free tier the first run
  needs two days, because the 1,083 chunks exceed the limit of 1,000 texts per day.
- `python -m baldur_docs_rag search "<question>"`: shows the closest chunks with score,
  file path, heading and the source commit. It refuses to run on an incomplete ingest.
- `docker-compose.yml`: local PostgreSQL with pgvector, reachable only from this machine.
