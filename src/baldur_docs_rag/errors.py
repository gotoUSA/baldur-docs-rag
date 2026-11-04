"""Exceptions raised by this package."""


class BaldurDocsRagError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(BaldurDocsRagError):
    """Settings are missing or invalid."""


class SourceReadError(BaldurDocsRagError):
    """The Baldur documents could not be read from git."""


class EmbeddingError(BaldurDocsRagError):
    """The embedding API failed or returned an unusable response."""


class EmbeddingQuotaError(EmbeddingError):
    """The embedding API rejected the request because a quota was reached."""


class StoreError(BaldurDocsRagError):
    """A PostgreSQL operation failed."""


class CorpusNotReadyError(BaldurDocsRagError):
    """Search was requested before a complete, matching ingest exists."""
