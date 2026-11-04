"""Settings, loaded from the project's .env file only."""

from pathlib import Path

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from baldur_docs_rag.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# The free tier limit seen in AI Studio is 30,000 tokens per minute.
# The default stays below it because tokens are estimated, not counted.
DEFAULT_EMBEDDING_TOKENS_PER_MINUTE = 24_000
# The free tier allows 100 per minute and counts each text in a batch as one request.
DEFAULT_EMBEDDING_TEXTS_PER_MINUTE = 80


class Settings(BaseSettings):
    # utf-8-sig also accepts the byte order mark some Windows editors add.
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8-sig",
        extra="ignore",
        frozen=True,
    )

    gemini_api_key: SecretStr = Field(min_length=1)
    database_url: str = "postgresql://baldur_rag@127.0.0.1:5433/baldur_rag"
    baldur_repo_path: Path = PROJECT_ROOT.parent / "baldur"
    source_ref: str = Field(default="HEAD", min_length=1)

    embedding_model: str = Field(default="gemini-embedding-2", min_length=1)
    embedding_dimensions: int = Field(default=768, ge=128, le=3072)
    embedding_batch_size: int = Field(default=20, ge=1, le=100)
    embedding_texts_per_minute: int = Field(
        default=DEFAULT_EMBEDDING_TEXTS_PER_MINUTE, gt=0
    )
    embedding_tokens_per_minute: int = Field(
        default=DEFAULT_EMBEDDING_TOKENS_PER_MINUTE, gt=0
    )
    embedding_timeout_ms: int = Field(default=60_000, gt=0)

    chunk_max_chars: int = Field(default=2000, ge=200)
    search_top_k: int = Field(default=5, ge=1, le=50)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # OS environment variables are ignored on purpose. This machine has a
        # GEMINI_API_KEY for a project of unknown billing tier, and only the free
        # project's key in .env may be used. A missing .env therefore fails.
        return (init_settings, dotenv_settings)


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        fields = ", ".join(str(err["loc"][0]) for err in e.errors() if err["loc"])
        raise ConfigError(f"Invalid or missing settings in {ENV_FILE}: {fields}") from e
