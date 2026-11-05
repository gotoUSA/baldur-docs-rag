import pytest
from pydantic import ValidationError

from baldur_docs_rag import config
from baldur_docs_rag.config import Settings, load_settings
from baldur_docs_rag.errors import ConfigError


@pytest.fixture
def os_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key-from-os-environment")


def test_env_file_wins_over_the_os_environment(tmp_path, os_key):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=key-from-env-file\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.gemini_api_key.get_secret_value() == "key-from-env-file"


def test_os_environment_key_alone_is_rejected(tmp_path, os_key):
    with pytest.raises(ValidationError):
        Settings(_env_file=tmp_path / "missing.env")


def test_env_file_with_a_byte_order_mark_is_read(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xef\xbb\xbfGEMINI_API_KEY=key-with-bom\n")

    settings = Settings(_env_file=env_file)

    assert settings.gemini_api_key.get_secret_value() == "key-with-bom"


@pytest.mark.parametrize(
    "field, value",
    [
        ("embedding_dimensions", 100),
        ("embedding_dimensions", 4000),
        ("embedding_batch_size", 0),
        ("embedding_tokens_per_minute", 0),
        ("embedding_timeout_ms", 0),
        ("chunk_max_chars", 10),
        ("search_top_k", 0),
        ("answer_model", ""),
        ("answer_timeout_seconds", 0),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, field, value):
    with pytest.raises(ValidationError):
        Settings(
            _env_file=tmp_path / "missing.env", gemini_api_key="k", **{field: value}
        )


def test_load_settings_reports_the_field_without_the_key_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=real-looking-key\nSEARCH_TOP_K=0\n", encoding="utf-8"
    )
    monkeypatch.setitem(Settings.model_config, "env_file", env_file)
    monkeypatch.setattr(config, "ENV_FILE", env_file)

    with pytest.raises(ConfigError) as caught:
        load_settings()

    assert "search_top_k" in str(caught.value)
    assert "real-looking-key" not in str(caught.value)
