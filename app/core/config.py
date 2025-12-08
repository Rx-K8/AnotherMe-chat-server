from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AnotherMe Chat Server"
    llm_provider: str = "gemma3"  # "gemma3", "qwen3", "mock"
    llm_model_name: str | None = None  # モデル名（Noneの場合はデフォルト値を使用）


@lru_cache
def get_settings() -> Settings:
    return Settings()
