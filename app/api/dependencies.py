"""API依存性注入モジュール。

FastAPIの依存性注入システムで使用する依存関係を定義する。
"""

from app.core.config import get_settings
from app.llm.abc import LLMProvider
from app.llm.gemma3 import Gemma3Provider
from app.llm.mock import MockLLMProvider
from app.llm.qwen3 import Qwen3Provider
from app.services.chat_server import ChatServe

# グローバルなLLMプロバイダーインスタンス
# サーバー起動時に初期化される
_llm_provider: LLMProvider | None = None


def initialize_llm_provider() -> None:
    """LLMプロバイダーを初期化する。

    サーバー起動時に一度だけ呼び出される。
    モデルのロードはこのタイミングで行われる。
    設定に基づいて使用するプロバイダーを決定する。
    """
    global _llm_provider
    if _llm_provider is None:
        settings = get_settings()
        provider_name = settings.llm_provider.lower()
        model_name = settings.llm_model_name

        if provider_name == "gemma3":
            _llm_provider = (
                Gemma3Provider(model_name) if model_name else Gemma3Provider()
            )
        elif provider_name == "qwen3":
            _llm_provider = Qwen3Provider(model_name) if model_name else Qwen3Provider()
        elif provider_name == "mock":
            _llm_provider = MockLLMProvider()
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Supported providers: gemma3, qwen3, mock"
            )


def get_llm_provider() -> LLMProvider:
    """LLMプロバイダーのインスタンスを取得する。

    Returns:
        LLMProvider: LLMプロバイダーインスタンス

    Raises:
        RuntimeError: プロバイダーが初期化されていない場合
    """
    if _llm_provider is None:
        raise RuntimeError(
            "LLM provider is not initialized. Call initialize_llm_provider() first."
        )
    return _llm_provider


def get_chat_service() -> ChatServe:
    """ChatServeインスタンスを取得する。

    依存性逆転原則(DIP)に従い、具体的なLLMプロバイダーの実装は
    get_llm_provider関数で解決される。

    Returns:
        ChatServe: チャットサービスインスタンス
    """
    return ChatServe(llm_provider=get_llm_provider())
