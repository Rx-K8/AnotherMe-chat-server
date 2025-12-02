"""API依存性注入モジュール。

FastAPIの依存性注入システムで使用する依存関係を定義する。
"""

from app.llm.mock import MockLLMProvider
from app.services.chat_server import ChatServe


def get_chat_service() -> ChatServe:
    """ChatServeインスタンスを取得する。

    依存性逆転原則(DIP)に従い、具体的なLLMプロバイダーの実装は
    このファクトリ関数内で解決される。

    Returns:
        ChatServe: チャットサービスインスタンス
    """
    # TODO: 実際のLLMプロバイダーを設定から取得して注入する
    # 例: llm_provider = OpenAIProvider(api_key=settings.openai_api_key)
    llm_provider = MockLLMProvider()
    return ChatServe(llm_provider=llm_provider)
