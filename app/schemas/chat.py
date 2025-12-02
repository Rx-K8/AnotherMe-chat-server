import time
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """チャット会話内のメッセージ。"""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="このメッセージの作成者のロール。"
    )
    content: str = Field(..., description="メッセージの内容。")


class ChatCompletionRequest(BaseModel):
    """チャット完了作成のリクエストボディ。"""

    model: str = Field(..., description="チャット完了に使用するモデルのID。")
    messages: list[Message] = Field(
        ...,
        description="これまでの会話を構成するメッセージのリスト。",
    )
    stream: bool = Field(
        default=False,
        description="設定すると、部分的なメッセージデルタがSSEで送信される。",
    )


class Choice(BaseModel):
    """モデルが生成したチャット完了の選択肢。"""

    index: int = Field(..., description="選択肢リスト内のこの選択肢のインデックス。")
    message: Message = Field(
        ..., description="モデルが生成したチャット完了メッセージ。"
    )
    finish_reason: str | None = Field(
        default="stop",
        description=(
            "モデルがトークン生成を停止した理由。"
            "自然な停止点または指定された停止シーケンスの場合は'stop'、"
            "最大トークン数に達した場合は'length'、"
            "コンテンツフィルターにより省略された場合は'content_filter'。"
        ),
    )


class ChatCompletionResponse(BaseModel):
    """モデルが返すチャット完了レスポンス。"""

    id: str = Field(..., description="チャット完了の一意な識別子。")
    object: str = Field(
        default="chat.completion",
        description="オブジェクトタイプ。常に'chat.completion'。",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="チャット完了が作成されたUnixタイムスタンプ（秒）。",
    )
    model: str = Field(..., description="チャット完了に使用されたモデル。")
    choices: list[Choice] = Field(
        ...,
        description="チャット完了の選択肢リスト。nが1より大きい場合、複数になる。",
    )


class Delta(BaseModel):
    """ストリーミングモデルレスポンスで生成されたチャット完了デルタ。"""

    role: Literal["assistant"] | None = Field(
        default=None, description="このメッセージの作成者のロール。"
    )
    content: str | None = Field(
        default=None,
        description=(
            "チャンクメッセージの内容。"
            "最初のチャンクでは空で、後続のチャンクで値が入る。"
        ),
    )


class ChoiceChunk(BaseModel):
    """ストリーミング中に生成されたチャット完了の選択肢チャンク。"""

    index: int = Field(..., description="選択肢リスト内のこの選択肢のインデックス。")
    delta: Delta = Field(
        ...,
        description="ストリーミングモデルレスポンスで生成されたチャット完了デルタ。",
    )
    finish_reason: str | None = Field(
        default=None,
        description=(
            "モデルがトークン生成を停止した理由。最後のチャンク以外ではnull。"
        ),
    )


class ChatCompletionChunkResponse(BaseModel):
    """ストリーミングされたチャット完了レスポンスのチャンク。"""

    id: str = Field(..., description="チャット完了チャンクの一意な識別子。")
    object: str = Field(
        default="chat.completion.chunk",
        description="オブジェクトタイプ。常に'chat.completion.chunk'。",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="チャット完了チャンクが作成されたUnixタイムスタンプ（秒）。",
    )
    model: str = Field(..., description="チャット完了に使用されたモデル。")
    choices: list[ChoiceChunk] = Field(
        ...,
        description="チャット完了の選択肢リスト。nが1より大きい場合、複数になる。",
    )
