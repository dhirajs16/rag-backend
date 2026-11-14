"""Chat-model integrations."""

from rag_backend.integrations.llm.base import ChatMessage, ChatModel, LLMError
from rag_backend.integrations.llm.ollama import OllamaChatModel

__all__ = ["ChatMessage", "ChatModel", "LLMError", "OllamaChatModel"]
