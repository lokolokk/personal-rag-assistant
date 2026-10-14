"""Multi-turn query reformulator LangGraph node (Task 3.2).

Rewrites conversational user queries containing coreferences (pronouns, relative references)
and multi-document comparative references into self-contained standalone search queries
using the prompt template defined in docs/DESIGN.md Section 3.2.
"""

import threading
from collections.abc import Sequence
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import LLMConfigurationError, QueryReformulationError
from app.schemas.chat import ChatMessage, ConversationalRAGState
from app.services.evaluation.usage import observed_completion, observed_text

REFORMULATION_PROMPT_TEMPLATE = (
    "You are a search query optimizer for a document retrieval system.\n"
    "Given the active documents in this conversation, recent conversation history, and a user's "
    "follow-up question, rewrite the question into a self-contained, standalone search query.\n"
    '- Resolve all pronouns ("it", "they", "that table") and comparative references '
    '("the first paper", "the second method", "both models") into exact entity names '
    "and document filenames from the active documents.\n"
    "- Do NOT answer the question. Only return the reformulated query.\n"
    "- If the question is already self-contained, return it verbatim without modifications.\n\n"
    "Active Documents:\n"
    "{active_document_titles}\n\n"
    "Conversation History:\n"
    "{chat_history}\n\n"
    "User Follow-up:\n"
    "{user_query}\n\n"
    "Standalone Search Query:"
)


_SENTINEL_DEFAULT_KEY = object()


def format_chat_history(messages: Sequence[ChatMessage]) -> str:
    """Format dialogue messages into a readable conversation transcript for prompt injection.

    Args:
        messages: Sequence of ChatMessage objects.

    Returns:
        String with lines formatted as 'Role: Content', or fallback placeholder if empty.
    """
    if not messages:
        return "(No previous conversation history)"
    lines: list[str] = []
    for msg in messages:
        role_display = msg.role.capitalize()
        lines.append(f"{role_display}: {msg.content}")
    return "\n".join(lines)


def build_reformulation_prompt(
    active_document_titles: str,
    chat_history: str | Sequence[ChatMessage],
    user_query: str,
) -> str:
    """Build the query reformulation prompt strictly adhering to docs/DESIGN.md Section 3.2.

    Args:
        active_document_titles: Formatted string of active document filenames.
        chat_history: Formatted history string or sequence of ChatMessage objects.
        user_query: Raw user input.

    Returns:
        Formatted prompt string ready for LLM completion.
    """
    if isinstance(chat_history, str):
        formatted_history = chat_history.strip() or "(No previous conversation history)"
    else:
        formatted_history = format_chat_history(chat_history)

    formatted_titles = active_document_titles.strip() or "(None)"
    formatted_query = user_query.strip()

    return REFORMULATION_PROMPT_TEMPLATE.format(
        active_document_titles=formatted_titles,
        chat_history=formatted_history,
        user_query=formatted_query,
    )


def clean_reformulated_query(raw_query: str, fallback_query: str) -> str:
    """Clean model completion into a normalized standalone search query.

    Strips redundant label echoes, wrapping quotes, and surrounding whitespace.

    Args:
        raw_query: Raw completion text from the LLM.
        fallback_query: Original user query used as fallback if output is empty.

    Returns:
        Cleaned query string.
    """
    cleaned = raw_query.strip()
    if not cleaned:
        return fallback_query.strip()

    lower = cleaned.lower()
    prefixes = [
        "standalone search query:",
        "search query:",
        "reformulated query:",
    ]
    for prefix in prefixes:
        if prefix in lower:
            idx = lower.rfind(prefix)
            cleaned = cleaned[idx + len(prefix) :].strip()
            lower = cleaned.lower()

    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    return cleaned if cleaned else fallback_query.strip()


class QueryReformulator:
    """Multi-turn query reformulation service and LangGraph node runner."""

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str | None = None,
        api_key: str | None | object = _SENTINEL_DEFAULT_KEY,
        api_base: str | None = None,
    ) -> None:
        """Initialize QueryReformulator with optional client or API configuration.

        Args:
            llm_client: Injected OpenAI client instance, mock, or callable for testing.
            model: Model identifier (defaults to settings.chat_model / settings.LLM_MODEL).
            api_key: OpenAI API key (defaults to settings.OPENAI_API_KEY). Pass None to disable.
            api_base: Base URL for OpenAI-compatible endpoint (e.g. Alibaba DashScope).
        """
        settings = get_settings()

        self._model = model or settings.chat_model

        if llm_client is not None:
            self._client = llm_client
        else:
            if api_key is _SENTINEL_DEFAULT_KEY:
                resolved_key = settings.OPENAI_API_KEY
            elif isinstance(api_key, str) and api_key.strip():
                resolved_key = api_key.strip()
            else:
                resolved_key = None

            resolved_base = (
                api_base.strip()
                if isinstance(api_base, str) and api_base.strip()
                else settings.OPENAI_API_BASE
            )

            if resolved_key:
                import openai

                client_kwargs: dict[str, Any] = {"api_key": resolved_key}
                if resolved_base:
                    client_kwargs["base_url"] = resolved_base
                self._client = openai.OpenAI(**client_kwargs)
            else:
                self._client = None

    @property
    def client(self) -> Any | None:
        """Return the underlying LLM client or None."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self._model

    def reformulate(
        self,
        user_query: str,
        chat_history: str | Sequence[ChatMessage] = "",
        active_document_titles: str = "",
    ) -> str:
        """Reformulate a conversational query into a standalone search query.

        Args:
            user_query: Raw user input.
            chat_history: Either formatted history string or sequence of ChatMessage objects.
            active_document_titles: Formatted string of active document filenames.

        Returns:
            The reformulated standalone search query.

        Raises:
            LLMConfigurationError: If reformulation requires LLM call without client/key.
            QueryReformulationError: If the LLM invocation fails.
        """

        trimmed_query = user_query.strip()
        if not trimmed_query:
            return ""

        if isinstance(chat_history, str):
            has_history = (
                bool(chat_history.strip())
                and chat_history.strip() != "(No previous conversation history)"
            )
        else:
            has_history = len(chat_history) > 0

        # Offline pass-through when no client or key is configured and no history exists
        if self._client is None:
            if has_history:
                raise LLMConfigurationError(
                    "OPENAI_API_KEY is not configured and no LLM client was provided "
                    "for multi-turn query reformulation."
                )
            return trimmed_query

        prompt = build_reformulation_prompt(
            active_document_titles=active_document_titles,
            chat_history=chat_history,
            user_query=trimmed_query,
        )

        try:
            if hasattr(self._client, "chat"):
                completion = observed_completion(
                    self._client,
                    "reformulation",
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw_response = completion.choices[0].message.content or ""
            elif callable(self._client):
                raw_response = observed_text(
                    "reformulation", self._model, prompt, lambda: str(self._client(prompt))
                )
            else:
                raw_response = trimmed_query
        except Exception as err:
            raise QueryReformulationError(f"Query reformulation failed: {err}") from err

        return clean_reformulated_query(raw_response, fallback_query=trimmed_query)

    def reformulate_node(self, state: ConversationalRAGState) -> dict[str, str]:
        """LangGraph node execution method for StateGraph integration.

        Accepts state and returns state delta dict updating standalone_query.

        Args:
            state: ConversationalRAGState state dictionary.

        Returns:
            State delta dictionary with {"standalone_query": <reformulated_query>}.
        """
        user_query = state.get("user_query", "")
        active_document_titles = state.get("active_document_titles", "")
        messages = state.get("messages", [])

        standalone_query = self.reformulate(
            user_query=user_query,
            chat_history=messages,
            active_document_titles=active_document_titles,
        )
        return {"standalone_query": standalone_query}


_GLOBAL_REFORMULATOR: QueryReformulator | None = None
_INIT_LOCK = threading.Lock()


def get_query_reformulator() -> QueryReformulator:
    """Return the cached singleton QueryReformulator instance."""
    global _GLOBAL_REFORMULATOR
    if _GLOBAL_REFORMULATOR is None:
        with _INIT_LOCK:
            if _GLOBAL_REFORMULATOR is None:
                _GLOBAL_REFORMULATOR = QueryReformulator()
    return _GLOBAL_REFORMULATOR


def set_query_reformulator(reformulator: QueryReformulator | None) -> None:
    """Set or reset the cached singleton QueryReformulator instance (primarily for tests)."""
    global _GLOBAL_REFORMULATOR
    with _INIT_LOCK:
        _GLOBAL_REFORMULATOR = reformulator


def reformulate_query_node(state: ConversationalRAGState) -> dict[str, str]:
    """Pure LangGraph node function conforming to StateGraph node contract.

    Accepts ConversationalRAGState and returns {"standalone_query": str}.

    Args:
        state: ConversationalRAGState dictionary.

    Returns:
        Dictionary {"standalone_query": reformulated_query}.
    """
    return get_query_reformulator().reformulate_node(state)


def bypass_reformulation_node(state: ConversationalRAGState) -> dict[str, str]:
    """Pure Python LangGraph node that sets standalone_query to user_query without calling LLM.

    Used when conversation history is empty (Turn 1), eliminating prompt token overhead
    and preventing reformulation prompt truncation on initial complex queries.

    Args:
        state: ConversationalRAGState dictionary.

    Returns:
        Dictionary {"standalone_query": user_query}.
    """
    user_query = state.get("user_query", "").strip()
    return {"standalone_query": user_query}
