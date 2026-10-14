"""Grounding engine, citation resolution, and retrieval grading nodes (Task 3.4).

Implements:
1. Standardized Source-Prefixed Chunk Framing (docs/DESIGN.md Section 4.1).
2. Strict Grounding System Prompt Assembly.
3. Citation Resolution ([N] -> structured metadata with documentId & filename).
4. Node 3: grade_retrieval_node (evaluates factual relevance, sets 'grounded'/'insufficient').
5. Node 4: generate_grounded_response_node (invokes LLM, maps citations).
6. Node 5: inability_fallback_node (standardized fallback phrasing and empty citations).
"""

import logging
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

from langgraph.types import StreamWriter

from app.core.config import get_settings
from app.core.exceptions import LLMConfigurationError
from app.schemas.chat import ConversationalRAGState
from app.schemas.document import DocumentChunk, ScoredChunk
from app.services.evaluation.usage import observed_completion, observed_text

logger = logging.getLogger(__name__)


def _default_stream_writer(_: Any) -> None:
    """Default no-op StreamWriter fallback when not invoked in streaming mode."""
    return None


FALLBACK_RESPONSE = (
    "Based on the provided document(s), I cannot find information to answer this question."
)

GROUNDING_PROMPT_TEMPLATE = (
    "You are an expert technical assistant analyzing verified document(s).\n"
    "Answer the user's question using ONLY the provided source chunks below.\n\n"
    "Rules:\n"
    "1. Strict Grounding: Every factual statement must be supported directly by "
    "the retrieved context.\n"
    "2. Multi-Document Attribution: When multiple documents are present, explicitly state which "
    'document provides which finding or metric (e.g., "In attention_is_all_you_need.pdf, ... '
    'whereas in llama2.pdf, ...").\n'
    "3. Citations: Reference sources using bracketed numbers corresponding to the chunk index, "
    "e.g., [1] or [2]. Place citations immediately after the supported clause.\n"
    "4. Inability Fallback: If the provided chunks do not contain enough information to answer "
    f'the question, state: "{FALLBACK_RESPONSE}" Do NOT speculate or extrapolate beyond the text.'
    "\n\n"
    "Retrieved Context Chunks:\n"
    "{context_chunks}\n\n"
    "User Question:\n"
    "{standalone_query}"
)

GRADING_PROMPT_TEMPLATE = (
    "You are an evaluation assistant grading document retrieval quality for a question.\n"
    "Given the standalone search query and retrieved document context, determine if the "
    "context contains sufficient factual evidence to answer the question.\n\n"
    "Query:\n{query}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "Evaluation Instruction:\n"
    'Respond with ONLY one word: "grounded" if the context provides relevant factual evidence '
    'to answer the query, or "insufficient" if it does not contain the answer.'
)

_SENTINEL_DEFAULT_KEY = object()


def _extract_chunk_attributes(chunk: Any) -> tuple[str, str, str, int, str, str]:
    """Extract standard attributes from DocumentChunk, ScoredChunk, or dictionary.

    Returns:
        tuple of (chunk_id, document_id, filename, page_number, section_title, content)
    """
    if isinstance(chunk, (DocumentChunk, ScoredChunk)):
        chunk_id = chunk.chunk_id
        meta = chunk.metadata
        document_id = meta.document_id
        filename = meta.filename
        page_number = meta.page_number
        section_title = meta.section_title
        content = chunk.content
    elif isinstance(chunk, dict):
        chunk_id = str(chunk.get("chunk_id", chunk.get("id", "")))
        meta_dict = chunk.get("metadata", {})
        if isinstance(meta_dict, dict):
            document_id = str(meta_dict.get("document_id", chunk.get("document_id", "")))
            filename = str(meta_dict.get("filename", chunk.get("filename", "unknown.pdf")))
            raw_page = meta_dict.get("page_number", chunk.get("page_number", 1))
            page_number = int(raw_page) if raw_page is not None else 1
            section_title = str(meta_dict.get("section_title", chunk.get("section_title", "")))
        else:
            document_id = str(chunk.get("document_id", ""))
            filename = str(chunk.get("filename", "unknown.pdf"))
            raw_page = chunk.get("page_number", 1)
            page_number = int(raw_page) if raw_page is not None else 1
            section_title = str(chunk.get("section_title", ""))
        content = str(chunk.get("content", ""))
    else:
        chunk_id = getattr(chunk, "chunk_id", "")
        document_id = getattr(chunk, "document_id", "")
        filename = getattr(chunk, "filename", "unknown.pdf")
        page_number = getattr(chunk, "page_number", 1)
        section_title = getattr(chunk, "section_title", "")
        content = getattr(chunk, "content", "")

    return chunk_id, document_id, filename, page_number, section_title, content


def format_chunk_header(chunk: Any, index: int) -> str:
    """Format an individual retrieved chunk with standardized document header.

    Conforms to docs/DESIGN.md Section 4.1 specification.

    Args:
        chunk: DocumentChunk, ScoredChunk, or dict representation.
        index: 1-based chunk index for citation referencing.

    Returns:
        Formatted chunk string with metadata prefix.
    """
    _, _, filename, page_number, section_title, content = _extract_chunk_attributes(chunk)
    clean_section = section_title.strip() if section_title and section_title.strip() else "(None)"

    return (
        f"--- CHUNK [{index}] ---\n"
        f"Document: {filename}\n"
        f"Page: {page_number}\n"
        f"Section: {clean_section}\n"
        "Content:\n"
        f"{content.strip()}"
    )


def format_context_chunks(chunks: Sequence[Any]) -> str:
    """Format a sequence of retrieved chunks into framed prompt context blocks.

    Args:
        chunks: Sequence of DocumentChunk, ScoredChunk, or dict objects.

    Returns:
        Consolidated framed context string separated by double newlines.
    """
    if not chunks:
        return "(No relevant document chunks retrieved)"

    formatted_blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        formatted_blocks.append(format_chunk_header(chunk, index=i))

    return "\n\n".join(formatted_blocks)


def build_grounding_prompt(
    context_chunks: str | Sequence[Any],
    standalone_query: str,
) -> str:
    """Assemble the strict grounding system prompt.

    Args:
        context_chunks: Formatted context string or sequence of chunk objects.
        standalone_query: Search query to answer.

    Returns:
        Formatted grounding prompt string.
    """
    if isinstance(context_chunks, str):
        formatted_context = context_chunks.strip()
    else:
        formatted_context = format_context_chunks(context_chunks)

    return GROUNDING_PROMPT_TEMPLATE.format(
        context_chunks=formatted_context,
        standalone_query=standalone_query.strip(),
    )


def resolve_citations(
    response_text: str,
    chunks: Sequence[Any],
) -> list[dict[str, Any]]:
    """Resolve bracketed citation tags ([1], [2]) into structured chunk metadata.

    Deduplicates citations while preserving first appearance order.

    Args:
        response_text: Generated response text containing potential [N] citation tags.
        chunks: The ordered list of retrieved chunks supplied in the context (1-indexed).

    Returns:
        List of structured citation dictionaries with document, page, and excerpt mappings.
    """
    if not response_text or not chunks:
        return []

    pattern = re.compile(r"\[(\d+)\]")
    matches = pattern.findall(response_text)
    if not matches:
        return []

    resolved_citations: list[dict[str, Any]] = []
    seen_indices: set[int] = set()

    for match_str in matches:
        idx = int(match_str)
        if idx in seen_indices:
            continue
        if 1 <= idx <= len(chunks):
            seen_indices.add(idx)
            chunk = chunks[idx - 1]
            (
                chunk_id,
                document_id,
                filename,
                page_number,
                section_title,
                content,
            ) = _extract_chunk_attributes(chunk)

            excerpt = content.strip()[:200]
            if len(content.strip()) > 200:
                excerpt += "..."

            clean_section = section_title.strip() if section_title else "(None)"

            resolved_citations.append(
                {
                    "id": idx,
                    "citationIndex": idx,
                    "chunk_id": chunk_id,
                    "chunkId": chunk_id,
                    "document_id": document_id,
                    "documentId": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "pageNumber": page_number,
                    "page": page_number,
                    "section_title": clean_section,
                    "sectionTitle": clean_section,
                    "quote": excerpt,
                    "sourceText": excerpt,
                }
            )

    return resolved_citations


class RetrievalGrader:
    """Evaluates whether retrieved chunks contain sufficient factual evidence."""

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str | None = None,
        api_key: str | None | object = _SENTINEL_DEFAULT_KEY,
        api_base: str | None = None,
        grading_fn: Callable[[str, Sequence[Any]], Literal["grounded", "insufficient"]]
        | None = None,
    ) -> None:
        """Initialize RetrievalGrader with optional LLM client or custom callable."""
        self._grading_fn = grading_fn
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
        """Return the underlying LLM client or mock."""
        return self._client

    def grade_retrieval(
        self,
        query: str,
        chunks: Sequence[Any],
    ) -> Literal["grounded", "insufficient"]:
        """Grade whether retrieved chunks contain factual evidence relevant to query."""
        if not chunks:
            return "insufficient"

        if self._grading_fn is not None:
            return self._grading_fn(query, chunks)

        if self._client is not None:
            formatted_context = format_context_chunks(chunks)
            prompt = GRADING_PROMPT_TEMPLATE.format(
                query=query.strip(),
                context=formatted_context,
            )
            try:
                if hasattr(self._client, "chat"):
                    completion = observed_completion(
                        self._client,
                        "grading",
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    raw_grade = completion.choices[0].message.content or ""
                elif callable(self._client):
                    raw_grade = observed_text(
                        "grading", self._model, prompt, lambda: str(self._client(prompt))
                    )
                else:
                    raw_grade = "insufficient"

                cleaned = raw_grade.strip().lower()
                if "grounded" in cleaned:
                    return "grounded"
                return "insufficient"
            except Exception as err:
                logger.warning("LLM grading failed, falling back to lexical check: %s", err)

        # Offline heuristic fallback
        query_words = {
            w.lower()
            for w in re.findall(r"\w+", query)
            if len(w) > 2 and w.lower() not in {"what", "when", "where", "which", "how", "does"}
        }
        if not query_words:
            return "grounded"

        total_content = " ".join(_extract_chunk_attributes(c)[5] for c in chunks).lower()
        has_overlap = any(w in total_content for w in query_words)
        return "grounded" if has_overlap else "insufficient"

    def grade_node(self, state: ConversationalRAGState) -> dict[str, Any]:
        """LangGraph node execution method returning retrieval_grade delta."""
        query = state.get("standalone_query", "").strip()
        if not query:
            query = state.get("user_query", "").strip()

        chunks = state.get("retrieved_chunks", [])
        grade = self.grade_retrieval(query=query, chunks=chunks)
        return {"retrieval_grade": grade}


class GroundedGenerator:
    """Synthesizes factual answers strictly grounded in retrieved chunks."""

    def __init__(
        self,
        llm_client: Any | None = None,
        model: str | None = None,
        api_key: str | None | object = _SENTINEL_DEFAULT_KEY,
        api_base: str | None = None,
        generating_fn: Callable[[str, Sequence[Any]], str] | None = None,
        stream_fn: Callable[[str, Sequence[Any]], Iterator[str]] | None = None,
    ) -> None:
        """Initialize GroundedGenerator with optional client or generation callable."""
        self._generating_fn = generating_fn
        self._stream_fn = stream_fn
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
        """Return the underlying LLM client or mock."""
        return self._client

    def generate(
        self,
        standalone_query: str,
        chunks: Sequence[Any],
        writer: StreamWriter | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate answer grounded in chunks and resolve citations.

        Returns:
            tuple of (response_text, resolved_citations)
        """
        prompt = build_grounding_prompt(context_chunks=chunks, standalone_query=standalone_query)

        is_streaming = writer is not None and writer is not _default_stream_writer

        if self._stream_fn is not None:
            collected_tokens: list[str] = []
            for token in self._stream_fn(prompt, chunks):
                collected_tokens.append(token)
                if is_streaming and writer is not None:
                    writer({"token": token})
            response_text = "".join(collected_tokens)
        elif self._generating_fn is not None:
            response_text = self._generating_fn(prompt, chunks)
            if is_streaming and writer is not None and response_text and response_text.strip():
                words = response_text.split(" ")
                for idx, word in enumerate(words):
                    token = word + (" " if idx < len(words) - 1 else "")
                    writer({"token": token})
        elif self._client is not None:
            try:
                if hasattr(self._client, "chat"):
                    if is_streaming and writer is not None:
                        try:
                            stream = observed_completion(
                                self._client,
                                "generation",
                                model=self._model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.0,
                                stream=True,
                            )
                            collected: list[str] = []
                            for chunk in stream:
                                if chunk.choices and chunk.choices[0].delta.content:
                                    tok = chunk.choices[0].delta.content
                                    collected.append(tok)
                                    writer({"token": tok})
                            response_text = "".join(collected)
                        except Exception:
                            # Fallback if injected mock does not support streaming
                            completion = observed_completion(
                                self._client,
                                "generation",
                                model=self._model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.0,
                            )
                            response_text = completion.choices[0].message.content or ""
                            if (
                                is_streaming
                                and writer is not None
                                and response_text
                                and response_text.strip()
                            ):
                                words = response_text.split(" ")
                                for idx, word in enumerate(words):
                                    token = word + (" " if idx < len(words) - 1 else "")
                                    writer({"token": token})
                    else:
                        completion = observed_completion(
                            self._client,
                            "generation",
                            model=self._model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                        )
                        response_text = completion.choices[0].message.content or ""
                elif callable(self._client):
                    response_text = observed_text(
                        "generation", self._model, prompt, lambda: str(self._client(prompt))
                    )
                    if (
                        is_streaming
                        and writer is not None
                        and response_text
                        and response_text.strip()
                    ):
                        words = response_text.split(" ")
                        for idx, word in enumerate(words):
                            token = word + (" " if idx < len(words) - 1 else "")
                            writer({"token": token})
                else:
                    response_text = FALLBACK_RESPONSE
                    if is_streaming and writer is not None:
                        writer({"token": FALLBACK_RESPONSE})
            except Exception as err:
                logger.error("Grounded generation failed: %s", err)
                response_text = FALLBACK_RESPONSE
                if is_streaming and writer is not None:
                    writer({"token": FALLBACK_RESPONSE})
        else:
            # Offline mode without LLM client configured
            if not chunks:
                response_text = FALLBACK_RESPONSE
                if is_streaming and writer is not None:
                    writer({"token": FALLBACK_RESPONSE})
            else:
                raise LLMConfigurationError(
                    "OPENAI_API_KEY is not configured and no LLM client was provided "
                    "for grounded generation."
                )

        if not response_text or not response_text.strip():
            response_text = FALLBACK_RESPONSE
            if is_streaming and writer is not None:
                writer({"token": FALLBACK_RESPONSE})

        citations = resolve_citations(response_text, chunks)
        return response_text, citations

    def generate_node(
        self,
        state: ConversationalRAGState,
        writer: StreamWriter | None = None,
    ) -> dict[str, Any]:
        """LangGraph node execution method returning response_text and citations."""
        query = state.get("standalone_query", "").strip()
        if not query:
            query = state.get("user_query", "").strip()

        chunks = state.get("retrieved_chunks", [])
        response_text, citations = self.generate(
            standalone_query=query,
            chunks=chunks,
            writer=writer,
        )
        output: dict[str, Any] = {
            "response_text": response_text,
            "citations": citations,
        }
        if state.get("retrieval_grade") != "grounded":
            output["retrieval_grade"] = (
                "insufficient" if response_text.strip() == FALLBACK_RESPONSE else "grounded"
            )
        return output


# Global singleton instances for dependency injection in testing
_GLOBAL_GRADER: RetrievalGrader | None = None
_GLOBAL_GENERATOR: GroundedGenerator | None = None
_INIT_LOCK = threading.Lock()


def get_retrieval_grader() -> RetrievalGrader:
    """Return the cached singleton RetrievalGrader instance."""
    global _GLOBAL_GRADER
    if _GLOBAL_GRADER is None:
        with _INIT_LOCK:
            if _GLOBAL_GRADER is None:
                _GLOBAL_GRADER = RetrievalGrader()
    return _GLOBAL_GRADER


def set_retrieval_grader(grader: RetrievalGrader | None) -> None:
    """Set or reset the cached singleton RetrievalGrader instance."""
    global _GLOBAL_GRADER
    with _INIT_LOCK:
        _GLOBAL_GRADER = grader


def get_grounded_generator() -> GroundedGenerator:
    """Return the cached singleton GroundedGenerator instance."""
    global _GLOBAL_GENERATOR
    if _GLOBAL_GENERATOR is None:
        with _INIT_LOCK:
            if _GLOBAL_GENERATOR is None:
                _GLOBAL_GENERATOR = GroundedGenerator()
    return _GLOBAL_GENERATOR


def set_grounded_generator(generator: GroundedGenerator | None) -> None:
    """Set or reset the cached singleton GroundedGenerator instance."""
    global _GLOBAL_GENERATOR
    with _INIT_LOCK:
        _GLOBAL_GENERATOR = generator


# LangGraph pure node functions conforming to state update contracts


def grade_retrieval_node(state: ConversationalRAGState) -> dict[str, Any]:
    """LangGraph node evaluating retrieval quality."""
    return get_retrieval_grader().grade_node(state)


def generate_grounded_response_node(
    state: ConversationalRAGState,
    writer: StreamWriter = _default_stream_writer,
) -> dict[str, Any]:
    """LangGraph node synthesizing response with citations and streaming tokens."""
    return get_grounded_generator().generate_node(state, writer=writer)


def inability_fallback_node(
    state: ConversationalRAGState,
    writer: StreamWriter = _default_stream_writer,
) -> dict[str, Any]:
    """LangGraph node emitting standardized inability fallback message."""
    _ = state  # Acknowledge argument without unused error
    if writer is not _default_stream_writer and writer is not None:
        writer({"token": FALLBACK_RESPONSE})
    return {
        "response_text": FALLBACK_RESPONSE,
        "citations": [],
        "retrieval_grade": "insufficient",
    }
