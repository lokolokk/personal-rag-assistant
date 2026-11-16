"""Bounded evidence assembly; never modifies stored chunks, vectors or citations."""

import logging
from collections.abc import Sequence

import tiktoken

from app.schemas.document import DocumentChunk, ScoredChunk
from app.services.ingestion.structure import continues_introduced_list

logger = logging.getLogger(__name__)


def assemble_continuation_context(
    anchors: Sequence[ScoredChunk],
    available: Sequence[DocumentChunk],
    max_chunks: int,
    max_tokens: int,
) -> list[ScoredChunk]:
    """Prioritize one direct list continuation over lower-ranked, independent anchors.

    Budgets count actual content tokens, not potentially stale metadata estimates.
    Whole chunks retain their original page/ID attribution. Provider admission still
    independently enforces the complete framed prompt and request token limits.
    """
    if max_chunks <= 0 or max_tokens <= 0 or not anchors:
        return []
    encoding = tiktoken.get_encoding("cl100k_base")
    result: list[ScoredChunk] = []
    seen: set[str] = set()
    used = 0

    def add(candidate: ScoredChunk, reason: str) -> bool:
        nonlocal used
        if candidate.chunk_id in seen:
            return False
        tokens = len(encoding.encode(candidate.content, disallowed_special=()))
        admitted = len(result) < max_chunks and used + tokens <= max_tokens
        logger.debug(
            "context id=%s reason=%s admitted=%s tokens=%d",
            candidate.chunk_id,
            reason,
            admitted,
            tokens,
        )
        if not admitted:
            return False
        result.append(candidate)
        seen.add(candidate.chunk_id)
        used += tokens
        return True

    for anchor in anchors:
        if not add(anchor, "anchor"):
            continue
        meta = anchor.metadata
        if meta.is_table or not anchor.content.rstrip().endswith(":"):
            continue
        adjacent = {
            c.chunk_id: c
            for c in available
            if c.chunk_id != anchor.chunk_id
            and c.metadata.document_id == meta.document_id
            and c.metadata.filename == meta.filename
            and c.metadata.page_number == meta.page_number
            and 0 <= c.metadata.char_start - meta.char_end <= 2
        }
        # Ambiguous offsets must not select an arbitrary source.
        if len(adjacent) != 1:
            continue
        following = next(iter(adjacent.values()))
        if following.metadata.is_table or not continues_introduced_list(
            anchor.content, following.content
        ):
            continue
        represented = {c.metadata.document_id for c in result}
        required: dict[str, ScoredChunk] = {}
        for candidate in anchors:
            if candidate.metadata.document_id not in represented:
                required.setdefault(candidate.metadata.document_id, candidate)
        reserved_tokens = sum(
            len(encoding.encode(c.content, disallowed_special=())) for c in required.values()
        )
        following_tokens = len(encoding.encode(following.content, disallowed_special=()))
        if (
            len(result) + 1 + len(required) > max_chunks
            or used + following_tokens + reserved_tokens > max_tokens
        ):
            logger.debug(
                "context id=%s reason=document_coverage admitted=False", following.chunk_id
            )
            continue
        original = next((c for c in anchors if c.chunk_id == following.chunk_id), None)
        add(
            original
            if original is not None
            else ScoredChunk(
                chunk_id=following.chunk_id,
                content=following.content,
                metadata=following.metadata,
                score=0.0,
            ),
            "list_continuation",
        )
    logger.debug(
        "context final_ids=%s content_tokens=%d limit=%d",
        [c.chunk_id for c in result],
        used,
        max_tokens,
    )
    return result
