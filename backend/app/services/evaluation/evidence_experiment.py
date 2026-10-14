"""Local measurements for context ablations over frozen retrieval results."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.services.evaluation.usage import (
    DEFAULT_OUTPUT_CAPS,
    conservative_reservation_tokens,
    estimate_tokens,
)
from app.services.generation.grounding import (
    GRADING_PROMPT_TEMPLATE,
    build_grounding_prompt,
    format_context_chunks,
)

_COMPARISON_STAGES: tuple[Literal["grading", "generation"], ...] = (
    "grading",
    "generation",
)
_TABLE_NUMBER_RE = re.compile(r"\bTable\s+(\d+)\s*:", re.IGNORECASE)
_TABLE_TOPIC_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "column",
        "compare",
        "comparison",
        "different",
        "does",
        "for",
        "from",
        "get",
        "give",
        "how",
        "in",
        "is",
        "it",
        "method",
        "methods",
        "model",
        "models",
        "of",
        "on",
        "or",
        "prompting",
        "result",
        "results",
        "score",
        "scores",
        "setting",
        "show",
        "table",
        "that",
        "the",
        "this",
        "to",
        "two",
        "using",
        "what",
        "with",
    }
)


@dataclass(frozen=True)
class FrozenContextComparison:
    """Text-only context token comparison; values are not provider usage."""

    baseline_chunk_ids: tuple[str, ...]
    candidate_chunk_ids: tuple[str, ...]
    removed_chunk_ids: tuple[str, ...]
    context_stages: tuple[str, ...]
    baseline_context_tokens: int | None
    candidate_context_tokens: int | None
    saved_context_tokens_per_stage: int | None
    saved_context_tokens_across_stages: int | None
    context_reduction_ratio: float | None


def _chunk_id(chunk: Any) -> str:
    if isinstance(chunk, dict):
        value = chunk.get("chunk_id", chunk.get("id"))
    else:
        value = getattr(chunk, "chunk_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError("Every frozen retrieval result must have a non-empty chunk ID")
    return value


def _metadata_value(chunk: Any, key: str) -> Any:
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata", {})
        return metadata.get(key) if isinstance(metadata, dict) else None
    metadata = getattr(chunk, "metadata", None)
    return getattr(metadata, key, None)


def _chunk_content(chunk: Any) -> str:
    if isinstance(chunk, dict):
        content = chunk.get("content", "")
    else:
        content = getattr(chunk, "content", "")
    return content if isinstance(content, str) else ""


def _contains_table_caption(chunk: Any) -> bool:
    return re.search(r"\bTable\s+\d+\s*:", _chunk_content(chunk), re.IGNORECASE) is not None


def _table_numbers(chunk: Any) -> tuple[int, ...]:
    matches = _TABLE_NUMBER_RE.findall(_chunk_content(chunk))
    return tuple(dict.fromkeys(int(match) for match in matches))


def _table_topic_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z]+", text.casefold())
        if len(term) > 1 and term not in _TABLE_TOPIC_STOP_WORDS
    }


def _select_query_relevant_table_context(chunks: Sequence[Any], query: str) -> list[Any] | None:
    """Choose matching table groups, retaining ties and unmapped tables conservatively."""
    groups: dict[tuple[str | None, int | None, int], dict[str, Any]] = {}
    ungrouped_table_ids: set[str] = set()
    table_chunks = [chunk for chunk in chunks if _metadata_value(chunk, "is_table") is True]
    caption_chunks = [chunk for chunk in chunks if _table_numbers(chunk)]

    def group_for(document_id: str | None, page_number: int | None, number: int) -> dict[str, Any]:
        key = (document_id, page_number, number)
        if key not in groups:
            groups[key] = {"chunk_ids": set(), "text_parts": []}
        return groups[key]

    for caption in caption_chunks:
        document_id = _metadata_value(caption, "document_id")
        page_number = _metadata_value(caption, "page_number")
        document_key = document_id if isinstance(document_id, str) else None
        page_key = page_number if isinstance(page_number, int) else None
        for number in _table_numbers(caption):
            group = group_for(document_key, page_key, number)
            group["chunk_ids"].add(_chunk_id(caption))
            group["text_parts"].append(_chunk_content(caption))

    for table_chunk in table_chunks:
        document_id = _metadata_value(table_chunk, "document_id")
        page_number = _metadata_value(table_chunk, "page_number")
        table_start = _metadata_value(table_chunk, "char_start")
        document_key = document_id if isinstance(document_id, str) else None
        page_key = page_number if isinstance(page_number, int) else None
        preceding_captions = [
            caption
            for caption in caption_chunks
            if _metadata_value(caption, "document_id") == document_id
            and _metadata_value(caption, "page_number") == page_number
            and isinstance(table_start, int)
            and isinstance(_metadata_value(caption, "char_start"), int)
            and _metadata_value(caption, "char_start") < table_start
        ]
        if preceding_captions:
            closest = max(preceding_captions, key=lambda item: _metadata_value(item, "char_start"))
            table_numbers = _table_numbers(closest)
        else:
            table_numbers = _table_numbers(table_chunk)

        if not table_numbers:
            ungrouped_table_ids.add(_chunk_id(table_chunk))
            continue
        for number in table_numbers:
            group = group_for(document_key, page_key, number)
            group["chunk_ids"].add(_chunk_id(table_chunk))
            group["text_parts"].append(_chunk_content(table_chunk))

    if not groups:
        return None

    query_table_numbers = set(_table_numbers({"content": query}))
    if query_table_numbers:
        matching_groups = [
            group
            for (document_id, _page_number, number), group in groups.items()
            if number in query_table_numbers
        ]
        if not matching_groups:
            return None
    else:
        query_terms = _table_topic_terms(query)
        if not query_terms:
            return None
        scored_by_document: dict[str | None, list[tuple[int, dict[str, Any]]]] = {}
        for (document_id, _page_number, _number), group in groups.items():
            evidence_terms = _table_topic_terms(" ".join(group["text_parts"]))
            score = len(query_terms & evidence_terms)
            scored_by_document.setdefault(document_id, []).append((score, group))
        matching_groups = []
        for document_groups in scored_by_document.values():
            best_score = max(score for score, _group in document_groups)
            matching_groups.extend(group for score, group in document_groups if score == best_score)
            if best_score == 0:
                return None

    selected_ids = {_chunk_id(chunk) for chunk in chunks[:3]}
    selected_ids.update(ungrouped_table_ids)
    for group in matching_groups:
        selected_ids.update(group["chunk_ids"])
    return [chunk for chunk in chunks if _chunk_id(chunk) in selected_ids]


def select_table_evidence_context(chunks: Sequence[Any], *, query: str | None = None) -> list[Any]:
    """Keep top-ranked text, retrieved tables, and chunks that contain table captions.

    This diagnostic rule uses only retrieval payload metadata/content, never dataset
    annotations. It always retains the top three candidates and augments them with
    retrieved table rows, table captions, and their closest preceding same-page
    caption chunks. The caller must retrieve a sufficiently broad candidate pool.
    """
    if not chunks:
        return []

    if query is not None and query.strip():
        selected = _select_query_relevant_table_context(chunks, query)
        if selected is not None:
            return selected

    table_chunks = [chunk for chunk in chunks if _metadata_value(chunk, "is_table") is True]
    caption_chunks = [chunk for chunk in chunks if _contains_table_caption(chunk)]
    if not table_chunks and not caption_chunks:
        return list(chunks[:3])

    selected_ids = {_chunk_id(chunk) for chunk in chunks[:3]}
    selected_ids.update(_chunk_id(chunk) for chunk in table_chunks)
    selected_ids.update(_chunk_id(chunk) for chunk in caption_chunks)
    for table_chunk in table_chunks:
        document_id = _metadata_value(table_chunk, "document_id")
        page_number = _metadata_value(table_chunk, "page_number")
        table_start = _metadata_value(table_chunk, "char_start")
        if not isinstance(table_start, int):
            continue

        captions = [
            chunk
            for chunk in chunks
            if _metadata_value(chunk, "document_id") == document_id
            and _metadata_value(chunk, "page_number") == page_number
            and isinstance(_metadata_value(chunk, "char_start"), int)
            and _metadata_value(chunk, "char_start") < table_start
            and re.match(r"^\s*Table\s+\d+\s*:", _chunk_content(chunk), re.IGNORECASE)
        ]
        if captions:
            closest_caption = max(captions, key=lambda chunk: _metadata_value(chunk, "char_start"))
            selected_ids.add(_chunk_id(closest_caption))

    return [chunk for chunk in chunks if _chunk_id(chunk) in selected_ids]


def compare_frozen_contexts(
    baseline_chunks: Sequence[Any],
    candidate_chunk_ids: Sequence[str],
    *,
    context_stages: Sequence[str],
) -> FrozenContextComparison:
    """Measure framed context tokens for a subset of unchanged ranked results.

    This function only estimates local cl100k_base text tokens in the context blocks
    produced by the application formatter. It does not estimate provider framing,
    hidden reasoning, answer quality, or total request usage.
    """
    if not baseline_chunks:
        raise ValueError("The frozen baseline retrieval must not be empty")

    baseline_ids = tuple(_chunk_id(chunk) for chunk in baseline_chunks)
    if len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("Frozen baseline chunk IDs must be unique")

    candidate_ids = tuple(candidate_chunk_ids)
    if not candidate_ids:
        raise ValueError("The candidate context must retain at least one chunk")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate chunk IDs must be unique")
    if any(chunk_id not in baseline_ids for chunk_id in candidate_ids):
        raise ValueError("The candidate context may only use frozen baseline chunks")

    ranked_candidate_ids = tuple(chunk_id for chunk_id in baseline_ids if chunk_id in candidate_ids)
    if candidate_ids != ranked_candidate_ids:
        raise ValueError("Candidate chunks must preserve their baseline ranking order")

    stages = tuple(context_stages)
    if not stages or any(not stage.strip() for stage in stages):
        raise ValueError("At least one named context-using stage is required")
    if len(set(stages)) != len(stages):
        raise ValueError("Context stage names must be unique")

    candidate_ids_set = set(candidate_ids)
    candidate_chunks = [
        chunk
        for chunk, chunk_id in zip(baseline_chunks, baseline_ids, strict=True)
        if chunk_id in candidate_ids_set
    ]
    baseline_tokens = estimate_tokens(format_context_chunks(baseline_chunks))
    candidate_tokens = estimate_tokens(format_context_chunks(candidate_chunks))

    if baseline_tokens is None or candidate_tokens is None:
        saved_per_stage = None
        saved_across_stages = None
        reduction_ratio = None
    else:
        saved_per_stage = baseline_tokens - candidate_tokens
        saved_across_stages = saved_per_stage * len(stages)
        reduction_ratio = saved_per_stage / baseline_tokens if baseline_tokens else None

    return FrozenContextComparison(
        baseline_chunk_ids=baseline_ids,
        candidate_chunk_ids=candidate_ids,
        removed_chunk_ids=tuple(
            chunk_id for chunk_id in baseline_ids if chunk_id not in candidate_ids_set
        ),
        context_stages=stages,
        baseline_context_tokens=baseline_tokens,
        candidate_context_tokens=candidate_tokens,
        saved_context_tokens_per_stage=saved_per_stage,
        saved_context_tokens_across_stages=saved_across_stages,
        context_reduction_ratio=reduction_ratio,
    )


def preflight_frozen_response_comparison(
    *,
    query: str,
    baseline_chunks: Sequence[Any],
    candidate_chunk_ids: Sequence[str],
) -> dict[str, Any]:
    """Reserve a paired grader/generator comparison without dispatching providers.

    Both arms use the same fixed query and frozen retrieval results. The prompt token
    counts come from the production prompt builders; reservation settings come from
    the bounded evaluation runner. A local estimate is mandatory so unknown usage is
    never converted to a zero-token reservation.
    """
    from app.core.config import Settings
    from app.services.evaluation.live_runner import (
        FRAMING_TOKENS_PER_ATTEMPT,
        INPUT_MULTIPLIER,
        MAX_ATTEMPTS,
    )

    fixed_query = query.strip()
    if not fixed_query:
        raise ValueError("The frozen response comparison requires a non-empty fixed query")

    comparison = compare_frozen_contexts(
        baseline_chunks,
        candidate_chunk_ids,
        context_stages=_COMPARISON_STAGES,
    )
    if comparison.baseline_context_tokens is None or comparison.candidate_context_tokens is None:
        raise ValueError("Frozen response preflight cannot reserve without a local token estimate")

    candidate_ids = set(comparison.candidate_chunk_ids)
    candidate_chunks = [chunk for chunk in baseline_chunks if _chunk_id(chunk) in candidate_ids]

    def arm_preflight(chunks: Sequence[Any], context_tokens: int) -> dict[str, Any]:
        context = format_context_chunks(chunks)
        prompts = {
            "grading": GRADING_PROMPT_TEMPLATE.format(query=fixed_query, context=context),
            "generation": build_grounding_prompt(context, fixed_query),
        }
        stage_reservations: dict[str, dict[str, int]] = {}
        for stage in _COMPARISON_STAGES:
            prompt_tokens = estimate_tokens(prompts[stage])
            if prompt_tokens is None:
                raise ValueError(
                    "Frozen response preflight cannot reserve without a local token estimate"
                )
            output_cap = DEFAULT_OUTPUT_CAPS[stage]
            reservation = conservative_reservation_tokens(
                prompt_tokens,
                output_cap,
                max_attempts=MAX_ATTEMPTS,
                input_multiplier=INPUT_MULTIPLIER,
                framing_tokens_per_attempt=FRAMING_TOKENS_PER_ATTEMPT,
            )
            stage_reservations[stage] = {
                "prompt_tokens": prompt_tokens,
                "output_cap_tokens": output_cap,
                "reserved_tokens": reservation,
            }
        return {
            "chunk_ids": [_chunk_id(chunk) for chunk in chunks],
            "context_tokens": context_tokens,
            "stage_reservations": stage_reservations,
            "total_reserved_tokens": sum(
                stage["reserved_tokens"] for stage in stage_reservations.values()
            ),
        }

    settings = Settings()
    arms = {
        "baseline": arm_preflight(baseline_chunks, comparison.baseline_context_tokens),
        "candidate": arm_preflight(candidate_chunks, comparison.candidate_context_tokens),
    }
    return {
        "mode": "offline_frozen_response_preflight",
        "query_policy": "same_fixed_query_for_both_arms",
        "chat_model": settings.chat_model,
        "temperature": 0.0,
        "attempts_per_call": MAX_ATTEMPTS,
        "automatic_retries_per_call": MAX_ATTEMPTS - 1,
        "input_multiplier": INPUT_MULTIPLIER,
        "framing_tokens_per_attempt": FRAMING_TOKENS_PER_ATTEMPT,
        "maximum_provider_calls": 2 * len(_COMPARISON_STAGES),
        "arms": arms,
        "total_conservative_reservation_tokens": sum(
            arm["total_reserved_tokens"] for arm in arms.values()
        ),
        "reservation_formula": (
            f"{MAX_ATTEMPTS} attempt(s): {INPUT_MULTIPLIER}x local prompt text estimate + "
            f"{FRAMING_TOKENS_PER_ATTEMPT} framing tokens per attempt + "
            "the grading or generation output cap"
        ),
        "excluded_stages": [
            "reformulation",
            "retrieval",
            "query_embedding",
            "ingestion_embedding",
        ],
        "additional_selection_calls": 0,
        "provider_dispatches": 0,
    }
