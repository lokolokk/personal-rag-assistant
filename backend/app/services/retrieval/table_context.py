"""Promote clearly relevant retrieved tables into the bounded evidence context."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import tiktoken

from app.schemas.document import DocumentChunk, ScoredChunk

_CAPTION_RE = re.compile(r"^[ \t]*Table[ \t]+(\d+)[ \t]*:", re.IGNORECASE | re.MULTILINE)
_TABLE_REFERENCE_RE = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)
_MAX_NEIGHBOR_GAP = 1200
_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.IGNORECASE)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "get",
        "give",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "show",
        "table",
        "that",
        "the",
        "this",
        "to",
        "what",
        "which",
        "with",
    }
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TableGroup:
    document_id: str
    page_number: int
    table_number: int | None
    table: ScoredChunk
    caption: DocumentChunk | None
    header: DocumentChunk | None
    overlap: int
    rank: int


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in _TOKEN_RE.findall(text.casefold()):
        term = raw
        if term in _STOP_WORDS:
            continue
        if term.endswith("ies") and len(term) > 4:
            term = term[:-3] + "y"
        elif term.endswith("ing") and len(term) > 5:
            term = term[:-3]
        elif term.endswith("ed") and len(term) > 4:
            term = term[:-2]
        elif term.endswith("s") and len(term) > 3:
            term = term[:-1]
        terms.add(term)
    return terms


def _page_neighbors(table: ScoredChunk, available: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    return sorted(
        [
            chunk
            for chunk in available
            if chunk.chunk_id != table.chunk_id
            and chunk.metadata.document_id == table.metadata.document_id
            and chunk.metadata.filename == table.metadata.filename
            and chunk.metadata.page_number == table.metadata.page_number
        ],
        key=lambda chunk: (chunk.metadata.char_start, chunk.chunk_id),
    )


def _table_tail(text: str) -> bool:
    """Recognize short row fragments, not arbitrary intervening paragraphs."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (
        bool(lines)
        and all(len(line) <= 80 and len(line.split()) <= 6 for line in lines)
        and any(re.fullmatch(r"[\d.,%+−–—\-() ]+", line) for line in lines)
        and all(
            re.fullmatch(r"[\d.,%+−–—\-() ]+", line)
            or re.fullmatch(r"[\w /()+−–—\-]+", line)
            or re.fullmatch(r"[A-Za-z]{1,12}\.", line)
            for line in lines
        )
    )


def _caption_for(table: ScoredChunk, available: Sequence[DocumentChunk]) -> DocumentChunk | None:
    neighbors = _page_neighbors(table, available)
    candidates: list[DocumentChunk] = []
    for chunk in neighbors:
        matches = list(_CAPTION_RE.finditer(chunk.content))
        if chunk.metadata.is_table or len(matches) != 1:
            continue
        # An embedded caption may follow extracted row values. Do not jump over prose.
        prefix = chunk.content[: matches[0].start()].strip()
        if prefix and not _table_tail(prefix):
            continue
        before = chunk.metadata.char_start < table.metadata.char_start
        gap = (
            table.metadata.char_start - chunk.metadata.char_end
            if before
            else chunk.metadata.char_start - table.metadata.char_end
        )
        if gap < -1 or gap > _MAX_NEIGHBOR_GAP:
            continue
        between = [
            other
            for other in neighbors
            if min(chunk.metadata.char_start, table.metadata.char_start)
            < other.metadata.char_start
            < max(chunk.metadata.char_start, table.metadata.char_start)
        ]
        if any(
            _CAPTION_RE.search(other.content)
            or not (other.metadata.is_table or _table_tail(other.content))
            for other in between
        ):
            continue
        # Never link across a separate table with a different column layout.
        if any(
            other.metadata.is_table
            and other.content.partition("\n")[0] != table.content.partition("\n")[0]
            for other in between
        ):
            continue
        candidates.append(chunk)
    # Competing captions (including repeated numbers) are not resolved by guessing.
    return candidates[0] if len(candidates) == 1 else None


def _header_for(table: ScoredChunk, available: Sequence[DocumentChunk]) -> DocumentChunk | None:
    """Retain an adjacent source header when its column sequence matches the table."""
    rows = [line for line in table.content.splitlines() if line.strip().startswith("|")]
    if not rows:
        return None
    columns = [cell.strip().casefold() for cell in rows[0].strip().strip("|").split("|")][1:]
    if len(columns) < 2 or not all(columns):
        return None
    preceding = [
        chunk
        for chunk in _page_neighbors(table, available)
        if 0 <= table.metadata.char_start - chunk.metadata.char_end <= 2
        and not chunk.metadata.is_table
        and _CAPTION_RE.search(chunk.content) is None
    ]
    if len(preceding) != 1:
        return None
    header = preceding[0]
    lines = [line.strip().casefold() for line in header.content.splitlines() if line.strip()]
    # Exact ordered labels avoid treating a passing mention of a metric as a header.
    for start in range(len(lines) - len(columns) + 1):
        if lines[start : start + len(columns)] == columns:
            return header
    return None


def _groups(
    query: str, ranked_candidates: Sequence[ScoredChunk], available: Sequence[DocumentChunk]
) -> list[_TableGroup]:
    query_terms = _terms(query)
    if not query_terms:
        return []

    requested_numbers = {int(number) for number in _TABLE_REFERENCE_RE.findall(query)}
    grouped: dict[
        tuple[str, int, str],
        list[tuple[int, ScoredChunk, DocumentChunk | None, int | None]],
    ] = {}
    for rank, table in enumerate(ranked_candidates, start=1):
        if not table.metadata.is_table:
            continue
        caption = _caption_for(table, available)
        number_match = _CAPTION_RE.search(caption.content) if caption is not None else None
        if number_match is None:
            number_match = _CAPTION_RE.match(table.content)
        number = int(number_match.group(1)) if number_match is not None else None
        group_id = caption.chunk_id if caption is not None else f"chunk:{table.chunk_id}"
        grouped.setdefault(
            (table.metadata.document_id, table.metadata.page_number, group_id), []
        ).append((rank, table, caption, number))

    result: list[_TableGroup] = []
    for (document_id, page_number, _group_id), chunks in grouped.items():
        _, _, caption, number = min(chunks, key=lambda item: item[0])
        rank, table, _, _ = max(
            chunks,
            key=lambda item: (len(query_terms & _terms(item[1].content)), -item[0]),
        )
        if requested_numbers and number not in requested_numbers:
            continue
        text = " ".join(
            [chunk.content for _, chunk, _, _ in chunks]
            + ([caption.content] if caption is not None else [])
        )
        overlap = len(query_terms & _terms(text))
        if requested_numbers and number in requested_numbers:
            overlap += len(query_terms) + 1
        minimum_overlap = min(2, len(query_terms))
        if overlap >= minimum_overlap:
            result.append(
                _TableGroup(
                    document_id,
                    page_number,
                    number,
                    table,
                    caption,
                    _header_for(table, available),
                    overlap,
                    rank,
                )
            )

    if requested_numbers:
        return sorted(result, key=lambda group: (-group.overlap, group.rank))

    best_by_document: dict[str, _TableGroup] = {}
    for group in result:
        current = best_by_document.get(group.document_id)
        if current is None or (group.overlap, -group.rank) > (current.overlap, -current.rank):
            best_by_document[group.document_id] = group
    return sorted(best_by_document.values(), key=lambda group: (-group.overlap, group.rank))


def prioritize_table_context(
    query: str,
    ranked_candidates: Sequence[ScoredChunk],
    available: Sequence[DocumentChunk],
    max_chunks: int,
    max_tokens: int | None = None,
) -> list[ScoredChunk]:
    """Keep one top hit per represented document and promote matching table evidence.

    A table found by either dense or sparse search can survive rank fusion's final
    cutoff. Unambiguous same-page captions and adjacent source column headers are
    kept in source order, with unchanged content and citation metadata.
    Known evidence groups must fit the supplied content-token budget together.
    The caller still applies its final chunk and token budget to the candidate order.
    """
    if max_chunks <= 0:
        return []
    candidates = list(ranked_candidates)
    baseline = candidates[:max_chunks]
    if not baseline:
        return []

    groups = _groups(query, candidates, available)
    if not groups:
        return baseline
    encoding = tiktoken.get_encoding("cl100k_base") if max_tokens is not None else None

    def content_tokens(chunks: Sequence[DocumentChunk]) -> int:
        if encoding is None:
            return 0
        return sum(len(encoding.encode(c.content, disallowed_special=())) for c in chunks)

    grouped_documents = {group.document_id for group in groups}
    required: list[ScoredChunk] = []
    represented: set[str] = set()
    for chunk in baseline:
        document_id = chunk.metadata.document_id
        if document_id not in grouped_documents and document_id not in represented:
            required.append(chunk)
            represented.add(document_id)

    selected = list(required)
    seen = {chunk.chunk_id for chunk in selected}
    baseline_documents = {chunk.metadata.document_id for chunk in baseline}
    for group in groups:
        if group.document_id not in represented and len(selected) >= max_chunks:
            continue
        evidence: list[DocumentChunk] = [group.table]
        evidence.extend(chunk for chunk in (group.caption, group.header) if chunk is not None)
        evidence.sort(key=lambda chunk: (chunk.metadata.char_start, chunk.chunk_id))
        pending = [chunk for chunk in evidence if chunk.chunk_id not in seen]
        # Do not promote a table at the expense of its known explanatory sources.
        remaining_documents = baseline_documents - represented - {group.document_id}
        if len(selected) + len(pending) + len(remaining_documents) > max_chunks:
            continue
        reserved: dict[str, ScoredChunk] = {}
        for candidate in baseline:
            if candidate.metadata.document_id in remaining_documents:
                reserved.setdefault(candidate.metadata.document_id, candidate)
        if (
            max_tokens is not None
            and content_tokens([*selected, *pending, *reserved.values()]) > max_tokens
        ):
            continue
        logger.debug(
            "table_context selected id=%s document=%s page=%d rank=%d overlap=%d evidence=%s",
            group.table.chunk_id,
            group.document_id,
            group.page_number,
            group.rank,
            group.overlap,
            [source.chunk_id for source in evidence],
        )
        for source in pending:
            selected.append(
                source
                if isinstance(source, ScoredChunk)
                else ScoredChunk(
                    chunk_id=source.chunk_id,
                    content=source.content,
                    metadata=source.metadata,
                    score=0.0,
                )
            )
            seen.add(source.chunk_id)
        represented.add(group.document_id)

    for chunk in baseline:
        if chunk.metadata.document_id not in represented and len(selected) < max_chunks:
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            represented.add(chunk.metadata.document_id)

    for chunk in candidates:
        if len(selected) >= max_chunks:
            break
        if chunk.chunk_id not in seen:
            selected.append(chunk)
            seen.add(chunk.chunk_id)
    return selected
