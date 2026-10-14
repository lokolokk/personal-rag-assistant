"""Structural chunking engine with layout awareness, table isolation, and rich metadata."""

import logging
import re

import tiktoken

from app.core.config import get_settings
from app.schemas.document import (
    ChunkMetadata,
    DocumentChunk,
    ParsedDocument,
    ParsedPage,
)

logger = logging.getLogger(__name__)

# Standard heading patterns for academic papers and technical reports
HEADING_PATTERNS = [
    # Numbered sections e.g. "1. Introduction", "1 Introduction", "2.1 Background", "3.2.1 Analysis"
    re.compile(r"^\s*(?:\d+\.?)+\s+[A-Z][A-Za-z0-9\s\-_:]{1,80}$"),
    # Roman numeral headings e.g. "I. INTRODUCTION", "IV. EXPERIMENTAL SETUP"
    re.compile(r"^\s*[IVXLCDM]+\.\s+[A-Z0-9\s\-_:]{2,80}$"),
    # Lettered appendix e.g. "Appendix A. Proofs", "A. Dataset Details"
    re.compile(r"^\s*(?:Appendix\s+)?[A-Z]\.\s+[A-Z0-9\s\-_:]{2,80}$"),
    # Standard unnumbered academic headings
    re.compile(
        r"^\s*(?:Abstract|Introduction|Background|Related Work|Methodology|Method|Methods|"
        r"System Architecture|Experiments?|Experimental Results?|Results?|Discussion|"
        r"Conclusions?|References?|Acknowledgments?|Appendix)\b.*$",
        re.IGNORECASE,
    ),
    # Markdown headings e.g. "# Section", "## Subsection"
    re.compile(r"^\s*#{1,6}\s+.*$"),
]


class StructuralChunker:
    """Partitions layout-aware parsed documents into structural chunks with rich metadata.

    Enforces:
    - Target chunk size (default 600 tokens) with sliding overlap (default 100 tokens).
    - Immediate hard boundaries at major section headers.
    - Immediate hard boundaries at physical document page transitions.
    - Atomic table isolation (is_table=True blocks are never split or mixed with text).
    - Deterministic chunk_id formatting: doc_<id>_p<page>_c<idx>.
    """

    def __init__(
        self,
        target_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        cfg = get_settings()
        self.target_chunk_size = target_chunk_size or cfg.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or cfg.CHUNK_OVERLAP
        self.encoding_name = encoding_name

        try:
            self._tokenizer: tiktoken.Encoding | None = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            logger.warning(
                "Failed to initialize tiktoken encoding %s: %s. Using word fallback.",
                encoding_name,
                exc,
            )
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback whitespace approximation."""
        if not text.strip():
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        # Word-based token approximation (1 word ~ 1.33 tokens)
        return max(1, int(len(text.split()) * 1.33))

    def _encode_tokens(self, text: str) -> list[int] | list[str]:
        """Encode text to tokens."""
        if self._tokenizer is not None:
            return list(self._tokenizer.encode(text))
        return text.split()

    def _decode_tokens(self, tokens: list[int] | list[str]) -> str:
        """Decode tokens back to string."""
        if self._tokenizer is not None and tokens and isinstance(tokens[0], int):
            int_tokens = [t for t in tokens if isinstance(t, int)]
            decoded: str = self._tokenizer.decode(int_tokens)
            return decoded
        return " ".join(str(t) for t in tokens)

    def _is_heading(self, line: str) -> bool:
        """Check whether a line corresponds to a major structural heading."""
        cleaned = line.strip()
        if not cleaned:
            return False

        # Headings are typically compact (< 100 characters and < 15 words)
        if len(cleaned) > 100 or len(cleaned.split()) > 15:
            return False

        # Exclude regular sentences ending with period (unless Roman or digit prefix like '1.')
        is_numbering = bool(re.match(r"^\s*(?:\d+\.?|[IVXLCDM]+\.)$", cleaned))
        if cleaned.endswith((".", ",", ";", ":")) and not is_numbering:
            if not any(pat.match(cleaned) for pat in HEADING_PATTERNS):
                return False

        return any(pat.match(cleaned) is not None for pat in HEADING_PATTERNS)

    def _clean_section_title(self, heading: str) -> str:
        """Normalize extracted heading into a clean section title."""
        cleaned = re.sub(r"^\s*#{1,6}\s*", "", heading).strip()
        # Remove trailing colon if any
        if cleaned.endswith(":"):
            cleaned = cleaned[:-1].strip()
        return cleaned

    def chunk_document(
        self,
        parsed_doc: ParsedDocument,
        document_id: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunk a parsed document into structured DocumentChunk items.

        Args:
            parsed_doc: ParsedDocument containing parsed pages and text blocks.
            document_id: Optional document identifier. If omitted, uses parsed_doc.document_id
                         or generates a deterministic identifier based on the filename.

        Returns:
            List of structured DocumentChunk objects ordered by page and reading position.
        """
        doc_id = document_id or parsed_doc.document_id
        if not doc_id:
            safe_hash = abs(hash(parsed_doc.filename)) % 100_000_000
            doc_id = f"doc_{safe_hash:08x}"

        chunks: list[DocumentChunk] = []
        current_section = "Document Header"
        global_char_offset = 0

        for page in parsed_doc.pages:
            page_chunks = self._chunk_page(
                page=page,
                doc_id=doc_id,
                filename=parsed_doc.filename,
                current_section=current_section,
                global_char_offset=global_char_offset,
            )
            chunks.extend(page_chunks)

            # Update section context for subsequent pages
            if page_chunks:
                current_section = page_chunks[-1].metadata.section_title

            # Increment global char offset based on page text
            page_len = len(page.text) if page.text else sum(len(b.text) for b in page.blocks)
            global_char_offset += page_len + 1

        return chunks

    def _chunk_page(
        self,
        page: ParsedPage,
        doc_id: str,
        filename: str,
        current_section: str,
        global_char_offset: int,
    ) -> list[DocumentChunk]:
        """Chunk a single page, respecting section boundaries and isolating tables."""
        chunks: list[DocumentChunk] = []
        chunk_idx = 1
        active_section = current_section

        # Buffer for accumulating text blocks within the same section and page
        buffer_text_parts: list[str] = []
        page_char_offset = global_char_offset

        def flush_buffer() -> None:
            nonlocal chunk_idx, buffer_text_parts, page_char_offset
            if not buffer_text_parts:
                return

            full_text = "\n\n".join(buffer_text_parts).strip()
            buffer_text_parts = []
            if not full_text:
                return

            # Split large text into target chunks with sliding overlap
            text_chunks = self._split_text_with_overlap(full_text)
            for text_chunk in text_chunks:
                token_count = self.count_tokens(text_chunk)
                chunk_id = f"{doc_id}_p{page.page_number}_c{chunk_idx}"
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    filename=filename,
                    page_number=page.page_number,
                    section_title=active_section,
                    is_table=False,
                    token_count=token_count,
                    char_start=page_char_offset,
                    char_end=page_char_offset + len(text_chunk),
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, content=text_chunk, metadata=meta))
                page_char_offset += len(text_chunk) + 1
                chunk_idx += 1

        for block in page.blocks:
            cleaned_text = block.text.strip()
            if not cleaned_text:
                continue

            # Case 1: Atomic Table Isolation & Header-Preserving Framing
            if block.is_table:
                # Flush any preceding text chunk first
                flush_buffer()

                table_sub_chunks = self._split_table_with_header_preservation(cleaned_text)
                for t_chunk in table_sub_chunks:
                    token_count = self.count_tokens(t_chunk)
                    chunk_id = f"{doc_id}_p{page.page_number}_c{chunk_idx}"
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        filename=filename,
                        page_number=page.page_number,
                        section_title=active_section,
                        is_table=True,
                        token_count=token_count,
                        char_start=page_char_offset,
                        char_end=page_char_offset + len(t_chunk),
                    )
                    chunks.append(DocumentChunk(chunk_id=chunk_id, content=t_chunk, metadata=meta))
                    page_char_offset += len(t_chunk) + 1
                    chunk_idx += 1
                continue

            # Case 2: Check for Major Section Headings
            # The block might be a single heading line or start with a heading line
            lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
            if lines and self._is_heading(lines[0]):
                # Flush preceding text under previous section
                flush_buffer()

                # Update active section title
                active_section = self._clean_section_title(lines[0])

                # Accumulate the heading and any remaining lines in this block
                buffer_text_parts.append(cleaned_text)
                continue

            # Check if an internal line in the block is a heading
            has_internal_heading = False
            for idx, line in enumerate(lines):
                if idx > 0 and self._is_heading(line):
                    # Split block before this heading
                    pre_text = "\n".join(lines[:idx]).strip()
                    post_text = "\n".join(lines[idx:]).strip()

                    if pre_text:
                        buffer_text_parts.append(pre_text)
                    flush_buffer()

                    active_section = self._clean_section_title(line)
                    buffer_text_parts.append(post_text)
                    has_internal_heading = True
                    break

            if not has_internal_heading:
                # Regular body text block
                buffer_text_parts.append(cleaned_text)

                # Check if buffer has reached or exceeded target size
                accumulated = "\n\n".join(buffer_text_parts)
                if self.count_tokens(accumulated) >= self.target_chunk_size:
                    flush_buffer()

        # Flush any remaining text at the end of the page
        flush_buffer()

        return chunks

    def _split_text_with_overlap(self, text: str) -> list[str]:
        """Split a long string into chunks obeying target token size with sliding overlap."""
        tokens = self._encode_tokens(text)
        total_tokens = len(tokens)

        if total_tokens <= self.target_chunk_size:
            return [text]

        chunks: list[str] = []
        step = max(1, self.target_chunk_size - self.chunk_overlap)
        start = 0

        while start < total_tokens:
            end = min(start + self.target_chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_str = self._decode_tokens(chunk_tokens).strip()

            if chunk_str:
                chunks.append(chunk_str)

            if end >= total_tokens:
                break
            start += step

        return chunks

    def _is_table_separator(self, line: str) -> bool:
        """Check whether a line corresponds to a Markdown table separator."""
        cleaned = line.strip()
        if not cleaned or "|" not in cleaned:
            return False
        parts = cleaned.split("|")
        if parts and not parts[0]:
            parts = parts[1:]
        if parts and not parts[-1]:
            parts = parts[:-1]
        if not parts:
            return False
        return all(re.match(r"^:?-+:?$", p.strip()) is not None for p in parts)

    def _split_table_with_header_preservation(self, table_text: str) -> list[str]:
        """Split a large table into sub-chunks, preserving headers and captions across all chunks.

        Enforces:
        - Atomic retention if total tokens <= target_chunk_size.
        - Strict splitting at Markdown row boundaries (never bisecting rows).
        - Prepending table caption (if present), column headers, and separator to each sub-chunk.
        """
        if self.count_tokens(table_text) <= self.target_chunk_size:
            return [table_text]

        lines = [ln.strip() for ln in table_text.splitlines() if ln.strip()]
        if len(lines) < 3:
            return [table_text]

        sep_idx = -1
        for i, line in enumerate(lines):
            if self._is_table_separator(line):
                sep_idx = i
                break

        if sep_idx < 1 or "|" not in lines[sep_idx - 1]:
            return [table_text]

        header_line = lines[sep_idx - 1]
        separator_line = lines[sep_idx]
        caption_lines = lines[: sep_idx - 1]
        data_rows = lines[sep_idx + 1 :]

        if not data_rows:
            return [table_text]

        prefix_parts: list[str] = []
        if caption_lines:
            prefix_parts.append("\n".join(caption_lines))
        prefix_parts.append(header_line)
        prefix_parts.append(separator_line)
        prefix = "\n".join(prefix_parts)

        sub_chunks: list[str] = []
        current_rows: list[str] = []

        for row in data_rows:
            candidate_content = "\n".join([prefix] + current_rows + [row])
            if self.count_tokens(candidate_content) > self.target_chunk_size and current_rows:
                sub_chunks.append("\n".join([prefix] + current_rows))
                current_rows = [row]
            else:
                current_rows.append(row)

        if current_rows:
            sub_chunks.append("\n".join([prefix] + current_rows))

        return sub_chunks if sub_chunks else [table_text]
