import io
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from app.core.exceptions import DocumentParsingError
from app.schemas.document import ParsedDocument, ParsedPage, TextBlock

logger = logging.getLogger(__name__)


def serialize_table_to_markdown(matrix: Sequence[Sequence[str | None]]) -> str:
    """Serialize a 2D matrix of cell strings into a GitHub Flavored Markdown (GFM) table.

    Normalizes whitespace, flattens internal linebreaks to spaces, escapes pipe characters,
    pads rows to equal column lengths, and generates fallback column names if headers are blank.

    Args:
        matrix: 2D table matrix where matrix[0] represents the header row and
            matrix[1:] represent data rows.

    Returns:
        Formatted GFM Markdown table string, or empty string if input is invalid
        or contains fewer than 2 columns/rows.
    """
    if not matrix or len(matrix) < 2:
        return ""

    raw_cleaned: list[list[str]] = []
    for row in matrix:
        cleaned_row: list[str] = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                c = " ".join(str(cell).split())
                c = c.replace("|", r"\|")
                cleaned_row.append(c)
        raw_cleaned.append(cleaned_row)

    data_rows = [r for r in raw_cleaned[1:] if any(r)]
    if not data_rows:
        return ""

    data_max_cols = max(len(r) for r in data_rows)
    raw_header = raw_cleaned[0]

    if any(raw_header):
        max_cols = max(len(raw_header), data_max_cols)
        headers = raw_header + [""] * (max_cols - len(raw_header))
        for i in range(len(headers)):
            if not headers[i]:
                headers[i] = f"Column {i + 1}"
    else:
        max_cols = data_max_cols
        headers = [f"Column {i + 1}" for i in range(max_cols)]

    if max_cols < 2:
        return ""

    padded_data = [r + [""] * (max_cols - len(r)) for r in data_rows]

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * max_cols) + " |"
    data_lines = ["| " + " | ".join(r) + " |" for r in padded_data]

    return "\n".join([header_line, sep_line] + data_lines)


def _detect_tables_from_page(
    plumber_page: Any,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Extract table bounding boxes and Markdown serialized tables from a pdfplumber page.

    Supports both explicit grid/boxed tables via pdfplumber.Page.find_tables()
    and horizontal rule-bordered (e.g. LaTeX booktabs) tables via horizontal rule clustering.

    Args:
        plumber_page: The pdfplumber Page instance to extract tables from.

    Returns:
        List of tuples containing:
            - bbox: (x0, top, x1, bottom) bounding box
            - md: Formatted GitHub Flavored Markdown table string
    """
    detected: list[tuple[tuple[float, float, float, float], str]] = []

    # 1. Native find_tables for explicit bordered / grid tables
    for t in plumber_page.find_tables():
        matrix = t.extract()
        if matrix and len(matrix) >= 2 and any(len(r) >= 2 for r in matrix):
            md = serialize_table_to_markdown(matrix)
            if md:
                detected.append((t.bbox, md))

    # 2. Rule-based / booktabs detection from horizontal lines and thin rects
    rules = [
        {
            "top": float(line["top"]),
            "bottom": float(line["bottom"]),
            "x0": float(line["x0"]),
            "x1": float(line["x1"]),
        }
        for line in plumber_page.lines
        if abs(line["top"] - line["bottom"]) < 2 and (line["x1"] - line["x0"]) > 50
    ] + [
        {
            "top": float(r["top"]),
            "bottom": float(r["bottom"]),
            "x0": float(r["x0"]),
            "x1": float(r["x1"]),
        }
        for r in plumber_page.rects
        if (r["bottom"] - r["top"]) <= 3.0 and (r["x1"] - r["x0"]) > 50.0
    ]
    h_lines = sorted(rules, key=lambda rule: (rule["top"], rule["x0"]))

    def is_contained(
        inner: tuple[float, float, float, float],
        outer: tuple[float, float, float, float],
    ) -> bool:
        return (
            inner[0] >= outer[0] - 3.0
            and inner[1] >= outer[1] - 3.0
            and inner[2] <= outer[2] + 3.0
            and inner[3] <= outer[3] + 3.0
        )

    i = 0
    while i < len(h_lines):
        line = h_lines[i]
        matching = [line]
        j = i + 1
        while j < len(h_lines):
            nxt = h_lines[j]
            if abs(nxt["x0"] - line["x0"]) < 8 and abs(nxt["x1"] - line["x1"]) < 8:
                if nxt["top"] - matching[-1]["bottom"] < 400:
                    matching.append(nxt)
                    if len(matching) == 3:
                        j += 1
                        break
            j += 1

        if len(matching) >= 2:
            top_y = matching[0]["top"]
            bottom_y = matching[-1]["bottom"]
            x0 = min(m["x0"] for m in matching)
            x1 = max(m["x1"] for m in matching)
            bbox = (x0 - 2, top_y - 2, x1 + 2, bottom_y + 2)

            already = any(is_contained(bbox, d_bbox) for d_bbox, _ in detected)
            if not already:
                try:
                    crop = plumber_page.crop(bbox)
                    matrix = crop.extract_table(
                        {"vertical_strategy": "text", "horizontal_strategy": "text"}
                    )
                    if matrix and len(matrix) >= 2 and any(len(r) >= 2 for r in matrix):
                        md = serialize_table_to_markdown(matrix)
                        if md:
                            # Replace any smaller detected table fully contained inside this table
                            detected = [(b, m) for b, m in detected if not is_contained(b, bbox)]
                            detected.append((bbox, md))
                except Exception as err:
                    logger.debug("Failed to extract candidate table at %s: %s", bbox, err)
            i = j
        else:
            i += 1

    return detected


class PDFParser:
    """Extracts text blocks and reconstructs natural reading order across documents.

    Supports single-column and multi-column academic layouts with gutter detection,
    multi-author row banding, font-size differential footnote deferral, and table extraction.
    """

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PDF document from a filesystem path.

        Args:
            file_path: Absolute or relative path to the target PDF file.

        Returns:
            ParsedDocument containing per-page layout metadata and ordered text blocks.

        Raises:
            FileNotFoundError: If the file does not exist on disk.
            DocumentParsingError: If PyMuPDF fails to open or parse the document.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            raise DocumentParsingError(f"Failed to open PDF document {path.name}: {e}") from e

        try:
            with pdfplumber.open(str(path)) as plumber_doc:
                pages: list[ParsedPage] = []
                for page_index in range(len(doc)):
                    page = doc[page_index]
                    plumber_page = (
                        plumber_doc.pages[page_index]
                        if page_index < len(plumber_doc.pages)
                        else None
                    )
                    parsed_page = self.parse_page(
                        page, page_number=page_index + 1, plumber_page=plumber_page
                    )
                    pages.append(parsed_page)

            doc_text = "\n\n".join(p.text for p in pages if p.text)
            return ParsedDocument(
                filename=path.name,
                page_count=len(pages),
                pages=pages,
                text=doc_text,
            )
        finally:
            doc.close()

    def parse_bytes(self, content: bytes, filename: str = "document.pdf") -> ParsedDocument:
        """Parse a PDF document from an in-memory byte stream.

        Args:
            content: Raw PDF bytes.
            filename: Original or display filename for metadata tagging.

        Returns:
            ParsedDocument containing per-page layout metadata and ordered text blocks.

        Raises:
            DocumentParsingError: If bytes are empty or PyMuPDF fails to parse the stream.
        """
        if not content:
            raise DocumentParsingError("PDF content bytes cannot be empty")

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            raise DocumentParsingError(f"Failed to open PDF stream for {filename}: {e}") from e

        try:
            with pdfplumber.open(io.BytesIO(content)) as plumber_doc:
                pages: list[ParsedPage] = []
                for page_index in range(len(doc)):
                    page = doc[page_index]
                    plumber_page = (
                        plumber_doc.pages[page_index]
                        if page_index < len(plumber_doc.pages)
                        else None
                    )
                    parsed_page = self.parse_page(
                        page, page_number=page_index + 1, plumber_page=plumber_page
                    )
                    pages.append(parsed_page)

            doc_text = "\n\n".join(p.text for p in pages if p.text)
            return ParsedDocument(
                filename=filename,
                page_count=len(pages),
                pages=pages,
                text=doc_text,
            )
        finally:
            doc.close()

    def parse_page(
        self,
        page: fitz.Page,
        page_number: int | None = None,
        plumber_page: Any = None,
    ) -> ParsedPage:
        """Parse an individual PDF page into layout-sorted text blocks and Markdown tables.

        Args:
            page: PyMuPDF Page instance.
            page_number: Optional 1-indexed page number override.
            plumber_page: Optional pdfplumber Page instance for table extraction.

        Returns:
            ParsedPage with column metadata, reading-order preserved text, and tables.
        """
        p_num = page_number if page_number is not None else (page.number + 1)
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)

        # Detect tables from plumber_page (or fallback to page.parent if plumber_page not provided)
        table_tuples: list[tuple[tuple[float, float, float, float], str]] = []
        if plumber_page is not None:
            table_tuples = _detect_tables_from_page(plumber_page)
        elif page.parent is not None:
            try:
                with pdfplumber.open(io.BytesIO(page.parent.tobytes())) as parent_doc:
                    if page.number < len(parent_doc.pages):
                        table_tuples = _detect_tables_from_page(parent_doc.pages[page.number])
            except Exception as err:
                logger.debug("Fallback pdfplumber table extraction failed: %s", err)
                table_tuples = []

        table_blocks: list[TextBlock] = []
        table_bboxes: list[tuple[float, float, float, float]] = []
        for t_bbox, t_md in table_tuples:
            table_bboxes.append(t_bbox)
            table_blocks.append(
                TextBlock(
                    text=t_md + "\n",
                    bbox=t_bbox,
                    page_number=p_num,
                    block_type=0,
                    is_table=True,
                )
            )

        def overlaps_table(bbox: tuple[float, float, float, float]) -> bool:
            bx0, by0, bx1, by1 = bbox
            cx = (bx0 + bx1) / 2.0
            cy = (by0 + by1) / 2.0
            for tx0, ty0, tx1, ty1 in table_bboxes:
                if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                    return True
                ix0 = max(bx0, tx0)
                iy0 = max(by0, ty0)
                ix1 = min(bx1, tx1)
                iy1 = min(by1, ty1)
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    b_area = (bx1 - bx0) * (by1 - by0)
                    if b_area > 0 and (inter_area / b_area) >= 0.50:
                        return True
            return False

        d = page.get_text("dict")
        raw_dict_blocks = d.get("blocks", [])

        char_sizes: list[float] = []
        raw_candidates: list[tuple[TextBlock, float]] = []

        for b in raw_dict_blocks:
            if b.get("type") == 0:
                bbox = (
                    float(b["bbox"][0]),
                    float(b["bbox"][1]),
                    float(b["bbox"][2]),
                    float(b["bbox"][3]),
                )
                if overlaps_table(bbox):
                    continue

                lines = b.get("lines", [])
                lines_text = [
                    "".join(s.get("text", "") for s in line.get("spans", [])) for line in lines
                ]
                text = "\n".join(lines_text).strip()
                if not text:
                    continue

                spans = [s for line in lines for s in line.get("spans", [])]
                chars = sum(len(s.get("text", "")) for s in spans)
                weighted_size = (
                    sum(float(s.get("size", 10.0)) * len(s.get("text", "")) for s in spans)
                    / max(1, chars)
                    if chars > 0
                    else 10.0
                )

                for s in spans:
                    char_sizes.extend([float(s.get("size", 10.0))] * len(s.get("text", "")))

                tb = TextBlock(
                    text=text + "\n",
                    bbox=bbox,
                    page_number=p_num,
                    block_type=0,
                )
                raw_candidates.append((tb, weighted_size))

        if not raw_candidates and not table_blocks:
            return ParsedPage(
                page_number=p_num,
                width=page_w,
                height=page_h,
                is_dual_column=False,
                gutter_x=None,
                blocks=[],
                text="",
            )

        median_font = statistics.median(char_sizes) if char_sizes else 10.0

        # Separate footnotes / marginalia (bottom 25% of page with font size < 85% of median)
        footnote_blocks: list[TextBlock] = []
        body_blocks: list[TextBlock] = []
        for tb, weighted_size in raw_candidates:
            if (tb.bbox[1] >= 0.75 * page_h) and (weighted_size < 0.85 * median_font):
                tb.is_footnote = True
                footnote_blocks.append(tb)
            else:
                tb.is_footnote = False
                body_blocks.append(tb)

        # Include extracted table blocks in the body blocks for layout sorting
        body_blocks.extend(table_blocks)

        is_dual, gutter_x = self._detect_columns(body_blocks, page_w)
        sorted_blocks = self._sort_blocks(body_blocks, footnote_blocks, is_dual, gutter_x, page_w)
        page_text = "\n\n".join(b.text.strip() for b in sorted_blocks if b.text.strip())

        return ParsedPage(
            page_number=p_num,
            width=page_w,
            height=page_h,
            is_dual_column=is_dual,
            gutter_x=gutter_x,
            blocks=sorted_blocks,
            text=page_text,
        )

    def _detect_columns(
        self,
        blocks: list[TextBlock],
        page_width: float,
    ) -> tuple[bool, float | None]:
        """Detect if the page contains a dual-column layout and determine the gutter coordinate."""
        if not blocks:
            return False, None

        mid_x = page_width / 2.0

        def is_spanning_x(b: TextBlock) -> bool:
            x0, _, x1, _ = b.bbox
            width = x1 - x0
            crosses = (x0 < mid_x - 10.0) and (x1 > mid_x + 10.0)
            wide = (
                x0 < (mid_x - 0.05 * page_width)
                and x1 > (mid_x + 0.05 * page_width)
                and width > (0.40 * page_width)
            )
            return crosses or wide

        non_spanning = [b for b in blocks if not is_spanning_x(b)]
        left_cands = [b for b in non_spanning if (b.bbox[0] + b.bbox[2]) / 2.0 < mid_x]
        right_cands = [b for b in non_spanning if (b.bbox[0] + b.bbox[2]) / 2.0 >= mid_x]

        if len(left_cands) < 2 or len(right_cands) < 2:
            return False, None

        left_text_len = sum(len(b.text.strip()) for b in left_cands)
        right_text_len = sum(len(b.text.strip()) for b in right_cands)
        if left_text_len < 40 or right_text_len < 40:
            return False, None

        max_left_x1 = max(b.bbox[2] for b in left_cands)
        min_right_x0 = min(b.bbox[0] for b in right_cands)

        gutter_cand: float | None = None
        if max_left_x1 <= min_right_x0 + 15.0:
            gutter_cand = (max_left_x1 + min_right_x0) / 2.0
        else:
            sorted_left_x1 = sorted(b.bbox[2] for b in left_cands)
            sorted_right_x0 = sorted(b.bbox[0] for b in right_cands)
            p80_left = sorted_left_x1[int(len(sorted_left_x1) * 0.8)]
            p20_right = sorted_right_x0[int(len(sorted_right_x0) * 0.2)]
            if p80_left <= p20_right:
                gutter_cand = (p80_left + p20_right) / 2.0

        if gutter_cand is not None and (0.30 * page_width <= gutter_cand <= 0.70 * page_width):
            return True, gutter_cand

        return False, None

    def _sort_blocks(
        self,
        body_blocks: list[TextBlock],
        footnote_blocks: list[TextBlock],
        is_dual_column: bool,
        gutter_x: float | None,
        page_width: float,
    ) -> list[TextBlock]:
        """Order text blocks into natural reading sequence based on detected layout."""
        footers_sorted = sorted(footnote_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
        for b in footers_sorted:
            b.column_index = None

        if not is_dual_column or gutter_x is None:
            sorted_single = sorted(body_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
            for b in sorted_single:
                b.column_index = None
            return sorted_single + footers_sorted

        def is_spanning_gutter(b: TextBlock) -> bool:
            x0, _, x1, _ = b.bbox
            width = x1 - x0
            gutter_margin = 10.0
            crosses = (x0 < gutter_x - gutter_margin) and (x1 > gutter_x + gutter_margin)
            wide = (
                x0 < (gutter_x - 0.05 * page_width)
                and x1 > (gutter_x + 0.05 * page_width)
                and width > (0.40 * page_width)
            )
            return crosses or wide

        def v_iou(b1: TextBlock, b2: TextBlock) -> float:
            inter = max(0.0, min(b1.bbox[3], b2.bbox[3]) - max(b1.bbox[1], b2.bbox[1]))
            union = max(b1.bbox[3], b2.bbox[3]) - min(b1.bbox[1], b2.bbox[1])
            return inter / union if union > 0.0 else 0.0

        spanning_cands = [b for b in body_blocks if is_spanning_gutter(b)]

        # Group blocks that share vertical IoU > 0.50 with a spanning block (e.g. author cards)
        spanning_group_ids: set[int] = {id(b) for b in spanning_cands}
        for sb in spanning_cands:
            for b in body_blocks:
                if v_iou(b, sb) > 0.50:
                    spanning_group_ids.add(id(b))

        column_candidates = [b for b in body_blocks if id(b) not in spanning_group_ids]
        col_left = [b for b in column_candidates if (b.bbox[0] + b.bbox[2]) / 2.0 < gutter_x]
        col_right = [b for b in column_candidates if (b.bbox[0] + b.bbox[2]) / 2.0 >= gutter_x]

        if not col_left or not col_right:
            sorted_single = sorted(body_blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
            for b in sorted_single:
                b.column_index = None
            return sorted_single + footers_sorted

        right_top_y = min(b.bbox[1] for b in col_right)
        top_spans = [
            b for b in body_blocks if id(b) in spanning_group_ids and b.bbox[1] < right_top_y
        ]
        header_threshold_y = min(
            max((b.bbox[3] for b in top_spans), default=right_top_y),
            right_top_y,
        )

        headers = [b for b in body_blocks if b.bbox[3] <= header_threshold_y + 5.0]

        # Cluster headers into horizontal bands; within each band sort left-to-right (x0)
        h_bands: list[list[TextBlock]] = []
        for b in sorted(headers, key=lambda x: x.bbox[1]):
            placed = False
            for hb in h_bands:
                if any(v_iou(b, m) > 0.50 for m in hb):
                    hb.append(b)
                    placed = True
                    break
            if not placed:
                h_bands.append([b])

        headers_sorted: list[TextBlock] = []
        for hb in h_bands:
            for b in sorted(hb, key=lambda x: x.bbox[0]):
                b.column_index = None
                headers_sorted.append(b)

        col_zone = [b for b in body_blocks if b not in headers]
        mid_spanning = [b for b in col_zone if is_spanning_gutter(b)]
        col_only = [b for b in col_zone if not is_spanning_gutter(b)]

        col_ordered: list[TextBlock] = []
        if not mid_spanning:
            left_sorted = sorted(
                [b for b in col_only if (b.bbox[0] + b.bbox[2]) / 2.0 < gutter_x],
                key=lambda b: b.bbox[1],
            )
            right_sorted = sorted(
                [b for b in col_only if (b.bbox[0] + b.bbox[2]) / 2.0 >= gutter_x],
                key=lambda b: b.bbox[1],
            )
            for b in left_sorted:
                b.column_index = 0
            for b in right_sorted:
                b.column_index = 1
            col_ordered = left_sorted + right_sorted
        else:
            sorted_mid_spans = sorted(mid_spanning, key=lambda b: b.bbox[1])
            bands: list[tuple[float, float, TextBlock | None]] = []
            curr_y = min(b.bbox[1] for b in col_only) if col_only else 0.0
            for sb in sorted_mid_spans:
                if sb.bbox[1] > curr_y:
                    bands.append((curr_y, sb.bbox[1], None))
                bands.append((sb.bbox[1], sb.bbox[3], sb))
                curr_y = max(curr_y, sb.bbox[3])
            col_end_y = max(b.bbox[3] for b in col_only) if col_only else curr_y
            if curr_y < col_end_y:
                bands.append((curr_y, col_end_y, None))

            for y_start, y_end, span_block in bands:
                if span_block is not None:
                    span_block.column_index = None
                    col_ordered.append(span_block)
                else:
                    band_b = [
                        b for b in col_only if y_start <= (b.bbox[1] + b.bbox[3]) / 2.0 < y_end
                    ]
                    band_left = sorted(
                        [b for b in band_b if (b.bbox[0] + b.bbox[2]) / 2.0 < gutter_x],
                        key=lambda b: b.bbox[1],
                    )
                    band_right = sorted(
                        [b for b in band_b if (b.bbox[0] + b.bbox[2]) / 2.0 >= gutter_x],
                        key=lambda b: b.bbox[1],
                    )
                    for b in band_left:
                        b.column_index = 0
                    for b in band_right:
                        b.column_index = 1
                    col_ordered.extend(band_left)
                    col_ordered.extend(band_right)

        return headers_sorted + col_ordered + footers_sorted
