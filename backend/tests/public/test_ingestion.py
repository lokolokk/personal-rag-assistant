"""Public ingestion contracts using original synthetic data only.

Historical real-paper regressions remain in the local suite. These tests exercise
the same application entry points without claiming to replay historical evidence.
"""

from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from app.main import create_application
from app.schemas.document import ParsedDocument, ParsedPage, TextBlock
from app.services.ingestion.chunker import StructuralChunker
from app.services.ingestion.parser import PDFParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_single_column_layout_and_byte_parsing_match() -> None:
    path = FIXTURES / "sample_singlecolumn.pdf"
    parser = PDFParser()
    parsed = parser.parse_file(path)
    assert parsed.page_count == 1
    assert not parsed.pages[0].is_dual_column
    assert all(block.column_index is None for block in parsed.pages[0].blocks)
    assert (
        parsed.text.index("1. Executive Summary")
        < parsed.text.index("2. Scalability and Distributed")
        < parsed.text.index("3. Conclusion and Future Directions")
    )
    assert parser.parse_bytes(path.read_bytes(), filename=path.name).model_dump() == (
        parsed.model_dump()
    )


def test_original_dual_column_fixture_preserves_reading_order(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_columns.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page(width=595, height=842)
        rectangles = [
            ((54, 40, 541, 110), "Synthetic Layout Contract\nOriginal test authors"),
            ((54, 180, 275, 280), "1 Introduction\nLeft upper quartz narrative. " * 5),
            ((54, 340, 275, 440), "Left lower quartz paragraph. " * 8),
            ((320, 180, 541, 280), "2 Results\nRight upper cobalt narrative. " * 5),
            ((320, 340, 541, 440), "Right lower cobalt paragraph. " * 8),
        ]
        for coordinates, text in rectangles:
            assert page.insert_textbox(fitz.Rect(*coordinates), text, fontsize=10) >= 0
        pdf.save(path)
    parsed = PDFParser().parse_file(path)
    page_data = parsed.pages[0]
    assert page_data.is_dual_column
    assert page_data.gutter_x is not None and 250 <= page_data.gutter_x <= 350
    text = page_data.text
    assert text.index("Synthetic Layout Contract") < text.index("Left upper")
    assert text.index("Left upper") < text.index("Left lower") < text.index("Right upper")
    assert text.index("Right upper") < text.index("Right lower")
    assert all(block.column_index == 0 for block in page_data.blocks if "Left" in block.text)
    assert all(block.column_index == 1 for block in page_data.blocks if "Right" in block.text)


def test_grid_table_extraction_and_chunk_metadata() -> None:
    path = FIXTURES / "sample_table.pdf"
    parsed = PDFParser().parse_file(path)
    tables = [block for block in parsed.pages[0].blocks if block.is_table]
    assert len(tables) == 1
    header = "| Model Architecture | Parameters | Inference Latency |"
    assert header in tables[0].text
    assert "| GPT-4-Turbo | 1.8T | 120ms |" in tables[0].text
    assert "| Claude-3.5-Sonnet | Dense-MoE | 95ms |" in tables[0].text
    assert "| Llama-3-70B | 70B | 45ms |" in tables[0].text
    texts = [block.text for block in parsed.pages[0].blocks]
    assert (
        next(i for i, text in enumerate(texts) if "Table 1:" in text)
        < texts.index(tables[0].text)
        < next(i for i, text in enumerate(texts) if "2. Analytical Discussion" in text)
    )
    assert not any(text.strip() == "120ms" for text in texts)
    chunks = StructuralChunker().chunk_document(parsed, document_id="public-table")
    table_chunks = [chunk for chunk in chunks if chunk.metadata.is_table]
    assert len(table_chunks) == 1 and header in table_chunks[0].content
    assert all(chunk.metadata.page_number == 1 for chunk in chunks)
    assert all(chunk.metadata.filename == path.name for chunk in chunks)
    assert all(chunk.metadata.document_id == "public-table" for chunk in chunks)
    assert all(chunk.metadata.token_count > 0 for chunk in chunks)


def test_large_table_preserves_each_row_and_repeats_headers() -> None:
    caption = "Table 1: Synthetic sensor readings"
    header = "| Sensor | Quartz | Cobalt |"
    separator = "| --- | --- | --- |"
    rows = [f"| Sensor-{i:02d} | {i * 7} units | {i * 11} units |" for i in range(40)]
    table = "\n".join([caption, header, separator, *rows])
    chunker = StructuralChunker(target_chunk_size=100, chunk_overlap=10)
    assert chunker.count_tokens(table) > 100
    document = ParsedDocument(
        filename="synthetic_sensor_table.pdf",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                width=595,
                height=842,
                is_dual_column=False,
                blocks=[
                    TextBlock(text=table, bbox=(54, 50, 541, 700), page_number=1, is_table=True)
                ],
            )
        ],
    )
    chunks = chunker.chunk_document(document, document_id="sensor-test")
    assert len(chunks) > 1
    recovered: list[str] = []
    for chunk in chunks:
        lines = chunk.content.splitlines()
        assert lines[:3] == [caption, header, separator]
        assert chunk.metadata.is_table and chunk.metadata.page_number == 1
        assert all(line.startswith("|") and line.endswith("|") for line in lines[3:])
        recovered.extend(lines[3:])
    assert recovered == rows


def test_public_upload_list_and_delete_are_offline() -> None:
    path = FIXTURES / "sample_table.pdf"
    with TestClient(create_application()) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": (path.name, path.read_bytes(), "application/pdf")},
        )
        assert response.status_code == 201
        item = response.json()
        assert item["pageCount"] == 1 and item["chunkCount"] >= 1
        identifier = item["documentId"]
        listed = client.get("/api/v1/documents")
        assert listed.status_code == 200
        assert [document["id"] for document in listed.json()] == [identifier]
        assert client.delete(f"/api/v1/documents/{identifier}").status_code == 200
        assert client.get("/api/v1/documents").json() == []
