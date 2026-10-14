"""Pydantic schemas for document ingestion and layout parsing."""

from typing import Any

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """Represents an extracted text block with bounding box and reading order metadata."""

    text: str
    bbox: tuple[float, float, float, float]
    page_number: int
    block_type: int = 0
    column_index: int | None = None
    is_footnote: bool = False
    is_table: bool = False


class ParsedPage(BaseModel):
    """Represents a single parsed page with layout metadata and sorted blocks."""

    page_number: int
    width: float
    height: float
    is_dual_column: bool
    gutter_x: float | None = None
    blocks: list[TextBlock] = Field(default_factory=list)
    text: str = ""


class ParsedDocument(BaseModel):
    """Represents a parsed multi-page document."""

    filename: str
    page_count: int
    pages: list[ParsedPage] = Field(default_factory=list)
    text: str = ""
    document_id: str = ""


class ChunkMetadata(BaseModel):
    """Metadata attributes attached to an individual document chunk."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section_title: str
    is_table: bool = False
    token_count: int
    char_start: int
    char_end: int


class DocumentChunk(BaseModel):
    """Represents an isolated, retrievable text or table chunk with rich metadata."""

    chunk_id: str
    content: str
    metadata: ChunkMetadata


class ScoredChunk(DocumentChunk):
    """A retrieved document chunk associated with a search relevance score."""

    score: float


class DocumentItem(BaseModel):
    """Represents summary metadata of an ingested document."""

    id: str
    filename: str
    page_count: int = 0
    chunk_count: int = 0
    uploaded_at: str = ""
    status: str = "ready"
    document_id: str = ""
    pageCount: int = 0
    chunkCount: int = 0
    uploadedAt: str = ""
    documentId: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.document_id:
            self.document_id = self.id
        if not self.documentId:
            self.documentId = self.id
        if self.pageCount == 0 and self.page_count != 0:
            self.pageCount = self.page_count
        elif self.page_count == 0 and self.pageCount != 0:
            self.page_count = self.pageCount
        if self.chunkCount == 0 and self.chunk_count != 0:
            self.chunkCount = self.chunk_count
        elif self.chunk_count == 0 and self.chunkCount != 0:
            self.chunk_count = self.chunkCount
        if not self.uploadedAt and self.uploaded_at:
            self.uploadedAt = self.uploaded_at
        elif not self.uploaded_at and self.uploadedAt:
            self.uploaded_at = self.uploadedAt


class DocumentDeleteResponse(BaseModel):
    """Response payload confirming document deletion."""

    message: str = "Document deleted successfully"
    id: str
    document_id: str = ""
    documentId: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.document_id:
            self.document_id = self.id
        if not self.documentId:
            self.documentId = self.id
