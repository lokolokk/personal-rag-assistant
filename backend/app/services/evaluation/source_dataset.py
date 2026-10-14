"""Load source-verified RAG evaluation cases and guard their evidence mappings."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import fitz
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.services.ingestion.chunker import StructuralChunker
from app.services.ingestion.parser import PDFParser

_CL100K_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
_CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ChunkMapping(_StrictModel):
    status: Literal["mapped", "extraction_failure"]
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_sha256: dict[str, str] = Field(default_factory=dict)
    note: str | None = None


class SourceAnchor(_StrictModel):
    id: str
    page_number: int
    locator: str
    source_quote: str
    verification_method: Literal["original_pdf_text", "original_pdf_visual"] = "original_pdf_text"
    chunk_mapping: ChunkMapping


class ChunkingIdentity(_StrictModel):
    encoding: str
    target_chunk_size: int
    chunk_overlap: int


class ExtractionIssue(_StrictModel):
    anchor_id: str
    page_number: int
    kind: str
    details: str


class SourceDocument(_StrictModel):
    id: str
    filename: str
    title: str
    fixture_path: str
    sha256: str
    page_count: int
    chunking: ChunkingIdentity
    anchors: list[SourceAnchor]
    extraction_issues: list[ExtractionIssue] = Field(default_factory=list)


class ConversationTurn(_StrictModel):
    role: Literal["user", "assistant"]
    content: str


class RequiredFact(_StrictModel):
    id: str
    claim: str
    exact_values: list[str]
    unit: str | None = None
    qualifications: list[str] = Field(default_factory=list)
    source_anchor_ids: list[str]
    acceptable_alternative_evidence_anchor_ids: list[str] = Field(default_factory=list)


class AbsenceDocumentReview(_StrictModel):
    document_id: str
    pages_reviewed: list[int]
    search_terms: list[str]
    method: str


class AbsenceReview(_StrictModel):
    documents: list[AbsenceDocumentReview]
    note: str


class EvaluationCase(_StrictModel):
    id: str
    split: Literal["diagnostic", "held_out"]
    question: str
    retrieval_query: str
    scope_document_ids: list[str]
    history: list[ConversationTurn] = Field(default_factory=list)
    answerability: Literal["answerable", "unanswerable"]
    required_facts: list[RequiredFact] = Field(default_factory=list)
    unanswerable_reason: str | None = None
    absence_review: AbsenceReview | None = None
    out_of_scope_anchor_ids: list[str] = Field(default_factory=list)


class SourceDataset(_StrictModel):
    dataset_id: str
    created_at: str
    held_out_frozen_before_tuning: bool
    documents: list[SourceDocument]
    cases: list[EvaluationCase]


def _normalize_evidence(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _cl100k_cache_file() -> Path | None:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
    elif "DATA_GYM_CACHE_DIR" in os.environ:
        cache_dir = os.environ["DATA_GYM_CACHE_DIR"]
    else:
        cache_dir = str(Path(tempfile.gettempdir()) / "data-gym-cache")
    if not cache_dir:
        return None
    cache_key = hashlib.sha1(_CL100K_URL.encode("utf-8")).hexdigest()
    return Path(cache_dir) / cache_key


def _require_local_tokenizer_cache() -> None:
    cache_file = _cl100k_cache_file()
    if cache_file is None or not cache_file.is_file():
        raise ValueError("Cannot validate chunk mappings without the local cl100k_base cache")
    if hashlib.sha256(cache_file.read_bytes()).hexdigest() != _CL100K_SHA256:
        raise ValueError("Cannot validate chunk mappings: local cl100k_base cache is invalid")


def _validate_document_source(document: SourceDocument, project_root: Path) -> None:
    root = project_root.resolve()
    source_path = (root / document.fixture_path).resolve()
    if not source_path.is_relative_to(root) or not source_path.is_file():
        raise ValueError(f"Source PDF fixture is missing or outside the project: {document.id}")
    pdf_bytes = source_path.read_bytes()
    if hashlib.sha256(pdf_bytes).hexdigest() != document.sha256:
        raise ValueError(f"Source fingerprint mismatch for {document.id}")

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        if len(pdf) != document.page_count:
            raise ValueError(f"Source page count mismatch for {document.id}")
        pages = {index + 1: page.get_text() for index, page in enumerate(pdf)}

    anchor_pages: dict[str, int] = {}
    for anchor in document.anchors:
        if anchor.id in anchor_pages:
            raise ValueError(f"Duplicate source anchor {anchor.id}")
        anchor_pages[anchor.id] = anchor.page_number
        page_text = pages.get(anchor.page_number)
        if page_text is None or not anchor.source_quote.strip():
            raise ValueError(f"Invalid source page or empty quote for {anchor.id}")
        if anchor.verification_method == "original_pdf_text" and _normalize_evidence(
            anchor.source_quote
        ) not in _normalize_evidence(page_text):
            raise ValueError(f"Source anchor quote is not on the original PDF page: {anchor.id}")
        mapping = anchor.chunk_mapping
        if mapping.status == "mapped":
            if anchor.verification_method != "original_pdf_text":
                raise ValueError(
                    f"Visual-only source anchor cannot have a chunk mapping: {anchor.id}"
                )
            if not mapping.chunk_ids or set(mapping.chunk_sha256) != set(mapping.chunk_ids):
                raise ValueError(f"Incomplete chunk mapping for {anchor.id}")
        elif mapping.chunk_ids or mapping.chunk_sha256 or not mapping.note:
            raise ValueError(f"Extraction failure needs an explicit unmapped note: {anchor.id}")
        elif anchor.verification_method != "original_pdf_visual":
            raise ValueError(f"Unmapped source anchor needs visual PDF verification: {anchor.id}")

    issues_by_anchor: set[str] = set()
    for issue in document.extraction_issues:
        if anchor_pages.get(issue.anchor_id) != issue.page_number:
            raise ValueError(f"Extraction issue does not match a source anchor: {issue.anchor_id}")
        issues_by_anchor.add(issue.anchor_id)
    for anchor in document.anchors:
        if (
            anchor.chunk_mapping.status == "extraction_failure"
            and anchor.id not in issues_by_anchor
        ):
            raise ValueError(f"Extraction failure is not recorded for {anchor.id}")


def _validate_chunk_mappings(document: SourceDocument, project_root: Path) -> None:
    mapped_anchors = [
        anchor for anchor in document.anchors if anchor.chunk_mapping.status == "mapped"
    ]
    if not mapped_anchors:
        return
    if document.chunking.encoding != "cl100k_base":
        raise ValueError(f"Unsupported mapping encoding for {document.id}")
    _require_local_tokenizer_cache()
    source_path = (project_root / document.fixture_path).resolve()
    parsed = PDFParser().parse_bytes(source_path.read_bytes(), document.filename)
    chunker = StructuralChunker(
        target_chunk_size=document.chunking.target_chunk_size,
        chunk_overlap=document.chunking.chunk_overlap,
        encoding_name=document.chunking.encoding,
    )
    chunks = {
        chunk.chunk_id: chunk for chunk in chunker.chunk_document(parsed, document_id=document.id)
    }
    for anchor in mapped_anchors:
        for chunk_id in anchor.chunk_mapping.chunk_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Stale chunk mapping for {anchor.id}: {chunk_id}")
            if chunk.metadata.page_number != anchor.page_number:
                raise ValueError(f"Chunk mapping points to another source page: {chunk_id}")
            expected = anchor.chunk_mapping.chunk_sha256[chunk_id]
            actual = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if actual != expected:
                raise ValueError(f"chunk mapping fingerprint mismatch for {chunk_id}")


def _validate_cases(dataset: SourceDataset) -> None:
    documents = {document.id: document for document in dataset.documents}
    if len(documents) != len(dataset.documents):
        raise ValueError("Duplicate source document ID")
    anchors: dict[str, str] = {}
    for document in dataset.documents:
        for anchor in document.anchors:
            if anchor.id in anchors:
                raise ValueError(f"Duplicate source anchor {anchor.id}")
            anchors[anchor.id] = document.id

    case_ids: set[str] = set()
    for case in dataset.cases:
        if case.id in case_ids:
            raise ValueError(f"Duplicate evaluation case ID {case.id}")
        case_ids.add(case.id)
        if len(set(case.scope_document_ids)) != len(case.scope_document_ids):
            raise ValueError(f"Duplicate scope document in {case.id}")
        if not set(case.scope_document_ids).issubset(documents):
            raise ValueError(f"Unknown scope document in {case.id}")

        fact_ids: set[str] = set()
        for fact in case.required_facts:
            if fact.id in fact_ids or not fact.exact_values or not fact.source_anchor_ids:
                raise ValueError(f"Incomplete required fact annotation in {case.id}")
            fact_ids.add(fact.id)
            referenced = fact.source_anchor_ids + fact.acceptable_alternative_evidence_anchor_ids
            for anchor_id in referenced:
                document_id = anchors.get(anchor_id)
                if document_id is None:
                    raise ValueError(f"Unknown evidence anchor {anchor_id} in {case.id}")
                if case.scope_document_ids and document_id not in case.scope_document_ids:
                    raise ValueError(f"Evidence anchor outside scope in {case.id}: {anchor_id}")

        if case.answerability == "answerable":
            if not case.required_facts or case.unanswerable_reason or case.absence_review:
                raise ValueError(f"Answerable case has incomplete labels: {case.id}")
            if case.out_of_scope_anchor_ids:
                raise ValueError(f"Answerable case has out-of-scope evidence: {case.id}")
        else:
            if case.required_facts or not case.unanswerable_reason:
                raise ValueError(f"Unanswerable case has missing or conflicting labels: {case.id}")
            if case.out_of_scope_anchor_ids:
                if case.absence_review:
                    raise ValueError(f"Unanswerable case mixes absence and scope labels: {case.id}")
                if not case.scope_document_ids:
                    raise ValueError(f"Global scope cannot have out-of-scope evidence: {case.id}")
                for anchor_id in case.out_of_scope_anchor_ids:
                    document_id = anchors.get(anchor_id)
                    if document_id is None or document_id in case.scope_document_ids:
                        raise ValueError(f"Invalid out-of-scope evidence in {case.id}")
            elif case.absence_review is None:
                raise ValueError(f"Unanswerable case needs an absence review: {case.id}")
            else:
                expected_docs = set(case.scope_document_ids) or set(documents)
                reviewed = {item.document_id for item in case.absence_review.documents}
                if reviewed != expected_docs:
                    raise ValueError(f"Absence review does not match scope in {case.id}")
                for item in case.absence_review.documents:
                    document = documents[item.document_id]
                    if sorted(item.pages_reviewed) != list(range(1, document.page_count + 1)):
                        raise ValueError(
                            f"Absence review does not cover every source page in {case.id}"
                        )

    diagnostic_count = sum(case.split == "diagnostic" for case in dataset.cases)
    held_out_count = sum(case.split == "held_out" for case in dataset.cases)
    if not 20 <= diagnostic_count <= 30 or held_out_count < 1:
        raise ValueError("Dataset needs 20-30 diagnostic cases and preassigned held-out cases")
    if not dataset.held_out_frozen_before_tuning:
        raise ValueError("Held-out cases must be reserved before tuning")


def load_source_dataset(manifest_path: Path, *, project_root: Path) -> SourceDataset:
    """Load and verify source PDFs, manual case labels, and current chunk mappings."""
    try:
        dataset = SourceDataset.model_validate(json.loads(manifest_path.read_text("utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"Invalid source dataset manifest: {error}") from error

    _validate_cases(dataset)
    for document in dataset.documents:
        _validate_document_source(document, project_root)
        _validate_chunk_mappings(document, project_root)
    return dataset
