"""Fail closed on damaged live storage and recover journaled writes."""

from pathlib import Path

import pytest

from app.services.access import AccessStore
from app.services.workspaces import WorkspaceManager


def test_corrupt_workspace_never_becomes_empty(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    manager = WorkspaceManager(tmp_path, access, None)
    bundle = manager.get(workspace)
    (bundle.root / "documents.json").write_text("broken", encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        bundle.documents.get_all()


def test_same_ids_remain_in_separate_registries(tmp_path: Path) -> None:
    from app.schemas.document import DocumentItem

    access = AccessStore(tmp_path / "control.sqlite3")
    a, _ = access.create_invitation()
    b, _ = access.create_invitation()
    manager = WorkspaceManager(tmp_path, access, None)
    manager.get(a).documents.add_or_update(DocumentItem(id="equal", filename="same.pdf"))
    assert manager.get(b).documents.get_all() == []


def test_missing_existing_manifest_is_rejected(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    manager = WorkspaceManager(tmp_path, access, None)
    root = manager.get(workspace).root
    (root / "manifest.json").unlink()
    with pytest.raises((ValueError, OSError)):
        WorkspaceManager(tmp_path, access, None).get(workspace)


def test_interrupted_replacement_restores_pdf_registry_and_vectors(tmp_path: Path) -> None:
    from app.schemas.document import ChunkMetadata, DocumentChunk, DocumentItem

    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    manager = WorkspaceManager(tmp_path, access, None)
    bundle = manager.get(workspace)
    identifier = "doc_" + "a" * 32
    chunk = DocumentChunk(
        chunk_id=identifier + "_p1_c1",
        content="quartz old evidence",
        metadata=ChunkMetadata(
            chunk_id=identifier + "_p1_c1",
            document_id=identifier,
            filename="same.pdf",
            page_number=1,
            section_title="Test",
            token_count=3,
            char_start=0,
            char_end=19,
        ),
    )
    vector = [1.0] + [0.0] * 1023
    bundle.documents.add_or_update(DocumentItem(id=identifier, filename="same.pdf", chunk_count=1))
    bundle.vectors.add_chunks([chunk], embeddings=[vector])
    pdf = bundle.root / "uploads" / f"{identifier}.pdf"
    pdf.write_bytes(b"old PDF")
    bundle.begin_mutation(identifier)
    bundle.vectors.delete_document(identifier)
    bundle.documents.remove(identifier)
    pdf.write_bytes(b"interrupted replacement")
    recovered = WorkspaceManager(tmp_path, access, None).get(workspace)
    assert pdf.read_bytes() == b"old PDF"
    assert recovered.documents.get(identifier) is not None
    assert recovered.vectors.query("quartz", query_embedding=vector)[0].content == chunk.content
    assert recovered.bm25.query("quartz")[0].content == chunk.content
    assert not (bundle.root / "staging/mutation.json").exists()


def test_missing_control_database_cannot_reset_populated_storage(tmp_path: Path) -> None:
    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    WorkspaceManager(tmp_path, access, None).get(workspace)
    access.path.unlink()
    with pytest.raises(OSError):
        AccessStore(access.path)


def test_lost_vectors_cannot_leave_document_marked_ready(tmp_path: Path) -> None:
    from app.schemas.document import DocumentItem

    access = AccessStore(tmp_path / "control.sqlite3")
    workspace, _ = access.create_invitation()
    bundle = WorkspaceManager(tmp_path, access, None).get(workspace)
    bundle.documents.add_or_update(
        DocumentItem(id="doc_" + "b" * 32, filename="lost.pdf", chunk_count=1)
    )
    with pytest.raises(ValueError):
        WorkspaceManager(tmp_path, access, None).get(workspace)
