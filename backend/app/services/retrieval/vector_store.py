"""ChromaDB vector store for dense semantic retrieval."""

import hashlib
import logging
import math
from functools import partial
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.core.config import get_settings
from app.schemas.document import ChunkMetadata, DocumentChunk, ScoredChunk
from app.services.evaluation.usage import observed_embeddings

logger = logging.getLogger(__name__)


class DeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic local embedding function for offline testing and development.

    Produces normalized float vectors in dimension D using character/word hashing.
    Enables reproducible cosine similarity without requiring external API keys.
    """

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of text documents into normalized vector representations."""
        embeddings: Embeddings = []
        for text in input:
            vec = [0.0] * self._dimension
            tokens = text.lower().split()
            if not tokens:
                vec[0] = 1.0
            else:
                for token in tokens:
                    h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                    idx = h % self._dimension
                    sign = 1.0 if (h >> 16) & 1 else -1.0
                    vec[idx] += sign
                norm = math.sqrt(sum(x * x for x in vec))
                if norm > 0.0:
                    vec = [x / norm for x in vec]
                else:
                    vec[0] = 1.0
            embeddings.append(np.array(vec, dtype=np.float32))
        return embeddings


class VectorStore:
    """Manages persistent ChromaDB vector storage and scoped dense semantic retrieval."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "document_chunks",
        embedding_function: Any | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()

        if client is not None:
            self._client = client
        else:
            resolved_dir = (
                Path(persist_dir) if persist_dir is not None else settings.resolved_chroma_dir
            )
            resolved_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(resolved_dir))

        self._collection_name = collection_name

        if embedding_function is not None:
            self._embedding_function = embedding_function
        elif settings.OPENAI_API_KEY:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            ef_kwargs: dict[str, Any] = {
                "api_key": settings.OPENAI_API_KEY,
                "model_name": settings.EMBEDDING_MODEL,
            }
            if settings.OPENAI_API_BASE:
                ef_kwargs["api_base"] = settings.OPENAI_API_BASE

            self._embedding_function = OpenAIEmbeddingFunction(**ef_kwargs)
        else:
            self._embedding_function = DeterministicEmbeddingFunction(
                dimension=settings.EMBEDDING_DIMENSION
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]] | None = None,
        batch_size: int = 16,
    ) -> None:
        """Add or update document chunks in the vector collection in safe batches.

        Args:
            chunks: List of document chunks to insert/upsert.
            embeddings: Optional precomputed embeddings matching the chunks.
            batch_size: Maximum chunk batch size per ChromaDB upsert call (default: 16,
                conforming to Alibaba Cloud DashScope embedding API <= 20 item limit).
        """
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.metadata.model_dump() for c in chunks]

        step = max(1, batch_size)
        for i in range(0, len(chunks), step):
            batch_ids = ids[i : i + step]
            batch_documents = documents[i : i + step]
            batch_metadatas = metadatas[i : i + step]

            if embeddings is not None:
                batch_embeddings = embeddings[i : i + step]
                self._collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=batch_embeddings,
                )
            else:
                observed_embeddings(
                    "ingestion_embedding",
                    model=str(
                        getattr(
                            self._embedding_function,
                            "model_name",
                            type(self._embedding_function).__name__,
                        )
                    ),
                    texts=batch_documents,
                    invoke=partial(
                        self._collection.upsert,
                        ids=batch_ids,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                    ),
                    client=getattr(self._embedding_function, "client", None),
                    local_only=isinstance(self._embedding_function, DeterministicEmbeddingFunction),
                )

    def query(
        self,
        query_text: str,
        document_ids: list[str] | None = None,
        top_n: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list[ScoredChunk]:
        """Perform scoped semantic search over the vector collection."""
        total_chunks = self.count()
        if total_chunks == 0:
            return []

        # Document scoping logic according to DESIGN.md Section 3.3.1
        where_filter: dict[str, Any] | None = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        n_results = min(top_n, total_chunks)
        if n_results <= 0:
            return []

        if query_embedding is not None:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = observed_embeddings(
                "query_embedding",
                model=str(
                    getattr(
                        self._embedding_function,
                        "model_name",
                        type(self._embedding_function).__name__,
                    )
                ),
                texts=[query_text],
                invoke=lambda: self._collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                ),
                client=getattr(self._embedding_function, "client", None),
                local_only=isinstance(self._embedding_function, DeterministicEmbeddingFunction),
            )

        scored_chunks: list[ScoredChunk] = []
        ids_list = results.get("ids", [])
        if not ids_list or not ids_list[0]:
            return []

        ids = ids_list[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            content = documents[i] if i < len(documents) and documents[i] is not None else ""
            meta_dict = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            dist = distances[i] if i < len(distances) and distances[i] is not None else 0.0

            # Cosine distance in Chroma is 1 - cosine_similarity
            similarity_score = 1.0 - dist
            chunk_meta = ChunkMetadata.model_validate(meta_dict)

            scored_chunks.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=chunk_meta,
                    score=similarity_score,
                )
            )

        return scored_chunks

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document_id."""
        self._collection.delete(where={"document_id": document_id})

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Retrieve all stored chunks from the persistent vector collection."""
        total = self.count()
        if total == 0:
            return []

        results = self._collection.get(include=["documents", "metadatas"])
        chunks: list[DocumentChunk] = []
        ids_list = results.get("ids", [])
        docs_list = results.get("documents", [])
        metas_list = results.get("metadatas", [])

        for i, chunk_id in enumerate(ids_list):
            content = (
                docs_list[i]
                if docs_list and i < len(docs_list) and docs_list[i] is not None
                else ""
            )
            meta_dict = (
                metas_list[i]
                if metas_list and i < len(metas_list) and metas_list[i] is not None
                else {}
            )
            chunk_meta = ChunkMetadata.model_validate(meta_dict)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=chunk_meta,
                )
            )

        return chunks

    def count(self) -> int:
        """Return the total number of chunks stored in this collection."""
        return int(self._collection.count())

    def reset(self) -> None:
        """Clear all records in the collection by recreating it."""
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception as err:
            logger.debug("Collection not found or already deleted during reset: %s", err)

        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )


_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the global singleton instance of the VectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


def set_vector_store(store: VectorStore | None) -> None:
    """Set or reset the global singleton instance of the VectorStore (for testing/DI)."""
    global _vector_store_instance
    _vector_store_instance = store
