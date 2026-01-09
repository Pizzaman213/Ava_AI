"""
Vector retrievers for RAG.

This module provides retriever implementations for FAISS and ChromaDB
vector stores, used to retrieve relevant documents for RAG.

Supported backends:
- FAISSRetriever: Fast, local vector search using Facebook's FAISS library
- ChromaDBRetriever: Persistent vector store with optional embedding functions
- InMemoryRetriever: Simple numpy-based retriever for testing

Usage:
    from ava.rag import FAISSRetriever, Document

    # Create retriever
    retriever = FAISSRetriever(embedding_dim=768)

    # Add documents
    docs = [Document("Hello world", embedding=torch.randn(768))]
    retriever.add_documents(docs)

    # Retrieve
    query = torch.randn(1, 768)
    results = retriever.retrieve(query, top_k=5)
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .document import Document, RetrievalResult

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """
    Abstract base class for document retrievers.

    All retrievers must implement:
    - retrieve(): Search for similar documents
    - add_documents(): Add documents to the index
    """

    @abstractmethod
    def retrieve(
        self, query_embeddings: torch.Tensor, top_k: int = 5
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve top-k documents for each query.

        Args:
            query_embeddings: Query vectors [batch_size, embedding_dim]
            top_k: Number of documents to retrieve per query

        Returns:
            List of lists of RetrievalResult objects, one per query
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever index.

        Args:
            documents: List of Document objects with embeddings
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of indexed documents."""
        pass


class FAISSRetriever(Retriever):
    """
    FAISS-based vector retriever.

    Uses Facebook's FAISS library for efficient similarity search.
    Supports both CPU and GPU operation, with optional normalization
    for cosine similarity.

    Args:
        embedding_dim: Dimension of document embeddings
        index_path: Optional path to load pre-built index
        normalize: Whether to L2-normalize vectors (for cosine similarity)
        use_gpu: Whether to use GPU for search (requires faiss-gpu)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_path: Optional[str] = None,
        normalize: bool = True,
        use_gpu: bool = False,
    ):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS is required for FAISSRetriever. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            )

        self.embedding_dim = embedding_dim
        self.normalize = normalize
        self.use_gpu = use_gpu
        self._faiss = faiss

        # Initialize or load index
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
        else:
            # Use inner product index for cosine similarity (with normalized vectors)
            # or L2 index for euclidean distance
            if normalize:
                self.index = faiss.IndexFlatIP(embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(embedding_dim)

        # Move to GPU if requested
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move FAISS index to GPU: {e}")

        # Document storage (FAISS only stores vectors)
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        if not documents:
            return

        # Extract and validate embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document '{doc.content[:50]}...' has no embedding")
            embeddings.append(doc.embedding.cpu().numpy())

        embeddings_np = np.vstack(embeddings).astype(np.float32)

        # Normalize if using cosine similarity
        if self.normalize:
            self._faiss.normalize_L2(embeddings_np)

        # Add to index
        self.index.add(embeddings_np)
        self.documents.extend(documents)

        logger.debug(f"Added {len(documents)} documents to FAISS index")

    def retrieve(
        self, query_embeddings: torch.Tensor, top_k: int = 5
    ) -> List[List[RetrievalResult]]:
        """Retrieve top-k documents for each query."""
        if len(self.documents) == 0:
            return [[] for _ in range(query_embeddings.size(0))]

        # Prepare query vectors
        query_np = query_embeddings.cpu().numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        # Normalize queries if using cosine similarity
        if self.normalize:
            self._faiss.normalize_L2(query_np)

        # Limit top_k to available documents
        k = min(top_k, len(self.documents))

        # Search
        scores, indices = self.index.search(query_np, k)

        # Build results
        results = []
        for i in range(query_np.shape[0]):
            query_results = []
            for rank, (score, idx) in enumerate(zip(scores[i], indices[i])):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    doc.score = float(score)
                    query_results.append(
                        RetrievalResult(document=doc, score=float(score), rank=rank)
                    )
            results.append(query_results)

        return results

    def save(self, path: str) -> None:
        """Save index to disk."""
        # Move to CPU for saving if on GPU
        index_to_save = self.index
        if self.use_gpu:
            index_to_save = self._faiss.index_gpu_to_cpu(self.index)

        self._faiss.write_index(index_to_save, path)
        logger.info(f"Saved FAISS index to {path}")

    def __len__(self) -> int:
        return len(self.documents)


class ChromaDBRetriever(Retriever):
    """
    ChromaDB-based vector retriever.

    Uses ChromaDB for persistent vector storage with optional
    embedding functions.

    Args:
        collection_name: Name of the ChromaDB collection
        persist_dir: Directory for persistent storage (None for in-memory)
        embedding_function: Optional embedding function for automatic encoding
    """

    def __init__(
        self,
        collection_name: str = "ava_rag",
        persist_dir: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required for ChromaDBRetriever. "
                "Install with: pip install chromadb"
            )

        self._chromadb = chromadb

        # Create client
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"ChromaDB using persistent storage at {persist_dir}")
        else:
            self.client = chromadb.Client()
            logger.info("ChromaDB using in-memory storage")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

        # Track document count
        self._doc_count = self.collection.count()

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the collection."""
        if not documents:
            return

        ids = []
        contents = []
        embeddings = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc_id = doc.doc_id or f"doc_{self._doc_count + i}"
            ids.append(doc_id)
            contents.append(doc.content)
            metadatas.append(doc.metadata or {})

            if doc.embedding is not None:
                embeddings.append(doc.embedding.cpu().numpy().tolist())

        # Add to collection
        if embeddings:
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            # Let ChromaDB compute embeddings
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
            )

        self._doc_count += len(documents)
        logger.debug(f"Added {len(documents)} documents to ChromaDB")

    def retrieve(
        self, query_embeddings: torch.Tensor, top_k: int = 5
    ) -> List[List[RetrievalResult]]:
        """Retrieve top-k documents for each query."""
        if self._doc_count == 0:
            return [[] for _ in range(query_embeddings.size(0))]

        # Convert to list format for ChromaDB
        query_np = query_embeddings.cpu().numpy()
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        query_list = query_np.tolist()

        # Query collection
        results = self.collection.query(
            query_embeddings=query_list,
            n_results=min(top_k, self._doc_count),
            include=["documents", "distances", "metadatas"],
        )

        # Build RetrievalResult objects
        all_results = []
        for i in range(len(query_list)):
            query_results = []
            if results["documents"] and results["documents"][i]:
                for rank, (doc_content, distance, metadata) in enumerate(
                    zip(
                        results["documents"][i],
                        results["distances"][i] if results["distances"] else [0.0] * len(results["documents"][i]),
                        results["metadatas"][i] if results["metadatas"] else [{}] * len(results["documents"][i]),
                    )
                ):
                    # Convert distance to similarity (ChromaDB returns L2 distance)
                    score = 1.0 / (1.0 + distance)

                    doc = Document(
                        content=doc_content,
                        metadata=metadata,
                        score=score,
                    )
                    query_results.append(
                        RetrievalResult(document=doc, score=score, rank=rank)
                    )
            all_results.append(query_results)

        return all_results

    def __len__(self) -> int:
        return self._doc_count


class InMemoryRetriever(Retriever):
    """
    Simple in-memory retriever using numpy for testing.

    Uses cosine similarity for retrieval. Not optimized for
    large-scale use.

    Args:
        embedding_dim: Dimension of embeddings
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever."""
        if not documents:
            return

        new_embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document has no embedding")
            new_embeddings.append(doc.embedding.cpu().numpy())

        new_embeddings_np = np.vstack(new_embeddings).astype(np.float32)

        # Normalize
        norms = np.linalg.norm(new_embeddings_np, axis=1, keepdims=True)
        new_embeddings_np = new_embeddings_np / (norms + 1e-8)

        if self.embeddings is None:
            self.embeddings = new_embeddings_np
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_np])

        self.documents.extend(documents)

    def retrieve(
        self, query_embeddings: torch.Tensor, top_k: int = 5
    ) -> List[List[RetrievalResult]]:
        """Retrieve using cosine similarity."""
        if len(self.documents) == 0 or self.embeddings is None:
            return [[] for _ in range(query_embeddings.size(0))]

        # Prepare queries
        query_np = query_embeddings.cpu().numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        # Normalize queries
        norms = np.linalg.norm(query_np, axis=1, keepdims=True)
        query_np = query_np / (norms + 1e-8)

        # Compute cosine similarity
        similarities = np.dot(query_np, self.embeddings.T)

        # Get top-k
        k = min(top_k, len(self.documents))
        results = []

        for i in range(query_np.shape[0]):
            scores = similarities[i]
            top_indices = np.argsort(scores)[::-1][:k]

            query_results = []
            for rank, idx in enumerate(top_indices):
                doc = self.documents[idx]
                doc.score = float(scores[idx])
                query_results.append(
                    RetrievalResult(document=doc, score=float(scores[idx]), rank=rank)
                )
            results.append(query_results)

        return results

    def __len__(self) -> int:
        return len(self.documents)


def create_retriever(config: Any) -> Retriever:
    """
    Factory function to create a retriever from config.

    Args:
        config: RAGConfig with retriever settings

    Returns:
        Retriever instance
    """
    retriever_type = getattr(config, "retriever_type", "faiss").lower()
    embedding_dim = getattr(config, "embedding_dim", 768)
    index_path = getattr(config, "index_path", None)
    normalize = getattr(config, "normalize_embeddings", True)

    if retriever_type == "faiss":
        return FAISSRetriever(
            embedding_dim=embedding_dim,
            index_path=index_path,
            normalize=normalize,
        )
    elif retriever_type == "chromadb":
        persist_dir = getattr(config, "chromadb_persist_dir", None)
        collection_name = getattr(config, "chromadb_collection", "ava_rag")
        return ChromaDBRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
    elif retriever_type == "memory":
        return InMemoryRetriever(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
