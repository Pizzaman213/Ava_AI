"""
RAG (Retrieval Augmented Generation) package for Ava.

This package provides components for retrieval-augmented generation:
- Document storage and representation
- Vector retrievers (FAISS, ChromaDB)
- Fusion strategies for combining retrieved context
- Index building utilities

Usage:
    from ava.rag import FAISSRetriever, Document, GatedFusion

    # Create retriever
    retriever = FAISSRetriever(embedding_dim=768)

    # Add documents
    docs = [Document("Hello world", embedding=torch.randn(768))]
    retriever.add_documents(docs)

    # Retrieve
    results = retriever.retrieve(query_embeddings, top_k=5)

    # Fuse with model hidden states
    fusion = GatedFusion(hidden_size=768)
    fused = fusion(hidden_states, retrieved_context)
"""

from .document import Document, RetrievalResult
from .retriever import (
    Retriever,
    FAISSRetriever,
    ChromaDBRetriever,
    InMemoryRetriever,
    create_retriever,
)
from .fusion import (
    RAGFusion,
    ConcatFusion,
    GatedFusion,
    CrossAttentionFusion,
    AdaptiveFusion,
    create_fusion,
)
from .index import (
    VectorIndex,
    IndexBuilder,
)

__all__ = [
    # Documents
    "Document",
    "RetrievalResult",
    # Retrievers
    "Retriever",
    "FAISSRetriever",
    "ChromaDBRetriever",
    "InMemoryRetriever",
    "create_retriever",
    # Fusion
    "RAGFusion",
    "ConcatFusion",
    "GatedFusion",
    "CrossAttentionFusion",
    "AdaptiveFusion",
    "create_fusion",
    # Index
    "VectorIndex",
    "IndexBuilder",
]
