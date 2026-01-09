"""
Document representation for RAG.

This module provides the Document dataclass used to store
retrieved documents with their embeddings and metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class Document:
    """
    A document with content, embedding, and metadata for RAG retrieval.

    Attributes:
        content: The text content of the document
        embedding: The vector embedding of the document [embedding_dim]
        metadata: Optional metadata (source, timestamp, etc.)
        doc_id: Optional unique identifier
        score: Retrieval similarity score (set after retrieval)
    """

    content: str
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    score: float = 0.0

    def __post_init__(self):
        """Validate document after initialization."""
        if not self.content:
            raise ValueError("Document content cannot be empty")

    def to_device(self, device: torch.device) -> "Document":
        """Move embedding to specified device."""
        if self.embedding is not None:
            return Document(
                content=self.content,
                embedding=self.embedding.to(device),
                metadata=self.metadata,
                doc_id=self.doc_id,
                score=self.score,
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], embedding: Optional[torch.Tensor] = None) -> "Document":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            embedding=embedding,
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id"),
            score=data.get("score", 0.0),
        )


@dataclass
class RetrievalResult:
    """
    Result from a retrieval operation.

    Contains the document and its similarity score.
    """

    document: Document
    score: float
    rank: int = 0

    def __lt__(self, other: "RetrievalResult") -> bool:
        """Compare by score (higher is better)."""
        return self.score < other.score
