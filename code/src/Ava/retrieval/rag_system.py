"""
Retrieval-Augmented Generation (RAG) System for enhanced knowledge access.

This module implements a comprehensive RAG system that can retrieve relevant
information from external knowledge bases and integrate it into the generation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import faiss
import pickle
from pathlib import Path
import json


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    passages: List[str]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    embeddings: Optional[torch.Tensor] = None


class DenseRetriever(nn.Module):
    """
    Dense retrieval system using learned embeddings.

    This retriever uses a dual-encoder architecture to embed both
    queries and documents in the same dense vector space.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        normalize_embeddings: bool = True
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.normalize_embeddings = normalize_embeddings

        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Document encoder (shared weights with query encoder)
        self.doc_encoder = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Projection layer for contrastive learning
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def encode_query(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode query into dense vector."""
        encoded = self.query_encoder(query_embeddings)
        if self.normalize_embeddings:
            encoded = F.normalize(encoded, p=2, dim=-1)
        return encoded

    def encode_document(self, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode document into dense vector."""
        encoded = self.doc_encoder(doc_embeddings)
        if self.normalize_embeddings:
            encoded = F.normalize(encoded, p=2, dim=-1)
        return encoded

    def compute_similarity(
        self,
        query_embeds: torch.Tensor,
        doc_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between query and document embeddings."""
        query_proj = self.projection(query_embeds)
        doc_proj = self.projection(doc_embeds)

        if self.normalize_embeddings:
            query_proj = F.normalize(query_proj, p=2, dim=-1)
            doc_proj = F.normalize(doc_proj, p=2, dim=-1)

        return torch.matmul(query_proj, doc_proj.transpose(-2, -1))


class KnowledgeBase:
    """
    Vector database for storing and retrieving documents.

    This class manages the storage and retrieval of document embeddings
    using FAISS for efficient similarity search.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "IVF",
        n_clusters: int = 1024,
        device: str = "cpu"
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.device = device

        # Initialize FAISS index
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_clusters)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

        # Storage for documents and metadata
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.is_trained = False

    def add_documents(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the knowledge base."""
        if not self.is_trained and self.index_type == "IVF":
            # Train the index if using IVF
            if embeddings.shape[0] >= self.n_clusters:
                self.index.train(embeddings.astype(np.float32))
                self.is_trained = True

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))

        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.0
    ) -> RetrievalResult:
        """Search for similar documents."""
        query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]  # Remove batch dimension
        indices = indices[0]

        # Filter by threshold
        valid_mask = scores >= threshold
        scores = scores[valid_mask]
        indices = indices[valid_mask]

        # Retrieve documents and metadata
        retrieved_docs = [self.documents[idx] for idx in indices if idx < len(self.documents)]
        retrieved_metadata = [self.metadata[idx] for idx in indices if idx < len(self.metadata)]

        return RetrievalResult(
            passages=retrieved_docs,
            scores=scores.tolist(),
            metadata=retrieved_metadata
        )

    def save(self, path: Path):
        """Save the knowledge base to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss_index.idx"))

        # Save documents and metadata
        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f)

        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def load(self, path: Path):
        """Load the knowledge base from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss_index.idx"))

        # Load documents and metadata
        with open(path / "documents.json", "r") as f:
            self.documents = json.load(f)

        with open(path / "metadata.json", "r") as f:
            self.metadata = json.load(f)


class RAGFusion(nn.Module):
    """
    Fusion module for combining retrieved information with generated content.

    This module implements various fusion strategies to integrate
    retrieved knowledge into the generation process.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_retrieved: int = 5,
        fusion_type: str = "attention",
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_retrieved = num_retrieved
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

        elif fusion_type == "gate":
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )

        elif fusion_type == "concat":
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_repr: torch.Tensor,
        retrieved_repr: torch.Tensor,
        retrieval_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse query representation with retrieved information.

        Args:
            query_repr: Query representation [batch, seq_len, hidden_dim]
            retrieved_repr: Retrieved passages [batch, num_retrieved, hidden_dim]
            retrieval_scores: Retrieval scores [batch, num_retrieved]
        """
        batch_size, seq_len, hidden_dim = query_repr.shape

        if self.fusion_type == "attention":
            # Use cross-attention to fuse information
            fused, _ = self.fusion_attention(
                query=query_repr,
                key=retrieved_repr,
                value=retrieved_repr
            )
            fused = self.norm(fused + query_repr)  # Residual connection

        elif self.fusion_type == "gate":
            # Gated fusion
            expanded_retrieved = retrieved_repr.mean(dim=1, keepdim=True)  # Average retrieval
            expanded_retrieved = expanded_retrieved.expand(-1, seq_len, -1)

            concat_repr = torch.cat([query_repr, expanded_retrieved], dim=-1)
            gate = self.gate_network(concat_repr)
            fused = gate * query_repr + (1 - gate) * expanded_retrieved

        elif self.fusion_type == "concat":
            # Simple concatenation and projection
            expanded_retrieved = retrieved_repr.mean(dim=1, keepdim=True)
            expanded_retrieved = expanded_retrieved.expand(-1, seq_len, -1)

            concat_repr = torch.cat([query_repr, expanded_retrieved], dim=-1)
            fused = self.projection(concat_repr)

        else:
            # Weighted sum fusion
            if retrieval_scores is not None:
                weights = F.softmax(retrieval_scores, dim=-1).unsqueeze(-1)
                weighted_retrieved = (retrieved_repr * weights).sum(dim=1, keepdim=True)
            else:
                weighted_retrieved = retrieved_repr.mean(dim=1, keepdim=True)

            weighted_retrieved = weighted_retrieved.expand(-1, seq_len, -1)
            fused = query_repr + weighted_retrieved

        return self.dropout(fused)


class RAGSystem(nn.Module):
    """
    Complete RAG (Retrieval-Augmented Generation) system.

    This system combines dense retrieval, knowledge base management,
    and fusion mechanisms for enhanced text generation.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        retrieval_dim: int = 256,
        max_retrieved: int = 10,
        fusion_type: str = "attention",
        retrieval_threshold: float = 0.5,
        use_reranking: bool = True
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.retrieval_dim = retrieval_dim
        self.max_retrieved = max_retrieved
        self.retrieval_threshold = retrieval_threshold
        self.use_reranking = use_reranking

        # Dense retriever
        self.retriever = DenseRetriever(
            encoder_dim=encoder_dim,
            hidden_dim=retrieval_dim
        )

        # Fusion module
        self.fusion = RAGFusion(
            hidden_dim=encoder_dim,
            num_retrieved=max_retrieved,
            fusion_type=fusion_type
        )

        # Reranking module (optional)
        if use_reranking:
            self.reranker = nn.Sequential(
                nn.Linear(encoder_dim * 2, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, 1)
            )

        # Knowledge base (initialized externally)
        self.knowledge_base: Optional[KnowledgeBase] = None

    def set_knowledge_base(self, knowledge_base: KnowledgeBase):
        """Set the knowledge base for retrieval."""
        self.knowledge_base = knowledge_base

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        k: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant passages for the given query.

        Args:
            query_embeddings: Query embeddings [batch_size, seq_len, embed_dim]
            k: Number of passages to retrieve

        Returns:
            RetrievalResult with retrieved passages and scores
        """
        if self.knowledge_base is None:
            raise ValueError("Knowledge base not set. Call set_knowledge_base() first.")

        k = k or self.max_retrieved
        batch_size = query_embeddings.shape[0]

        # Encode query
        query_repr = self.retriever.encode_query(query_embeddings.mean(dim=1))  # Pool sequence

        # Convert to numpy for FAISS
        query_np = query_repr.detach().cpu().numpy()

        # Retrieve for each query in batch
        all_results = []
        for i in range(batch_size):
            result = self.knowledge_base.search(
                query_np[i], k=k, threshold=self.retrieval_threshold
            )
            all_results.append(result)

        # Combine results
        combined_passages = []
        combined_scores = []
        combined_metadata = []

        for result in all_results:
            combined_passages.extend(result.passages)
            combined_scores.extend(result.scores)
            combined_metadata.extend(result.metadata)

        return RetrievalResult(
            passages=combined_passages,
            scores=combined_scores,
            metadata=combined_metadata
        )

    def rerank(
        self,
        query_repr: torch.Tensor,
        retrieved_repr: torch.Tensor,
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rerank retrieved passages using learned similarity."""
        if not self.use_reranking:
            return retrieved_repr, scores

        batch_size, num_retrieved, hidden_dim = retrieved_repr.shape
        query_expanded = query_repr.unsqueeze(1).expand(-1, num_retrieved, -1)

        # Compute reranking scores
        concat_features = torch.cat([query_expanded, retrieved_repr], dim=-1)
        rerank_scores = self.reranker(concat_features).squeeze(-1)

        # Combine with original scores
        combined_scores = 0.7 * scores + 0.3 * rerank_scores

        # Reorder by combined scores
        sorted_indices = torch.argsort(combined_scores, dim=-1, descending=True)

        reranked_repr = torch.gather(
            retrieved_repr, 1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        reranked_scores = torch.gather(combined_scores, 1, sorted_indices)

        return reranked_repr, reranked_scores

    def forward(
        self,
        query_embeddings: torch.Tensor,
        retrieved_passages: Optional[torch.Tensor] = None,
        retrieval_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the RAG system.

        Args:
            query_embeddings: Query embeddings [batch, seq_len, embed_dim]
            retrieved_passages: Pre-retrieved passage embeddings [batch, num_passages, embed_dim]
            retrieval_scores: Retrieval scores [batch, num_passages]

        Returns:
            Tuple of (enhanced_embeddings, retrieval_info)
        """
        if retrieved_passages is None:
            # Perform retrieval
            retrieval_result = self.retrieve(query_embeddings)
            # Note: In practice, you'd need to encode the retrieved passages
            # This is a simplified version
            retrieved_passages = torch.randn(
                query_embeddings.shape[0], self.max_retrieved, self.encoder_dim,
                device=query_embeddings.device
            )
            retrieval_scores = torch.tensor(
                retrieval_result.scores[:self.max_retrieved],
                device=query_embeddings.device
            ).unsqueeze(0).expand(query_embeddings.shape[0], -1)

        # Rerank if enabled
        if self.use_reranking:
            query_pooled = query_embeddings.mean(dim=1)  # Pool sequence dimension
            retrieved_passages, retrieval_scores = self.rerank(
                query_pooled, retrieved_passages, retrieval_scores
            )

        # Fuse query with retrieved information
        enhanced_embeddings = self.fusion(
            query_embeddings, retrieved_passages, retrieval_scores
        )

        retrieval_info = {
            'num_retrieved': retrieved_passages.shape[1],
            'avg_retrieval_score': retrieval_scores.mean().item(),
            'max_retrieval_score': retrieval_scores.max().item()
        }

        return enhanced_embeddings, retrieval_info


class AdaptiveRAG(nn.Module):
    """
    Adaptive RAG system that decides when to use retrieval.

    This module learns when retrieval is beneficial and can
    dynamically enable/disable retrieval based on the input.
    """

    def __init__(
        self,
        rag_system: RAGSystem,
        confidence_threshold: float = 0.7,
        retrieval_probability: float = 0.8
    ):
        super().__init__()
        self.rag_system = rag_system
        self.confidence_threshold = confidence_threshold
        self.retrieval_probability = retrieval_probability

        # Retrieval decision network
        self.decision_network = nn.Sequential(
            nn.Linear(rag_system.encoder_dim, rag_system.encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(rag_system.encoder_dim // 2, 1),
            nn.Sigmoid()
        )

    def should_retrieve(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Decide whether to perform retrieval for each query."""
        # Compute confidence/uncertainty scores
        confidence_scores = self.decision_network(query_embeddings.mean(dim=1))

        # Use threshold-based decision
        retrieve_mask = confidence_scores.squeeze(-1) < self.confidence_threshold

        return retrieve_mask

    def forward(
        self,
        query_embeddings: torch.Tensor,
        force_retrieval: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Adaptive forward pass that conditionally uses retrieval.
        """
        if force_retrieval:
            return self.rag_system(query_embeddings)

        # Decide whether to retrieve
        retrieve_mask = self.should_retrieve(query_embeddings)

        if retrieve_mask.any():
            # Perform RAG for queries that need it
            enhanced_embeddings, retrieval_info = self.rag_system(query_embeddings)
            retrieval_info['retrieval_used'] = True
            retrieval_info['retrieval_decisions'] = retrieve_mask
        else:
            # Skip retrieval
            enhanced_embeddings = query_embeddings
            retrieval_info = {
                'retrieval_used': False,
                'retrieval_decisions': retrieve_mask
            }

        return enhanced_embeddings, retrieval_info