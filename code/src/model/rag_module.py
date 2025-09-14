"""
Retrieval-Augmented Generation (RAG) Module
Integrates dense passage retrieval with the MoE++ model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import faiss
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG module"""
    retriever_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    index_type: str = "faiss"  # faiss, chromadb, pinecone
    embedding_dim: int = 768
    num_retrieved_docs: int = 5
    max_doc_length: int = 512
    rerank_documents: bool = True
    use_multi_hop: bool = True
    max_hops: int = 3
    retrieval_temperature: float = 1.0
    fusion_method: str = "concatenate"  # concatenate, weighted_sum, cross_attention
    doc_separator_token: str = " [DOC] "
    use_compression: bool = True
    compression_ratio: float = 0.5

class DenseRetriever(nn.Module):
    """Dense passage retriever using bi-encoder architecture"""
    def __init__(self, config: RAGConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained encoder
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(config.retriever_model_name)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        except ImportError:
            logger.warning("sentence-transformers not available, using simple encoder")
            self.embedding_dim = config.embedding_dim
            self.encoder = SimpleEncoder(self.embedding_dim)
        
        # Query encoder (can be fine-tuned separately)
        self.query_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Document encoder (for reranking)
        self.doc_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries into dense vectors"""
        # Encode with base encoder
        if hasattr(self.encoder, 'encode'):
            embeddings = self.encoder.encode(queries, convert_to_tensor=True)
        else:
            embeddings = self.encoder(queries)
        
        # Apply query-specific transformation
        query_embeddings = self.query_encoder(embeddings)
        return F.normalize(query_embeddings, p=2, dim=-1)

    def encode_documents(self, documents: List[str]) -> torch.Tensor:
        """Encode documents into dense vectors"""
        # Encode with base encoder
        if hasattr(self.encoder, 'encode'):
            embeddings = self.encoder.encode(documents, convert_to_tensor=True)
        else:
            embeddings = self.encoder(documents)
        
        # Apply document-specific transformation
        doc_embeddings = self.doc_encoder(embeddings)
        return F.normalize(doc_embeddings, p=2, dim=-1)

class SimpleEncoder(nn.Module):
    """Simple encoder fallback when sentence-transformers not available"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(50000, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, 8, dim_feedforward=2048),
            num_layers=6
        )
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Simple tokenization (placeholder)
        max_len = 128
        batch_size = len(texts)
        input_ids = torch.randint(0, 50000, (batch_size, max_len))
        
        embeddings = self.embedding(input_ids)
        encoded = self.transformer(embeddings.transpose(0, 1))
        return encoded.mean(dim=0)  # Simple pooling

class VectorIndex:
    """Vector index for efficient similarity search"""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index = None
        self.documents = []
        self.document_embeddings = None
        
    def build_index(self, embeddings: np.ndarray, documents: List[str]):
        """Build FAISS index from embeddings"""
        self.documents = documents
        self.document_embeddings = embeddings
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Create index
        if self.config.index_type == "faiss":
            # Use IVF index for large datasets
            if len(documents) > 50000:
                quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.config.embedding_dim, 100)
                self.index.train(embeddings)
            else:
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)
            
            self.index.add(embeddings)
        else:
            raise NotImplementedError(f"Index type {self.config.index_type} not implemented")
    
    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        # Normalize query embeddings
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        return scores, indices
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        """Add new documents to the index"""
        if self.index is None:
            self.build_index(embeddings, documents)
        else:
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.documents.extend(documents)
            if self.document_embeddings is not None:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])

class DocumentReranker(nn.Module):
    """Neural document reranker using cross-encoder architecture"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.cross_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, dim_feedforward=2048),
            num_layers=2
        )
        self.score_head = nn.Linear(hidden_size, 1)
        
    def forward(self, query_embeds: torch.Tensor, doc_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute relevance scores for query-document pairs
        
        Args:
            query_embeds: [batch_size, hidden_size]
            doc_embeds: [batch_size, num_docs, hidden_size]
            
        Returns:
            scores: [batch_size, num_docs]
        """
        batch_size, num_docs, hidden_size = doc_embeds.shape
        
        # Expand query embeddings
        query_expanded = query_embeds.unsqueeze(1).expand(-1, num_docs, -1)
        
        # Concatenate query and document
        combined = torch.cat([query_expanded, doc_embeds], dim=-1)
        combined = combined.view(batch_size * num_docs, -1)
        
        # Cross-encode
        encoded = self.cross_encoder(combined.unsqueeze(0))
        scores = self.score_head(encoded.squeeze(0))
        scores = scores.view(batch_size, num_docs)
        
        return scores

class RAGModule(nn.Module):
    """
    Main RAG module integrating retrieval with generation
    """
    def __init__(self, config: RAGConfig, model_config):
        super().__init__()
        self.config = config
        self.model_config = model_config
        
        # Components
        self.retriever = DenseRetriever(config)
        self.vector_index = VectorIndex(config)
        self.reranker = DocumentReranker(model_config.hidden_size) if config.rerank_documents else None
        
        # Fusion layers
        if config.fusion_method == "cross_attention":
            self.doc_attention = nn.MultiheadAttention(
                model_config.hidden_size,
                num_heads=8,
                batch_first=True
            )
        elif config.fusion_method == "weighted_sum":
            self.doc_weights = nn.Linear(model_config.hidden_size, config.num_retrieved_docs)
        
        # Document compression
        if config.use_compression:
            self.doc_compressor = nn.Linear(
                model_config.hidden_size,
                int(model_config.hidden_size * config.compression_ratio)
            )
            self.doc_decompressor = nn.Linear(
                int(model_config.hidden_size * config.compression_ratio),
                model_config.hidden_size
            )

    def retrieve(self, queries: List[str], k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents for queries
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve (overrides config)
            
        Returns:
            Dictionary with retrieved documents and scores
        """
        k = k or self.config.num_retrieved_docs
        
        # Encode queries
        query_embeddings = self.retriever.encode_queries(queries)
        query_np = query_embeddings.cpu().numpy()
        
        # Search index
        scores, indices = self.vector_index.search(query_np, k)
        
        # Get documents
        retrieved_docs = []
        for batch_indices in indices:
            docs = [self.vector_index.documents[idx] for idx in batch_indices]
            retrieved_docs.append(docs)
        
        return {
            "documents": retrieved_docs,
            "scores": scores,
            "indices": indices,
            "query_embeddings": query_embeddings
        }

    def rerank(self, queries: List[str], documents: List[List[str]], 
               query_embeddings: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Rerank retrieved documents using cross-encoder"""
        if self.reranker is None:
            return {"documents": documents, "scores": None}
        
        # Encode if needed
        if query_embeddings is None:
            query_embeddings = self.retriever.encode_queries(queries)
        
        # Encode documents
        all_docs = [doc for doc_list in documents for doc in doc_list]
        doc_embeddings = self.retriever.encode_documents(all_docs)
        
        # Reshape for batch processing
        batch_size = len(queries)
        num_docs = len(documents[0])
        doc_embeddings = doc_embeddings.view(batch_size, num_docs, -1)
        
        # Compute reranking scores
        rerank_scores = self.reranker(query_embeddings, doc_embeddings)
        
        # Sort by score
        sorted_scores, sorted_indices = torch.sort(rerank_scores, dim=-1, descending=True)
        
        # Reorder documents
        reranked_docs = []
        for i, (indices, doc_list) in enumerate(zip(sorted_indices, documents)):
            reranked = [doc_list[idx] for idx in indices]
            reranked_docs.append(reranked)
        
        return {
            "documents": reranked_docs,
            "scores": sorted_scores,
            "original_scores": rerank_scores
        }

    def multi_hop_retrieval(self, initial_query: str, max_hops: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform multi-hop retrieval for complex queries
        
        Args:
            initial_query: Initial query string
            max_hops: Maximum number of retrieval hops
            
        Returns:
            Dictionary with all retrieved documents and hop information
        """
        max_hops = max_hops or self.config.max_hops
        all_documents = []
        all_scores = []
        hop_queries = [initial_query]
        
        current_query = initial_query
        
        for hop in range(max_hops):
            # Retrieve documents
            retrieval_results = self.retrieve([current_query])
            documents = retrieval_results["documents"][0]
            scores = retrieval_results["scores"][0]
            
            all_documents.extend(documents)
            all_scores.extend(scores)
            
            # Generate next query based on retrieved content
            if hop < max_hops - 1:
                # Simple approach: use top document to refine query
                # In practice, this would use the language model
                top_doc = documents[0]
                current_query = f"{current_query} {top_doc[:100]}"  # Use first 100 chars
                hop_queries.append(current_query)
        
        return {
            "documents": all_documents,
            "scores": all_scores,
            "hop_queries": hop_queries,
            "num_hops": len(hop_queries)
        }

    def format_retrieved_context(self, documents: List[List[str]], 
                                truncate: bool = True) -> List[str]:
        """Format retrieved documents for model input"""
        formatted_contexts = []
        
        for doc_list in documents:
            # Join documents with separator
            context = self.config.doc_separator_token.join(doc_list)
            
            # Truncate if needed
            if truncate and len(context) > self.config.max_doc_length:
                context = context[:self.config.max_doc_length] + "..."
            
            formatted_contexts.append(context)
        
        return formatted_contexts

    def compress_documents(self, doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compress document embeddings to save memory"""
        if not self.config.use_compression:
            return doc_embeddings
        
        compressed = self.doc_compressor(doc_embeddings)
        return compressed

    def decompress_documents(self, compressed_embeddings: torch.Tensor) -> torch.Tensor:
        """Decompress document embeddings"""
        if not self.config.use_compression:
            return compressed_embeddings
        
        decompressed = self.doc_decompressor(compressed_embeddings)
        return decompressed

    def fuse_with_model_hidden_states(self, hidden_states: torch.Tensor, 
                                     doc_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse retrieved document embeddings with model hidden states
        
        Args:
            hidden_states: Model hidden states [batch_size, seq_len, hidden_size]
            doc_embeddings: Document embeddings [batch_size, num_docs, hidden_size]
            
        Returns:
            Fused hidden states
        """
        if self.config.fusion_method == "concatenate":
            # Simple concatenation (requires modifying sequence length)
            batch_size, seq_len, hidden_size = hidden_states.shape
            num_docs = doc_embeddings.shape[1]
            
            # Flatten documents into sequence
            doc_flat = doc_embeddings.view(batch_size, num_docs, hidden_size)
            fused = torch.cat([hidden_states, doc_flat], dim=1)
            
        elif self.config.fusion_method == "weighted_sum":
            # Weighted sum of documents
            weights = self.doc_weights(hidden_states.mean(dim=1))  # [batch_size, num_docs]
            weights = F.softmax(weights, dim=-1)
            
            # Weighted average of documents
            weighted_docs = torch.bmm(weights.unsqueeze(1), doc_embeddings)  # [batch_size, 1, hidden_size]
            
            # Add to hidden states
            fused = hidden_states + weighted_docs
            
        elif self.config.fusion_method == "cross_attention":
            # Cross-attention between hidden states and documents
            attended_docs, _ = self.doc_attention(
                hidden_states,  # Query
                doc_embeddings,  # Key
                doc_embeddings   # Value
            )
            fused = hidden_states + attended_docs
        
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
        
        return fused

    def update_index(self, new_documents: List[str], new_embeddings: Optional[np.ndarray] = None):
        """Update the vector index with new documents"""
        if new_embeddings is None:
            # Encode new documents
            new_embeddings = self.retriever.encode_documents(new_documents)
            new_embeddings = new_embeddings.cpu().numpy()
        
        # Add to index
        self.vector_index.add_documents(new_embeddings, new_documents)
        logger.info(f"Added {len(new_documents)} documents to index")