"""
Index building utilities for RAG.

This module provides utilities for building and managing vector indices
from various data sources.

Usage:
    from ava.rag import IndexBuilder

    # Build index from JSONL file
    builder = IndexBuilder(encoder_model="sentence-transformers/all-MiniLM-L6-v2")
    index = builder.build_from_jsonl("docs.jsonl", text_field="text")

    # Save for later use
    index.save("my_index")
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch

from .document import Document
from .retriever import FAISSRetriever, Retriever, create_retriever

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    Wrapper class for managing a vector index with documents.

    Provides utilities for building, saving, and loading indices.
    """

    def __init__(
        self,
        retriever: Retriever,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize vector index.

        Args:
            retriever: Underlying retriever implementation
            metadata: Optional metadata about the index
        """
        self.retriever = retriever
        self.metadata = metadata or {}

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the index."""
        self.retriever.add_documents(documents)

    def retrieve(
        self, query_embeddings: torch.Tensor, top_k: int = 5
    ) -> List[List[Any]]:
        """Retrieve documents for queries."""
        return self.retriever.retrieve(query_embeddings, top_k)

    def save(self, path: str) -> None:
        """
        Save index to disk.

        Args:
            path: Directory path to save index
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save retriever (if FAISS)
        if isinstance(self.retriever, FAISSRetriever):
            self.retriever.save(str(save_dir / "faiss.index"))

        # Save documents
        docs_data = []
        for doc in self.retriever.documents:
            docs_data.append(doc.to_dict())

        with open(save_dir / "documents.json", "w") as f:
            json.dump(docs_data, f)

        # Save metadata
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f)

        logger.info(f"Saved index to {path}")

    @classmethod
    def load(cls, path: str, embedding_dim: int = 768) -> "VectorIndex":
        """
        Load index from disk.

        Args:
            path: Directory path containing saved index
            embedding_dim: Embedding dimension (for FAISS)

        Returns:
            Loaded VectorIndex
        """
        load_dir = Path(path)

        # Load metadata
        with open(load_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Create retriever and load index
        faiss_path = load_dir / "faiss.index"
        if faiss_path.exists():
            retriever = FAISSRetriever(
                embedding_dim=embedding_dim,
                index_path=str(faiss_path),
            )
        else:
            retriever = FAISSRetriever(embedding_dim=embedding_dim)

        # Load documents
        with open(load_dir / "documents.json") as f:
            docs_data = json.load(f)

        # Note: Documents loaded from disk don't have embeddings
        # (they're in the FAISS index)
        retriever.documents = [
            Document.from_dict(d) for d in docs_data
        ]

        logger.info(f"Loaded index from {path} ({len(retriever.documents)} documents)")

        return cls(retriever=retriever, metadata=metadata)

    def __len__(self) -> int:
        return len(self.retriever)


class IndexBuilder:
    """
    Builder for creating vector indices from various data sources.

    Supports:
    - JSONL files
    - Arrow/Parquet files
    - Text files and directories
    - Custom iterators

    Uses sentence-transformers for encoding by default, but can
    accept custom encoder functions.
    """

    def __init__(
        self,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_fn: Optional[Callable[[List[str]], torch.Tensor]] = None,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Initialize index builder.

        Args:
            encoder_model: Sentence-transformers model name
            encoder_fn: Custom encoder function (overrides encoder_model)
            device: Device for encoding
            batch_size: Batch size for encoding
        """
        self.device = device
        self.batch_size = batch_size
        self.encoder_fn = encoder_fn

        if encoder_fn is None:
            self._init_sentence_transformer(encoder_model)
        else:
            self.encoder = None
            self.embedding_dim = None  # Must be set when using custom encoder

    def _init_sentence_transformer(self, model_name: str) -> None:
        """Initialize sentence-transformers encoder."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for IndexBuilder. "
                "Install with: pip install sentence-transformers"
            )

        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        logger.info(
            f"Initialized encoder: {model_name} (dim={self.embedding_dim})"
        )

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings tensor [num_texts, embedding_dim]
        """
        if self.encoder_fn is not None:
            return self.encoder_fn(texts)

        if self.encoder is None:
            raise ValueError("No encoder available")

        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_tensor=True,
        )

        return embeddings

    def build_from_jsonl(
        self,
        jsonl_path: str,
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
        max_documents: Optional[int] = None,
        retriever_type: str = "faiss",
    ) -> VectorIndex:
        """
        Build index from JSONL file.

        Args:
            jsonl_path: Path to JSONL file
            text_field: Field containing text content
            metadata_fields: Fields to include in metadata
            max_documents: Maximum documents to index
            retriever_type: Type of retriever to use

        Returns:
            Built VectorIndex
        """
        logger.info(f"Building index from {jsonl_path}")

        documents = []
        texts = []

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_documents and i >= max_documents:
                    break

                data = json.loads(line)
                text = data.get(text_field, "")

                if not text:
                    continue

                metadata = {}
                if metadata_fields:
                    for field in metadata_fields:
                        if field in data:
                            metadata[field] = data[field]

                documents.append(
                    Document(
                        content=text,
                        metadata=metadata,
                        doc_id=f"doc_{i}",
                    )
                )
                texts.append(text)

        if not documents:
            logger.warning("No documents found in JSONL file")
            retriever = FAISSRetriever(embedding_dim=self.embedding_dim or 768)
            return VectorIndex(retriever=retriever)

        # Encode texts
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.encode(texts)

        # Assign embeddings to documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        # Create retriever and add documents
        if retriever_type == "faiss":
            retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
        else:
            retriever = create_retriever(type(
                "Config", (), {"retriever_type": retriever_type}
            ))

        retriever.add_documents(documents)

        return VectorIndex(
            retriever=retriever,
            metadata={
                "source": jsonl_path,
                "text_field": text_field,
                "num_documents": len(documents),
            },
        )

    def build_from_arrow(
        self,
        arrow_path: str,
        text_column: str = "text",
        metadata_columns: Optional[List[str]] = None,
        max_documents: Optional[int] = None,
    ) -> VectorIndex:
        """
        Build index from Arrow/Parquet file.

        Args:
            arrow_path: Path to Arrow or Parquet file
            text_column: Column containing text content
            metadata_columns: Columns to include in metadata
            max_documents: Maximum documents to index

        Returns:
            Built VectorIndex
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Arrow files. "
                "Install with: pip install pyarrow"
            )

        logger.info(f"Building index from {arrow_path}")

        # Read Arrow/Parquet file
        table = pq.read_table(arrow_path)

        if text_column not in table.column_names:
            raise ValueError(f"Column '{text_column}' not found in Arrow file")

        documents = []
        texts = []

        text_col = table.column(text_column)

        for i in range(len(table)):
            if max_documents and i >= max_documents:
                break

            text = str(text_col[i].as_py())

            if not text:
                continue

            metadata = {}
            if metadata_columns:
                for col in metadata_columns:
                    if col in table.column_names:
                        metadata[col] = table.column(col)[i].as_py()

            documents.append(
                Document(
                    content=text,
                    metadata=metadata,
                    doc_id=f"doc_{i}",
                )
            )
            texts.append(text)

        if not documents:
            logger.warning("No documents found in Arrow file")
            retriever = FAISSRetriever(embedding_dim=self.embedding_dim or 768)
            return VectorIndex(retriever=retriever)

        # Encode texts
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.encode(texts)

        # Assign embeddings to documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        # Create retriever and add documents
        retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
        retriever.add_documents(documents)

        return VectorIndex(
            retriever=retriever,
            metadata={
                "source": arrow_path,
                "text_column": text_column,
                "num_documents": len(documents),
            },
        )

    def build_from_texts(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> VectorIndex:
        """
        Build index from list of texts.

        Args:
            texts: List of text strings
            metadata_list: Optional list of metadata dicts

        Returns:
            Built VectorIndex
        """
        logger.info(f"Building index from {len(texts)} texts")

        # Encode texts
        embeddings = self.encode(texts)

        # Create documents
        documents = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            metadata = metadata_list[i] if metadata_list else {}
            documents.append(
                Document(
                    content=text,
                    embedding=emb,
                    metadata=metadata,
                    doc_id=f"doc_{i}",
                )
            )

        # Create retriever and add documents
        retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
        retriever.add_documents(documents)

        return VectorIndex(
            retriever=retriever,
            metadata={"num_documents": len(documents)},
        )

    def build_from_directory(
        self,
        dir_path: str,
        extensions: List[str] = [".txt", ".md"],
        recursive: bool = True,
        max_documents: Optional[int] = None,
    ) -> VectorIndex:
        """
        Build index from text files in directory.

        Args:
            dir_path: Directory path
            extensions: File extensions to include
            recursive: Whether to search recursively
            max_documents: Maximum documents to index

        Returns:
            Built VectorIndex
        """
        logger.info(f"Building index from directory {dir_path}")

        dir_path = Path(dir_path)
        pattern = "**/*" if recursive else "*"

        documents = []
        texts = []

        for ext in extensions:
            for file_path in dir_path.glob(f"{pattern}{ext}"):
                if max_documents and len(documents) >= max_documents:
                    break

                try:
                    text = file_path.read_text(encoding="utf-8")
                    if not text.strip():
                        continue

                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                            },
                            doc_id=f"doc_{len(documents)}",
                        )
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")

        if not documents:
            logger.warning("No documents found in directory")
            retriever = FAISSRetriever(embedding_dim=self.embedding_dim or 768)
            return VectorIndex(retriever=retriever)

        # Encode texts
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self.encode(texts)

        # Assign embeddings to documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        # Create retriever and add documents
        retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
        retriever.add_documents(documents)

        return VectorIndex(
            retriever=retriever,
            metadata={
                "source": str(dir_path),
                "num_documents": len(documents),
            },
        )
