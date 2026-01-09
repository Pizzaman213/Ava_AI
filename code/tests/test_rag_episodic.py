"""
Tests for RAG and Episodic Memory features.

Run with: python -m pytest code/tests/test_rag_episodic.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


class TestEpisodicMemoryBuffer(unittest.TestCase):
    """Tests for EpisodicMemoryBuffer."""

    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        from ava.training.episodic_memory import EpisodicMemoryBuffer

        buffer = EpisodicMemoryBuffer(capacity=100)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.capacity, 100)

    def test_add_entry(self):
        """Test adding entries to buffer."""
        from ava.training.episodic_memory import EpisodicMemoryBuffer, MemoryEntry

        buffer = EpisodicMemoryBuffer(capacity=100)

        entry = MemoryEntry(
            input_ids=torch.randint(0, 1000, (128,)),
            attention_mask=torch.ones(128),
            labels=torch.randint(0, 1000, (128,)),
            loss=0.5,
        )
        buffer.add(entry)

        self.assertEqual(len(buffer), 1)

    def test_sample_from_buffer(self):
        """Test sampling from buffer."""
        from ava.training.episodic_memory import EpisodicMemoryBuffer, MemoryEntry

        buffer = EpisodicMemoryBuffer(capacity=100, selection_strategy="importance")

        # Add multiple entries
        for i in range(10):
            entry = MemoryEntry(
                input_ids=torch.randint(0, 1000, (128,)),
                attention_mask=torch.ones(128),
                labels=torch.randint(0, 1000, (128,)),
                loss=float(i) / 10,
            )
            buffer.add(entry)

        # Sample
        entries, indices, weights = buffer.sample(5)

        self.assertEqual(len(entries), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(weights), 5)
        self.assertTrue(all(w <= 1.0 for w in weights))

    def test_priority_update(self):
        """Test priority update after sampling."""
        from ava.training.episodic_memory import EpisodicMemoryBuffer, MemoryEntry

        buffer = EpisodicMemoryBuffer(capacity=100, selection_strategy="importance")

        # Add entries
        for i in range(10):
            entry = MemoryEntry(
                input_ids=torch.randint(0, 1000, (128,)),
                attention_mask=torch.ones(128),
                labels=torch.randint(0, 1000, (128,)),
                loss=0.1,
            )
            buffer.add(entry)

        # Sample and update priorities
        entries, indices, _ = buffer.sample(3)
        new_losses = np.array([0.9, 0.8, 0.7])
        buffer.update_priorities(indices, new_losses)

        # Check priorities were updated
        for idx, loss in zip(indices, new_losses):
            self.assertAlmostEqual(buffer.priorities[idx], abs(loss) + 1e-6, places=5)

    def test_buffer_capacity_limit(self):
        """Test buffer respects capacity limit."""
        from ava.training.episodic_memory import EpisodicMemoryBuffer, MemoryEntry

        buffer = EpisodicMemoryBuffer(capacity=10, selection_strategy="uniform")

        # Add more than capacity
        for i in range(20):
            entry = MemoryEntry(
                input_ids=torch.randint(0, 1000, (128,)),
                attention_mask=torch.ones(128),
                labels=torch.randint(0, 1000, (128,)),
                loss=0.5,
            )
            buffer.add(entry)

        self.assertEqual(len(buffer), 10)


class TestEpisodicMemoryManager(unittest.TestCase):
    """Tests for EpisodicMemoryManager."""

    def test_manager_disabled(self):
        """Test manager when disabled."""
        from ava.training.episodic_memory import EpisodicMemoryManager

        config = MagicMock()
        config.use_episodic_memory = False

        manager = EpisodicMemoryManager(config)
        self.assertFalse(manager.enabled)

    def test_manager_enabled(self):
        """Test manager when enabled."""
        from ava.training.episodic_memory import EpisodicMemoryManager

        config = MagicMock()
        config.use_episodic_memory = True
        config.memory_capacity = 1000
        config.memory_replay_ratio = 0.2
        config.buffer_warmup_steps = 10
        config.memory_selection_strategy = "importance"
        config.priority_exponent = 0.6
        config.importance_weight_exponent = 0.4
        config.store_aux_info = False
        config.silent_mode = True

        manager = EpisodicMemoryManager(config)
        self.assertTrue(manager.enabled)
        self.assertIsNotNone(manager.buffer)

    def test_augment_batch_during_warmup(self):
        """Test batch augmentation during warmup (no replay)."""
        from ava.training.episodic_memory import EpisodicMemoryManager

        config = MagicMock()
        config.use_episodic_memory = True
        config.memory_capacity = 1000
        config.memory_replay_ratio = 0.2
        config.buffer_warmup_steps = 100
        config.memory_selection_strategy = "importance"
        config.priority_exponent = 0.6
        config.importance_weight_exponent = 0.4
        config.store_aux_info = False
        config.silent_mode = True

        manager = EpisodicMemoryManager(config, device=torch.device("cpu"))

        batch = {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "attention_mask": torch.ones(4, 128),
            "labels": torch.randint(0, 1000, (4, 128)),
        }

        # During warmup (step < warmup_steps), no augmentation
        augmented, indices = manager.augment_batch(batch, 0, global_step=5)

        self.assertEqual(augmented["input_ids"].size(0), 4)  # No replay added
        self.assertEqual(len(indices), 0)

    def test_get_metrics(self):
        """Test metrics retrieval."""
        from ava.training.episodic_memory import EpisodicMemoryManager

        config = MagicMock()
        config.use_episodic_memory = True
        config.memory_capacity = 1000
        config.memory_replay_ratio = 0.2
        config.buffer_warmup_steps = 10
        config.memory_selection_strategy = "importance"
        config.priority_exponent = 0.6
        config.importance_weight_exponent = 0.4
        config.store_aux_info = False
        config.silent_mode = True

        manager = EpisodicMemoryManager(config, device=torch.device("cpu"))

        metrics = manager.get_metrics()
        self.assertIn("episodic/buffer_size", metrics)
        self.assertIn("episodic/fill_ratio", metrics)


class TestRAGDocument(unittest.TestCase):
    """Tests for RAG Document class."""

    def test_document_creation(self):
        """Test document creation."""
        from ava.rag import Document

        doc = Document(
            content="Hello world",
            embedding=torch.randn(768),
            metadata={"source": "test"},
        )

        self.assertEqual(doc.content, "Hello world")
        self.assertEqual(doc.embedding.shape, (768,))
        self.assertEqual(doc.metadata["source"], "test")

    def test_document_to_dict(self):
        """Test document serialization."""
        from ava.rag import Document

        doc = Document(
            content="Hello world",
            metadata={"source": "test"},
            doc_id="doc_1",
            score=0.9,
        )

        data = doc.to_dict()
        self.assertEqual(data["content"], "Hello world")
        self.assertEqual(data["doc_id"], "doc_1")
        self.assertEqual(data["score"], 0.9)


class TestInMemoryRetriever(unittest.TestCase):
    """Tests for InMemoryRetriever."""

    def test_retriever_add_documents(self):
        """Test adding documents to retriever."""
        from ava.rag import InMemoryRetriever, Document

        retriever = InMemoryRetriever(embedding_dim=64)

        docs = [
            Document("Hello world", embedding=torch.randn(64)),
            Document("Goodbye world", embedding=torch.randn(64)),
        ]
        retriever.add_documents(docs)

        self.assertEqual(len(retriever), 2)

    def test_retriever_retrieve(self):
        """Test document retrieval."""
        from ava.rag import InMemoryRetriever, Document

        retriever = InMemoryRetriever(embedding_dim=64)

        # Add documents with distinct embeddings
        doc1_emb = torch.zeros(64)
        doc1_emb[0] = 1.0
        doc2_emb = torch.zeros(64)
        doc2_emb[1] = 1.0

        docs = [
            Document("First", embedding=doc1_emb),
            Document("Second", embedding=doc2_emb),
        ]
        retriever.add_documents(docs)

        # Query similar to first document
        query = torch.zeros(1, 64)
        query[0, 0] = 1.0

        results = retriever.retrieve(query, top_k=2)

        self.assertEqual(len(results), 1)  # One batch
        self.assertEqual(len(results[0]), 2)  # Two documents
        # First result should be most similar to query
        self.assertEqual(results[0][0].document.content, "First")


class TestRAGFusion(unittest.TestCase):
    """Tests for RAG fusion strategies."""

    def test_concat_fusion(self):
        """Test ConcatFusion."""
        from ava.rag import ConcatFusion

        fusion = ConcatFusion(hidden_size=64, context_size=64)

        hidden = torch.randn(2, 10, 64)
        context = torch.randn(2, 5, 64)

        output = fusion(hidden, context)

        self.assertEqual(output.shape, hidden.shape)

    def test_gated_fusion(self):
        """Test GatedFusion."""
        from ava.rag import GatedFusion

        fusion = GatedFusion(hidden_size=64, context_size=64)

        hidden = torch.randn(2, 10, 64)
        context = torch.randn(2, 5, 64)

        output = fusion(hidden, context)

        self.assertEqual(output.shape, hidden.shape)

    def test_cross_attention_fusion(self):
        """Test CrossAttentionFusion."""
        from ava.rag import CrossAttentionFusion

        fusion = CrossAttentionFusion(hidden_size=64, context_size=64, num_heads=4)

        hidden = torch.randn(2, 10, 64)
        context = torch.randn(2, 5, 64)

        output = fusion(hidden, context)

        self.assertEqual(output.shape, hidden.shape)

    def test_create_fusion_factory(self):
        """Test fusion factory function."""
        from ava.rag import create_fusion, ConcatFusion, GatedFusion, CrossAttentionFusion

        config = MagicMock()
        config.rag_fusion_type = "concat"
        config.embedding_dim = 64
        config.fusion_dropout = 0.1
        config.fusion_num_heads = 8

        fusion = create_fusion(config, hidden_size=64)
        self.assertIsInstance(fusion, ConcatFusion)

        config.rag_fusion_type = "gated"
        fusion = create_fusion(config, hidden_size=64)
        self.assertIsInstance(fusion, GatedFusion)

        config.rag_fusion_type = "attention"
        fusion = create_fusion(config, hidden_size=64)
        self.assertIsInstance(fusion, CrossAttentionFusion)


class TestIndexBuilder(unittest.TestCase):
    """Tests for IndexBuilder."""

    def test_build_from_texts(self):
        """Test building index from text list."""
        from ava.rag import IndexBuilder

        # Mock sentence transformer
        with patch("ava.rag.index.SentenceTransformer") as mock_st:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = torch.randn(3, 64)
            mock_encoder.get_sentence_embedding_dimension.return_value = 64
            mock_st.return_value = mock_encoder

            builder = IndexBuilder(encoder_model="test-model")
            builder.embedding_dim = 64

            texts = ["Hello", "World", "Test"]
            index = builder.build_from_texts(texts)

            self.assertEqual(len(index), 3)


if __name__ == "__main__":
    unittest.main()
