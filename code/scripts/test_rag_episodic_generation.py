#!/usr/bin/env python3
"""
Test script for RAG and Episodic Memory features with generation.

This script tests:
1. RAG retrieval and fusion during generation
2. Episodic memory buffer and training integration
3. End-to-end generation with a small test model

Run: python code/scripts/test_rag_episodic_generation.py
"""

import sys
import torch
import logging

# Add project root to path
sys.path.insert(0, '/root/Ava_AI/code')

from ava.models.moe import EnhancedMoEModel, EnhancedMoEConfig
from ava.config.training_config import RAGConfig, EpisodicMemoryConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_episodic_memory():
    """Test episodic memory buffer and manager."""
    logger.info("=" * 60)
    logger.info("Testing Episodic Memory")
    logger.info("=" * 60)

    from ava.training.episodic_memory import (
        MemoryEntry,
        EpisodicMemoryBuffer,
        EpisodicMemoryManager,
    )

    # Test buffer
    logger.info("1. Testing EpisodicMemoryBuffer...")
    buffer = EpisodicMemoryBuffer(capacity=100, selection_strategy="importance")

    # Add entries
    for i in range(20):
        entry = MemoryEntry(
            input_ids=torch.randint(0, 1000, (64,)),
            attention_mask=torch.ones(64),
            labels=torch.randint(0, 1000, (64,)),
            loss=float(i) / 10 + 0.1,
        )
        buffer.add(entry)

    logger.info(f"   Buffer size: {len(buffer)}")
    logger.info(f"   Buffer stats: {buffer.get_stats()}")

    # Sample
    entries, indices, weights = buffer.sample(5)
    logger.info(f"   Sampled {len(entries)} entries")
    logger.info(f"   Weights: {weights}")

    # Update priorities
    import numpy as np
    buffer.update_priorities(indices, np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    logger.info("   Priorities updated successfully")

    # Test manager
    logger.info("\n2. Testing EpisodicMemoryManager...")

    class MockConfig:
        use_episodic_memory = True
        memory_capacity = 1000
        memory_replay_ratio = 0.2
        buffer_warmup_steps = 5
        memory_selection_strategy = "importance"
        priority_exponent = 0.6
        importance_weight_exponent = 0.4
        store_aux_info = False
        silent_mode = True

    manager = EpisodicMemoryManager(MockConfig(), device=torch.device("cpu"))
    logger.info(f"   Manager enabled: {manager.enabled}")

    # Test augment_batch
    batch = {
        "input_ids": torch.randint(0, 1000, (4, 64)),
        "attention_mask": torch.ones(4, 64),
        "labels": torch.randint(0, 1000, (4, 64)),
    }

    # During warmup (no replay)
    augmented, indices = manager.augment_batch(batch, 0, global_step=2)
    logger.info(f"   Warmup augmentation: batch size {augmented['input_ids'].size(0)} (expected 4)")

    # Store some samples to fill buffer
    for step in range(10):
        manager.store_batch(batch, torch.tensor(0.5))
        manager._global_step = step

    # After warmup (with replay)
    augmented, indices = manager.augment_batch(batch, 0, global_step=100)
    logger.info(f"   Post-warmup augmentation: batch size {augmented['input_ids'].size(0)} (expected > 4)")

    metrics = manager.get_metrics()
    logger.info(f"   Metrics: {metrics}")

    logger.info("Episodic Memory tests PASSED")
    return True


def test_rag_components():
    """Test RAG retriever and fusion components."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing RAG Components")
    logger.info("=" * 60)

    from ava.rag import (
        Document,
        InMemoryRetriever,
        ConcatFusion,
        GatedFusion,
        CrossAttentionFusion,
    )

    # Test Document
    logger.info("1. Testing Document...")
    doc = Document(
        content="The quick brown fox jumps over the lazy dog.",
        embedding=torch.randn(64),
        metadata={"source": "test"},
    )
    logger.info(f"   Document content: {doc.content[:30]}...")
    logger.info(f"   Embedding shape: {doc.embedding.shape}")

    # Test InMemoryRetriever
    logger.info("\n2. Testing InMemoryRetriever...")
    retriever = InMemoryRetriever(embedding_dim=64)

    # Add documents with distinct embeddings
    docs = []
    for i in range(10):
        emb = torch.zeros(64)
        emb[i % 64] = 1.0  # Distinct embeddings
        docs.append(Document(f"Document {i}", embedding=emb))

    retriever.add_documents(docs)
    logger.info(f"   Added {len(retriever)} documents")

    # Query
    query = torch.zeros(1, 64)
    query[0, 0] = 1.0  # Similar to document 0

    results = retriever.retrieve(query, top_k=3)
    logger.info(f"   Retrieved {len(results[0])} documents")
    logger.info(f"   Top result: '{results[0][0].document.content}' (score: {results[0][0].score:.4f})")

    # Test Fusion strategies
    logger.info("\n3. Testing Fusion Strategies...")

    hidden = torch.randn(2, 10, 64)
    context = torch.randn(2, 5, 64)

    # ConcatFusion
    concat_fusion = ConcatFusion(hidden_size=64, context_size=64)
    output = concat_fusion(hidden, context)
    logger.info(f"   ConcatFusion output shape: {output.shape}")
    assert output.shape == hidden.shape, "ConcatFusion output shape mismatch"

    # GatedFusion
    gated_fusion = GatedFusion(hidden_size=64, context_size=64)
    output = gated_fusion(hidden, context)
    logger.info(f"   GatedFusion output shape: {output.shape}")
    assert output.shape == hidden.shape, "GatedFusion output shape mismatch"

    # CrossAttentionFusion
    cross_fusion = CrossAttentionFusion(hidden_size=64, context_size=64, num_heads=4)
    output = cross_fusion(hidden, context)
    logger.info(f"   CrossAttentionFusion output shape: {output.shape}")
    assert output.shape == hidden.shape, "CrossAttentionFusion output shape mismatch"

    logger.info("RAG Component tests PASSED")
    return True


def test_model_with_rag():
    """Test EnhancedMoEModel with RAG enabled."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Model with RAG Integration")
    logger.info("=" * 60)

    # Create small model config
    logger.info("1. Creating small test model...")

    # Create RAG config
    rag_config = RAGConfig(
        use_rag=True,
        retriever_type="memory",  # Use in-memory retriever for testing
        embedding_dim=64,
        max_retrieved_docs=3,
        rag_fusion_type="gated",
    )

    # Create model config
    config = EnhancedMoEConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_token=2,
        dropout=0.0,
        attention_dropout=0.0,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )

    # Attach RAG config to model config
    config.rag_config = rag_config

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"   Using device: {device}")

    model = EnhancedMoEModel(config)
    model = model.to(device)

    logger.info(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"   RAG retriever: {model.retriever}")
    logger.info(f"   RAG fusion: {model.rag_fusion}")

    # Add documents to retriever if available
    if model.retriever is not None:
        logger.info("\n2. Adding documents to RAG retriever...")
        from ava.rag import Document

        docs = []
        for i in range(10):
            emb = torch.randn(config.hidden_size)
            docs.append(Document(f"Test document {i} with some content.", embedding=emb))

        model.retriever.add_documents(docs)
        logger.info(f"   Added {len(model.retriever)} documents")

    # Test forward pass
    logger.info("\n3. Testing forward pass with RAG...")

    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    attention_mask = torch.ones(2, 32, device=device)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    logger.info(f"   Loss: {outputs['loss'].item():.4f}")
    logger.info(f"   Logits shape: {outputs['logits'].shape}")

    # Test generation
    logger.info("\n4. Testing generation with RAG...")

    model.eval()
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)

    # Simple greedy generation
    generated = prompt_ids.clone()
    for _ in range(20):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            next_token_logits = outputs['logits'][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    logger.info(f"   Generated sequence length: {generated.shape[1]}")
    logger.info(f"   Generated tokens: {generated[0, :20].tolist()}")

    logger.info("Model with RAG tests PASSED")
    return True


def test_episodic_memory_training_integration():
    """Test episodic memory integration with training loop."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Episodic Memory Training Integration")
    logger.info("=" * 60)

    from ava.training.episodic_memory import EpisodicMemoryManager

    # Create config
    class MockMemConfig:
        use_episodic_memory = True
        memory_capacity = 100
        memory_replay_ratio = 0.25
        buffer_warmup_steps = 5
        memory_selection_strategy = "importance"
        priority_exponent = 0.6
        importance_weight_exponent = 0.4
        store_aux_info = False
        silent_mode = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = EpisodicMemoryManager(MockMemConfig(), device=device)

    # Create small model
    config = EnhancedMoEConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_token=2,
        dropout=0.0,
        use_flash_attention=False,
        gradient_checkpointing=False,
    )

    model = EnhancedMoEModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    logger.info("1. Simulating training steps...")

    # Simulate training loop
    for step in range(20):
        # Create batch
        batch = {
            "input_ids": torch.randint(0, config.vocab_size, (4, 32), device=device),
            "attention_mask": torch.ones(4, 32, device=device),
            "labels": torch.randint(0, config.vocab_size, (4, 32), device=device),
        }

        # Augment batch with episodic memory
        augmented_batch, replay_indices = manager.augment_batch(
            batch, batch_idx=step, global_step=step
        )

        # Forward pass
        outputs = model(
            input_ids=augmented_batch["input_ids"],
            attention_mask=augmented_batch["attention_mask"],
            labels=augmented_batch["labels"],
        )
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store batch in memory
        manager.store_batch(batch, loss.detach())

        # Update replay priorities
        if len(replay_indices) > 0:
            manager.update_replay_priorities(replay_indices, loss.detach().expand(len(replay_indices)))

        if step % 5 == 0:
            metrics = manager.get_metrics()
            logger.info(f"   Step {step}: loss={loss.item():.4f}, buffer_size={metrics.get('episodic/buffer_size', 0)}")

    logger.info("\n2. Final metrics...")
    final_metrics = manager.get_metrics()
    for key, value in final_metrics.items():
        logger.info(f"   {key}: {value}")

    logger.info("Episodic Memory Training Integration tests PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RAG and Episodic Memory Feature Tests")
    logger.info("=" * 60)

    results = {}

    try:
        results["episodic_memory"] = test_episodic_memory()
    except Exception as e:
        logger.error(f"Episodic Memory test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["episodic_memory"] = False

    try:
        results["rag_components"] = test_rag_components()
    except Exception as e:
        logger.error(f"RAG Components test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["rag_components"] = False

    try:
        results["model_with_rag"] = test_model_with_rag()
    except Exception as e:
        logger.error(f"Model with RAG test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["model_with_rag"] = False

    try:
        results["episodic_training"] = test_episodic_memory_training_integration()
    except Exception as e:
        logger.error(f"Episodic Memory Training Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["episodic_training"] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"   {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\nAll tests PASSED!")
        return 0
    else:
        logger.info("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())
