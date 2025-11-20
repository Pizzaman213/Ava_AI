#!/usr/bin/env python3
"""
Validation tests for Turn-Aware Conversation Data Loader

Tests:
1. Conversation parsing from various formats
2. Turn tokenization with speaker preservation
3. Batch collation without splitting conversations
4. Metadata preservation through pipeline
5. Integration with existing tokenizers
6. Performance benchmarking
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.Ava.data.conversation_turn_loader import (
    ConversationParser,
    Conversation,
    ConversationTurn,
    TurnAwareTokenizer,
    TurnAwareConversationDataset,
    ConversationBatchCollator,
    CONVERSATION_TOKENS,
)


class SimpleTokenizer:
    """Minimal tokenizer for testing."""
    def __init__(self):
        self.vocab = {}
        self.id2token = {}
        self.token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = text.split()
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.token_id
                self.id2token[self.token_id] = token
                self.token_id += 1
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return " ".join(self.id2token.get(id, f"<unk:{id}>") for id in ids)


def test_conversation_parsing():
    """Test parsing different conversation formats."""
    print("\n" + "=" * 80)
    print("TEST 1: Conversation Parsing")
    print("=" * 80)

    # Test 1.1: Text format parsing
    print("\n1.1 Testing text format (User: / Assistant: format)...")
    text_conv = "User: Hello! How are you?\nAssistant: I'm doing well, thank you! How can I help?"
    conv = ConversationParser.parse_text_format(text_conv)

    assert conv.num_turns == 2, f"Expected 2 turns, got {conv.num_turns}"
    assert conv.turns[0].speaker == "user", f"Expected first speaker 'user', got {conv.turns[0].speaker}"
    assert conv.turns[1].speaker == "assistant", f"Expected second speaker 'assistant', got {conv.turns[1].speaker}"
    assert "Hello" in conv.turns[0].content, "First turn content missing"
    print(f"✓ Parsed text format correctly: {conv.num_turns} turns")
    print(f"  Turn 0 ({conv.turns[0].speaker}): {conv.turns[0].content[:50]}...")
    print(f"  Turn 1 ({conv.turns[1].speaker}): {conv.turns[1].content[:50]}...")

    # Test 1.2: Messages format parsing
    print("\n1.2 Testing HuggingFace messages format...")
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]
    conv = ConversationParser.parse_messages_format(messages)
    assert conv.num_turns == 2, f"Expected 2 turns, got {conv.num_turns}"
    assert conv.turns[0].speaker == "user", "First turn should be user"
    assert conv.turns[1].speaker == "assistant", "Second turn should be assistant"
    print(f"✓ Parsed messages format correctly: {conv.num_turns} turns")

    # Test 1.3: JSONL format parsing
    print("\n1.3 Testing JSONL format...")
    jsonl_line = json.dumps({
        "text": "User: Hi there!\nAssistant: Hello! Nice to meet you.",
        "source": "test_dataset",
        "type": "conversation"
    })
    conv = ConversationParser.parse_jsonl_data(jsonl_line)
    assert conv.num_turns == 2, f"Expected 2 turns, got {conv.num_turns}"
    assert conv.source == "test_dataset", f"Expected source 'test_dataset', got {conv.source}"
    assert conv.domain == "conversation", f"Expected domain 'conversation', got {conv.domain}"
    print(f"✓ Parsed JSONL format correctly")
    print(f"  Turns: {conv.num_turns}, Source: {conv.source}, Domain: {conv.domain}")

    # Test 1.4: Auto-parse (mixed formats)
    print("\n1.4 Testing auto-parse on mixed formats...")
    # Text format
    conv1 = ConversationParser.auto_parse("User: Q1\nAssistant: A1")
    assert conv1.num_turns == 2
    # Messages format
    conv2 = ConversationParser.auto_parse([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}])
    assert conv2.num_turns == 2
    # Dict format
    conv3 = ConversationParser.auto_parse({"messages": [{"role": "user", "content": "Q"}]})
    assert conv3.num_turns == 1
    print(f"✓ Auto-parse handled all formats correctly")

    print("\n✓ All parsing tests passed!")
    return True


def test_turn_aware_tokenization():
    """Test turn-aware tokenization with speaker preservation."""
    print("\n" + "=" * 80)
    print("TEST 2: Turn-Aware Tokenization")
    print("=" * 80)

    tokenizer = SimpleTokenizer()
    turn_tokenizer = TurnAwareTokenizer(
        tokenizer=tokenizer,
        max_length=2048,
        include_speaker_tokens=True,
        preserve_turn_boundaries=True,
    )

    # Create test conversation
    conv = Conversation(conversation_id="test_001")
    conv.add_turn("user", "What is AI?")
    conv.add_turn("assistant", "AI is artificial intelligence.")

    print("\n2.1 Testing single turn tokenization...")
    turn = conv.turns[0]
    input_ids, attention_mask = turn_tokenizer.tokenize_turn(turn)
    assert len(input_ids) == len(attention_mask), "Mismatch between input_ids and attention_mask length"
    print(f"✓ Turn tokenized: {len(input_ids)} tokens")
    print(f"  Content: {tokenizer.decode(input_ids)}")

    print("\n2.2 Testing full conversation tokenization...")
    input_ids, attention_mask = turn_tokenizer.tokenize_conversation(conv)
    assert len(input_ids) == len(attention_mask), "Mismatch in full conversation tokenization"
    assert len(input_ids) > 0, "Empty tokenization"
    print(f"✓ Full conversation tokenized: {len(input_ids)} tokens")
    print(f"  First 50 tokens: {input_ids[:50]}")

    print("\n2.3 Testing conversation with turn preservation...")
    result = turn_tokenizer.tokenize_conversation_with_turns(conv)
    assert "turn_tokens" in result, "Missing turn_tokens in result"
    assert result["num_turns"] == 2, f"Expected 2 turns, got {result['num_turns']}"
    assert len(result["turn_tokens"]) == 2, "Turn tokens mismatch"
    for i, turn_data in enumerate(result["turn_tokens"]):
        assert "input_ids" in turn_data, f"Missing input_ids for turn {i}"
        assert "speaker" in turn_data, f"Missing speaker for turn {i}"
        print(f"✓ Turn {i}: {turn_data['speaker']} - {len(turn_data['input_ids'])} tokens")

    print("\n✓ All tokenization tests passed!")
    return True


def test_batch_collation():
    """Test batch collation preserves conversation structure."""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Collation")
    print("=" * 80)

    tokenizer = SimpleTokenizer()
    collator = ConversationBatchCollator(
        tokenizer=tokenizer,
        max_length=2048,
        pad_token_id=0,
        strategy="pad",
    )

    print("\n3.1 Creating test batch...")
    # Create 3 conversations with different lengths
    batch = []
    for i in range(3):
        conv = Conversation(conversation_id=f"test_{i:03d}")
        for j in range(i + 1):  # Variable number of turns
            conv.add_turn(
                "user" if j % 2 == 0 else "assistant",
                f"Turn {j} in conversation {i}"
            )
        batch.append({
            "input_ids": list(range(10 * (i + 1))),  # Variable length
            "attention_mask": [1] * (10 * (i + 1)),
            "conversation_id": conv.conversation_id,
            "source": "test",
            "domain": "general",
            "quality_score": 1.0,
            "num_turns": conv.num_turns,
        })

    print(f"✓ Created batch with 3 conversations:")
    for i, item in enumerate(batch):
        print(f"  Conv {i}: {len(item['input_ids'])} tokens, {item['num_turns']} turns")

    print("\n3.2 Testing collation...")
    collated = collator(batch)

    assert "input_ids" in collated, "Missing input_ids in collated batch"
    assert "attention_mask" in collated, "Missing attention_mask in collated batch"
    assert "conversation_metadata" in collated, "Missing conversation_metadata"

    print(f"✓ Collated batch shape: {collated['input_ids'].shape}")
    print(f"  Batch size: {collated['input_ids'].size(0)}")
    print(f"  Sequence length: {collated['input_ids'].size(1)}")

    # Check padding
    print("\n3.3 Verifying padding...")
    for i in range(len(batch)):
        mask = collated["attention_mask"][i]
        num_real_tokens = mask.sum().item()
        num_padded = (mask == 0).sum().item()
        print(f"  Conv {i}: {num_real_tokens} real tokens, {num_padded} padding tokens")

    # Check metadata preservation
    print("\n3.4 Checking metadata preservation...")
    metadata = collated["conversation_metadata"]
    assert len(metadata["conversation_ids"]) == 3, "Metadata mismatch"
    print(f"✓ Preserved metadata for {len(metadata['conversation_ids'])} conversations")
    for i, conv_id in enumerate(metadata["conversation_ids"]):
        print(f"  Conv {i}: {conv_id} (source={metadata['sources'][i]}, turns={metadata['num_turns'][i]})")

    print("\n✓ All collation tests passed!")
    return True


def test_dataset_loading():
    """Test loading conversations from actual JSONL file."""
    print("\n" + "=" * 80)
    print("TEST 4: Dataset Loading")
    print("=" * 80)

    # Create temporary test JSONL file
    test_file = Path("/tmp/test_conversations.jsonl")
    test_conversations = [
        {
            "text": "User: Hi!\nAssistant: Hello there!",
            "source": "test",
            "type": "conversation"
        },
        {
            "text": "User: What's your name?\nAssistant: I'm Claude.\nUser: Nice to meet you!\nAssistant: Nice to meet you too!",
            "source": "test",
            "type": "conversation"
        },
        {
            "text": "User: Tell me a joke\nAssistant: Why did the chicken cross the road?",
            "source": "test",
            "type": "conversation"
        }
    ]

    print("\n4.1 Creating test JSONL file...")
    with open(test_file, "w") as f:
        for conv_data in test_conversations:
            f.write(json.dumps(conv_data) + "\n")
    print(f"✓ Created test file: {test_file}")

    print("\n4.2 Loading dataset...")
    tokenizer = SimpleTokenizer()
    dataset = TurnAwareConversationDataset(
        data_path=test_file,
        tokenizer=tokenizer,
        max_length=2048,
        max_samples=None,
        include_turn_tokens=False,
        min_turns=1,
    )

    assert len(dataset) == 3, f"Expected 3 conversations, got {len(dataset)}"
    print(f"✓ Loaded {len(dataset)} conversations from JSONL")

    print("\n4.3 Testing dataset indexing...")
    for i in range(len(dataset)):
        item = dataset[i]
        assert "input_ids" in item, f"Missing input_ids in item {i}"
        assert "attention_mask" in item, f"Missing attention_mask in item {i}"
        assert "conversation_id" in item, f"Missing conversation_id in item {i}"
        print(f"  Item {i}: {len(item['input_ids'])} tokens, conv_id={item['conversation_id']}")

    print("\n4.4 Testing with quality filtering...")
    # Create new test file with clearer turn counts
    test_file_filtered = Path("/tmp/test_conversations_turns.jsonl")
    test_conversations_turns = [
        {"text": "User: Hi!", "source": "test", "type": "conversation"},  # 1 turn
        {"text": "User: What's your name?\nAssistant: I'm Claude.\nUser: Nice to meet you!\nAssistant: Nice to meet you too!", "source": "test", "type": "conversation"},  # 4 turns
    ]
    with open(test_file_filtered, "w") as f:
        for conv_data in test_conversations_turns:
            f.write(json.dumps(conv_data) + "\n")

    dataset_filtered = TurnAwareConversationDataset(
        data_path=test_file_filtered,
        tokenizer=tokenizer,
        max_length=2048,
        min_turns=2,  # Only conversations with 2+ turns
    )
    assert len(dataset_filtered) == 1, f"Expected 1 conversation with 2+ turns, got {len(dataset_filtered)}"
    print(f"✓ Filtered to {len(dataset_filtered)} conversations with min_turns=2")
    test_file_filtered.unlink()

    # Cleanup
    test_file.unlink()
    print("\n✓ All dataset loading tests passed!")
    return True


def test_performance():
    """Benchmark performance."""
    print("\n" + "=" * 80)
    print("TEST 5: Performance Benchmarking")
    print("=" * 80)

    tokenizer = SimpleTokenizer()

    # Create large test conversation
    print("\n5.1 Creating large test conversation...")
    conv = Conversation(conversation_id="large_test")
    for i in range(100):
        speaker = "user" if i % 2 == 0 else "assistant"
        conv.add_turn(speaker, f"Turn {i}: " + " ".join([f"word{j}" for j in range(20)]))

    print(f"✓ Created conversation with {conv.num_turns} turns")

    turn_tokenizer = TurnAwareTokenizer(
        tokenizer=tokenizer,
        max_length=2048,
        include_speaker_tokens=True,
    )

    # Benchmark tokenization
    print("\n5.2 Benchmarking turn tokenization...")
    start = time.time()
    for _ in range(10):
        _ = turn_tokenizer.tokenize_conversation(conv)
    elapsed = time.time() - start
    print(f"✓ 10 tokenizations completed in {elapsed:.3f}s ({elapsed/10:.3f}s per conversation)")

    # Benchmark with turn preservation
    print("\n5.3 Benchmarking with turn preservation...")
    start = time.time()
    for _ in range(10):
        _ = turn_tokenizer.tokenize_conversation_with_turns(conv)
    elapsed = time.time() - start
    print(f"✓ 10 full tokenizations completed in {elapsed:.3f}s ({elapsed/10:.3f}s per conversation)")

    print("\n✓ Performance benchmarks complete!")
    return True


def test_special_tokens():
    """Test special token handling."""
    print("\n" + "=" * 80)
    print("TEST 6: Special Token Handling")
    print("=" * 80)

    print("\n6.1 Verifying special tokens...")
    expected_tokens = [
        "turn_start", "turn_end",
        "user_start", "user_end",
        "assistant_start", "assistant_end",
        "context_start", "context_end"
    ]

    for token_key in expected_tokens:
        assert token_key in CONVERSATION_TOKENS, f"Missing token: {token_key}"
        token_value = CONVERSATION_TOKENS[token_key]
        assert token_value.startswith("<") and token_value.endswith(">"), \
            f"Token {token_key} has invalid format: {token_value}"
        print(f"  ✓ {token_key}: {token_value}")

    print("\n✓ All special tokens valid!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TURN-AWARE CONVERSATION DATA LOADER - VALIDATION SUITE")
    print("=" * 80)

    tests = [
        ("Conversation Parsing", test_conversation_parsing),
        ("Turn-Aware Tokenization", test_turn_aware_tokenization),
        ("Batch Collation", test_batch_collation),
        ("Dataset Loading", test_dataset_loading),
        ("Special Tokens", test_special_tokens),
        ("Performance", test_performance),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
