"""
Turn-Aware Conversation Data Loader

Preserves conversation structure, enables coherence-aware batching, and maintains
speaker context across turns. Designed for training conversational models with
better dialogue coherence and logical flow.

Key Features:
- Parses conversations into individual turns with metadata
- Preserves speaker labels and turn boundaries with special tokens
- Groups complete conversations for batching (not random splits)
- Enables coherence scoring and quality filtering per conversation
- Supports conversation-level metadata (source, quality score, domain)
- Context window management for multi-turn dialogue
- Efficient memory-mapped access to pre-tokenized data
- Backward compatible with existing tokenized datasets

Turn Structure:
{
    "conversation_id": "unique_hash",
    "turns": [
        {
            "speaker": "user" or "assistant",
            "content": "text of this turn",
            "turn_idx": 0,
            "turn_input_ids": [1, 2, 3, ...],
            "turn_attention_mask": [1, 1, 1, ...]
        },
        ...
    ],
    "num_turns": 2,
    "quality_score": 0.85,
    "source": "dataset_name",
    "domain": "general|technical|creative|etc"
}
"""

import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any
from dataclasses import dataclass, field
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, Dataset, DataLoader
from collections import defaultdict
import re
from functools import lru_cache


# Special tokens for turn boundaries
CONVERSATION_TOKENS = {
    "turn_start": "<turn_start>",
    "turn_end": "<turn_end>",
    "user_start": "<user>",
    "user_end": "</user>",
    "assistant_start": "<assistant>",
    "assistant_end": "</assistant>",
    "context_start": "<context>",
    "context_end": "</context>",
}


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    speaker: str  # "user", "assistant", "system"
    content: str
    turn_idx: int
    turn_input_ids: Optional[List[int]] = None
    turn_attention_mask: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "content": self.content,
            "turn_idx": self.turn_idx,
            "turn_input_ids": self.turn_input_ids,
            "turn_attention_mask": self.turn_attention_mask,
        }


@dataclass
class Conversation:
    """Represents a complete conversation with metadata."""
    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    quality_score: float = 1.0
    source: str = "unknown"
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def add_turn(self, speaker: str, content: str) -> None:
        """Add a turn to this conversation."""
        turn = ConversationTurn(
            speaker=speaker,
            content=content,
            turn_idx=len(self.turns)
        )
        self.turns.append(turn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "num_turns": self.num_turns,
            "quality_score": self.quality_score,
            "source": self.source,
            "domain": self.domain,
            "metadata": self.metadata,
        }


class ConversationParser:
    """
    Parses various conversation formats into standardized turn structure.

    Supports:
    - "User: text\nAssistant: text" format (current JSONL)
    - HuggingFace messages format [{"role": "user", "content": "..."}, ...]
    - Structured dialogue with explicit field names
    - Multi-speaker conversations
    """

    # Regex patterns for different speaker formats
    SPEAKER_PATTERNS = [
        (r"^User:\s*", "user"),
        (r"^Assistant:\s*", "assistant"),
        (r"^Human:\s*", "user"),
        (r"^AI:\s*", "assistant"),
        (r"^A:\s*", "assistant"),
        (r"^Q:\s*", "user"),
        (r"^Question:\s*", "user"),
        (r"^Answer:\s*", "assistant"),
        (r"^Customer:\s*", "user"),
        (r"^Agent:\s*", "assistant"),
        (r"^[*]?Speaker\s+\d+:\s*", "user"),  # Generic speaker
    ]

    @staticmethod
    def parse_text_format(text: str) -> Conversation:
        """
        Parse conversational text with speaker prefixes.
        Format: "User: message\nAssistant: message\n..."
        """
        # Generate deterministic conversation ID
        conv_id = hashlib.md5(text.encode()).hexdigest()[:16]
        conversation = Conversation(conversation_id=conv_id)

        lines = text.split("\n")
        current_speaker = None
        current_content = []

        for line in lines:
            if not line.strip():
                continue

            # Try to match speaker pattern
            matched = False
            for pattern, speaker_type in ConversationParser.SPEAKER_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous turn if exists
                    if current_speaker and current_content:
                        conversation.add_turn(
                            current_speaker,
                            " ".join(current_content).strip()
                        )
                    # Start new turn
                    current_speaker = speaker_type
                    current_content = [line[match.end():]]
                    matched = True
                    break

            if not matched and current_speaker:
                # Continuation of current turn
                current_content.append(line)

        # Save last turn
        if current_speaker and current_content:
            conversation.add_turn(current_speaker, " ".join(current_content).strip())

        return conversation

    @staticmethod
    def parse_messages_format(messages: List[Dict[str, str]]) -> Conversation:
        """
        Parse HuggingFace messages format.
        Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        conv_id = hashlib.md5(json.dumps(messages).encode()).hexdigest()[:16]
        conversation = Conversation(conversation_id=conv_id)

        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = msg["role"]
                # Normalize role names
                if role in ["user", "human", "question"]:
                    speaker = "user"
                elif role in ["assistant", "ai", "answer"]:
                    speaker = "assistant"
                else:
                    speaker = role

                conversation.add_turn(speaker, msg["content"])

        return conversation

    @staticmethod
    def parse_dialogue_format(dialogue: List[str]) -> Conversation:
        """
        Parse dialogue format with alternating turns.
        Format: ["First speaker text", "Second speaker text", ...]
        Assumes alternating user/assistant.
        """
        conv_id = hashlib.md5(json.dumps(dialogue).encode()).hexdigest()[:16]
        conversation = Conversation(conversation_id=conv_id)

        for idx, text in enumerate(dialogue):
            speaker = "user" if idx % 2 == 0 else "assistant"
            conversation.add_turn(speaker, text)

        return conversation

    @staticmethod
    def parse_jsonl_data(jsonl_text: str) -> Conversation:
        """
        Parse JSONL formatted conversation data.
        Attempts to parse as JSON first, then falls back to text format.
        """
        try:
            data = json.loads(jsonl_text)
            text_content = data.get("text", "")
            conv = ConversationParser.parse_text_format(text_content)

            # Preserve metadata
            if "source" in data:
                conv.source = data["source"]
            if "type" in data:
                conv.domain = data["type"]
            if "quality_score" in data:
                conv.quality_score = data["quality_score"]

            return conv
        except json.JSONDecodeError:
            return ConversationParser.parse_text_format(jsonl_text)

    @classmethod
    def auto_parse(cls, data: Any) -> Conversation:
        """
        Automatically detect and parse conversation format.
        Supports: dict with messages, list of dicts, list of strings, plain text.
        """
        if isinstance(data, dict):
            if "messages" in data:
                return cls.parse_messages_format(data["messages"])
            elif "text" in data:
                return cls.parse_text_format(data["text"])
            elif "conversation" in data:
                return cls.auto_parse(data["conversation"])
            else:
                # Try to parse as JSON first
                try:
                    return cls.parse_jsonl_data(json.dumps(data))
                except (json.JSONDecodeError, ValueError, TypeError):
                    return cls.parse_text_format(str(data))
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return cls.parse_messages_format(data)
            else:
                return cls.parse_dialogue_format(data)
        else:
            # Plain text
            return cls.parse_text_format(str(data))


class TurnAwareTokenizer:
    """
    Tokenizes conversations while preserving turn boundaries and structure.

    Adds special tokens for:
    - Turn starts/ends
    - Speaker indicators
    - Context boundaries
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        include_speaker_tokens: bool = True,
        preserve_turn_boundaries: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_speaker_tokens = include_speaker_tokens
        self.preserve_turn_boundaries = preserve_turn_boundaries

    def tokenize_turn(self, turn: ConversationTurn) -> Tuple[List[int], List[int]]:
        """Tokenize a single turn with optional speaker labels."""
        text = turn.content

        if self.include_speaker_tokens:
            # Add speaker-specific tokens
            if turn.speaker == "user":
                text = f"{CONVERSATION_TOKENS['user_start']} {text} {CONVERSATION_TOKENS['user_end']}"
            elif turn.speaker == "assistant":
                text = f"{CONVERSATION_TOKENS['assistant_start']} {text} {CONVERSATION_TOKENS['assistant_end']}"

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        attention_mask = [1] * len(tokens)

        return tokens, attention_mask

    def tokenize_conversation(
        self,
        conversation: Conversation,
        include_context: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenize entire conversation preserving turn structure.

        Returns:
            (input_ids, attention_mask) for the full conversation
        """
        all_tokens = []
        all_masks = []

        for turn_idx, turn in enumerate(conversation.turns):
            # Add turn start marker
            if self.preserve_turn_boundaries:
                turn_start = self.tokenizer.encode(
                    CONVERSATION_TOKENS["turn_start"],
                    add_special_tokens=False
                )
                all_tokens.extend(turn_start)
                all_masks.extend([1] * len(turn_start))

            # Tokenize turn content
            turn_tokens, turn_masks = self.tokenize_turn(turn)
            all_tokens.extend(turn_tokens)
            all_masks.extend(turn_masks)

            # Add turn end marker
            if self.preserve_turn_boundaries:
                turn_end = self.tokenizer.encode(
                    CONVERSATION_TOKENS["turn_end"],
                    add_special_tokens=False
                )
                all_tokens.extend(turn_end)
                all_masks.extend([1] * len(turn_end))

        # Truncate if needed
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length]
            all_masks = all_masks[:self.max_length]

        return all_tokens, all_masks

    def tokenize_conversation_with_turns(
        self,
        conversation: Conversation,
    ) -> Dict[str, Any]:
        """
        Tokenize conversation and preserve per-turn tokens.

        Returns dict with:
            - input_ids: full conversation tokens
            - attention_mask: full conversation masks
            - turn_tokens: list of per-turn tokenized data
            - num_turns: number of turns
        """
        full_input_ids, full_attention_mask = self.tokenize_conversation(conversation)

        turn_tokens = []
        for turn in conversation.turns:
            turn_ids, turn_masks = self.tokenize_turn(turn)
            turn_tokens.append({
                "input_ids": turn_ids,
                "attention_mask": turn_masks,
                "speaker": turn.speaker,
                "turn_idx": turn.turn_idx,
            })

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "turn_tokens": turn_tokens,
            "num_turns": conversation.num_turns,
            "conversation_id": conversation.conversation_id,
            "source": conversation.source,
            "domain": conversation.domain,
            "quality_score": conversation.quality_score,
        }


class TurnAwareConversationDataset(Dataset):
    """
    Map-style dataset for conversations with turn awareness.

    Groups complete conversations together in batches to preserve
    dialogue context and coherence.
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        include_turn_tokens: bool = True,
        min_turns: int = 1,
        quality_threshold: float = 0.0,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.include_turn_tokens = include_turn_tokens
        self.min_turns = min_turns
        self.quality_threshold = quality_threshold

        self.turn_aware_tokenizer = TurnAwareTokenizer(
            tokenizer,
            max_length=max_length,
            include_speaker_tokens=True,
            preserve_turn_boundaries=True,
        )

        self.conversations = self._load_conversations()

    def _load_conversations(self) -> List[Conversation]:
        """Load and parse conversations from JSONL file."""
        conversations = []

        with open(self.data_path, "r") as f:
            for idx, line in enumerate(f):
                if self.max_samples and len(conversations) >= self.max_samples:
                    break

                try:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse conversation from JSONL line
                    conv = ConversationParser.parse_jsonl_data(line)

                    # Filter by quality and turn count
                    if conv.num_turns < self.min_turns:
                        continue
                    if conv.quality_score < self.quality_threshold:
                        continue

                    conversations.append(conv)
                except Exception as e:
                    print(f"Warning: Failed to parse line {idx}: {e}")
                    continue

        print(f" Loaded {len(conversations)} conversations from {self.data_path}")
        return conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a conversation with tokenization and turn preservation."""
        conv = self.conversations[idx]

        if self.include_turn_tokens:
            return self.turn_aware_tokenizer.tokenize_conversation_with_turns(conv)
        else:
            input_ids, attention_mask = self.turn_aware_tokenizer.tokenize_conversation(conv)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "conversation_id": conv.conversation_id,
                "source": conv.source,
                "domain": conv.domain,
                "quality_score": conv.quality_score,
            }


class ConversationBatchCollator:
    """
    Custom collator for conversation batches.

    Ensures conversations are NOT split across batches.
    Applies padding, truncation, and manages variable-length sequences.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        pad_token_id: int = 0,
        strategy: str = "pad",  # "pad", "truncate", "pack"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.strategy = strategy

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of conversations.

        Preserves conversation structure while creating padded batches.
        """
        if not batch:
            return {}

        # Extract sequences
        input_ids_list = []
        attention_masks_list = []
        metadata = {
            "conversation_ids": [],
            "sources": [],
            "domains": [],
            "quality_scores": [],
            "num_turns": [],
        }

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]

            # Apply truncation if needed
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)

            # Collect metadata
            metadata["conversation_ids"].append(item.get("conversation_id", ""))
            metadata["sources"].append(item.get("source", "unknown"))
            metadata["domains"].append(item.get("domain", "general"))
            metadata["quality_scores"].append(item.get("quality_score", 1.0))
            metadata["num_turns"].append(item.get("num_turns", 1))

        # Pad sequences
        max_seq_len = min(max(len(seq) for seq in input_ids_list), self.max_length)

        padded_input_ids = []
        padded_attention_masks = []

        for input_ids, attention_mask in zip(input_ids_list, attention_masks_list):
            # Pad
            pad_len = max_seq_len - len(input_ids)
            padded_input_ids.append(
                input_ids + [self.pad_token_id] * pad_len
            )
            padded_attention_masks.append(
                attention_mask + [0] * pad_len
            )

        # Convert to tensors
        output = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
        }

        # Add metadata
        output["conversation_metadata"] = metadata

        return output


class TurnAwareConversationDataLoader:
    """
    Factory for creating turn-aware conversation dataloaders.

    Manages the full pipeline:
    1. Load conversations from JSONL
    2. Parse and preserve turn structure
    3. Tokenize with speaker labels
    4. Batch complete conversations together
    5. Apply coherence-aware batching strategies
    """

    @staticmethod
    def create_dataloader(
        data_path: str,
        tokenizer,
        batch_size: int = 32,
        max_length: int = 2048,
        num_workers: int = 0,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        min_turns: int = 1,
        quality_threshold: float = 0.0,
    ) -> DataLoader:
        """
        Create a turn-aware conversation dataloader.

        Args:
            data_path: Path to JSONL file with conversations
            tokenizer: Tokenizer to use
            batch_size: Batch size (one conversation per batch item)
            max_length: Maximum sequence length
            num_workers: Number of worker processes
            shuffle: Whether to shuffle conversations
            max_samples: Max conversations to load
            min_turns: Minimum turns per conversation
            quality_threshold: Minimum quality score (0-1)

        Returns:
            DataLoader with turn-aware batching
        """
        dataset = TurnAwareConversationDataset(
            data_path=Path(data_path),
            tokenizer=tokenizer,
            max_length=max_length,
            max_samples=max_samples,
            include_turn_tokens=False,
            min_turns=min_turns,
            quality_threshold=quality_threshold,
        )

        collator = ConversationBatchCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            strategy="pad",
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )


# Convenience function for easy integration
def create_turn_aware_dataloaders(
    data_dir: str,
    tokenizer,
    train_file: str = "train_conversations.jsonl",
    val_file: str = "val_conversations.jsonl",
    batch_size: int = 32,
    max_length: int = 2048,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with turn awareness.

    Assumes data_dir contains train_file and val_file.
    """
    train_path = Path(data_dir) / train_file
    val_path = Path(data_dir) / val_file

    train_loader = TurnAwareConversationDataLoader.create_dataloader(
        data_path=str(train_path),
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = TurnAwareConversationDataLoader.create_dataloader(
        data_path=str(val_path),
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=0,  # No shuffling for validation
        shuffle=False,
    )

    return train_loader, val_loader
