"""
Memory module for continual learning and episodic memory.
"""

from .episodic_memory import (
    EpisodicMemoryBank,
    MemoryEntry,
    MemoryRetriever,
    AdaptiveMemoryManager,
    ExperienceReplay
)

__all__ = [
    "EpisodicMemoryBank",
    "MemoryEntry",
    "MemoryRetriever",
    "AdaptiveMemoryManager",
    "ExperienceReplay"
]