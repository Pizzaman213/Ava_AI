"""
Backward compatibility module for data loading.

Re-exports from factory.py for backward compatibility with older import paths.
"""

from .factory import create_streaming_dataloaders

__all__ = ['create_streaming_dataloaders']
