"""
Data loader factory functions.

This module provides factory functions for creating optimized dataloaders:
- create_dataloaders: Unified factory for all dataloader types

Usage:
    from ava.data.factory import create_dataloaders

    # Pretokenized mode (default, fastest)
    train_loader, val_loader = create_dataloaders(
        mode='pretokenized',
        batch_size=32,
        max_length=2048,
        data_dir='/path/to/data',
    )

    # Indexed mode (true random shuffling)
    train_loader, val_loader = create_dataloaders(
        mode='indexed',
        batch_size=32,
        max_length=2048,
        data_dir='/path/to/data',
    )
"""

from typing import Any, Tuple


def create_dataloaders(
    mode: str = 'pretokenized',
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Unified factory function for creating dataloaders.

    This is the single entry point for all dataloader creation. It dispatches
    to the appropriate specialized factory based on mode.

    Args:
        mode: Dataloader mode:
            - 'pretokenized': Ultra-fast pretokenized Arrow loading (default)
            - 'indexed': Map-style with true random shuffling
            - 'multi_column': Multi-column dataset support
            - 'conversation': Turn-aware conversation loading
        **kwargs: Arguments passed to the specialized factory

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> # Pretokenized mode (default)
        >>> train, val = create_dataloaders(
        ...     mode='pretokenized',
        ...     batch_size=32,
        ...     max_length=2048,
        ...     data_dir='/path/to/pretokenized',
        ... )

        >>> # Indexed mode
        >>> train, val = create_dataloaders(
        ...     mode='indexed',
        ...     data_dir='/path/to/data',
        ...     batch_size=32,
        ...     max_length=2048,
        ... )
    """
    if mode == 'pretokenized':
        from .pretokenized import create_ultra_fast_dataloaders
        return create_ultra_fast_dataloaders(**kwargs)

    elif mode == 'indexed':
        from .indexed import create_indexed_dataloaders
        return create_indexed_dataloaders(**kwargs)

    elif mode == 'multi_column':
        from .multi_column import create_multi_column_dataloader
        # multi_column returns a single dataloader, wrap for consistency
        loader = create_multi_column_dataloader(**kwargs)
        return loader, None

    elif mode == 'conversation':
        from .conversation import create_turn_aware_dataloaders
        return create_turn_aware_dataloaders(**kwargs)

    else:
        raise ValueError(
            f"Unknown dataloader mode: {mode}. "
            f"Valid modes: pretokenized, indexed, multi_column, conversation"
        )


__all__ = [
    'create_dataloaders',
]
