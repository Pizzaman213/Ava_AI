"""MoE++ Language Learning Model Package

Note: This package provides interfaces to the Ava MoE++ architecture.
The actual model implementations are in the src.Ava package.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checking imports - won't run at runtime
    from typing import Any as EnhancedMoEModel
    from typing import Any as EnhancedMoEConfig
    from typing import Any as TextGenerator
else:
    try:
        # Import from the actual Ava package structure
        from src.Ava import EnhancedMoEModel as MoEModel  # type: ignore[attr-defined]
        from src.Ava import EnhancedMoEConfig  # type: ignore[attr-defined]
        from src.generation.generator import TextGenerator  # type: ignore[attr-defined]
        __all__ = ["MoEModel", "EnhancedMoEConfig", "TextGenerator"]
    except ImportError:
        # Handle cases where modules are not available
        MoEModel = None  # type: ignore[assignment,misc]
        EnhancedMoEConfig = None  # type: ignore[assignment,misc]
        TextGenerator = None  # type: ignore[assignment,misc]
        __all__ = []