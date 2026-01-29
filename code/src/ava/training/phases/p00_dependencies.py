"""
Phase 0: Dependency checking.

Optional quick verification that required packages are available.
Can be skipped with --skip-dep-check for faster startup.
"""

import logging
from pathlib import Path
from typing import List

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class DependencyPhase(TrainingPhase):
    """
    Phase 0: Verify required dependencies are available.

    This phase performs a quick check of required packages to catch
    missing dependencies early with clear error messages.

    Features:
        - Quick package verification (no heavy imports)
        - Optional: skip with --skip-dep-check
        - Graceful degradation if check script not found
    """

    name = "dependencies"
    description = "Check required packages"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Run dependency check if not skipped.

        Args:
            ctx: Phase context with args

        Returns:
            Unchanged context (dependencies don't modify state)
        """
        # Skip if requested
        if getattr(ctx.args, 'skip_dep_check', False):
            return ctx

        # Try to run dependency check
        try:
            # First try direct import
            try:
                from check_dependencies import quick_check
                quick_check(warn_only=True)
                return ctx
            except ImportError:
                pass

            # Try relative import from script location
            import importlib.util

            # Find check_dependencies.py relative to train_pipeline.py
            if hasattr(ctx.args, 'config') and ctx.args.config:
                config_path = Path(ctx.args.config).resolve()
                # Navigate to scripts directory
                scripts_dir = config_path.parent.parent.parent / 'scripts'
                dep_check_path = scripts_dir / 'check_dependencies.py'

                if dep_check_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "check_dependencies", dep_check_path
                    )
                    dep_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dep_module)
                    dep_module.quick_check(warn_only=True)
                else:
                    # Try alternate location
                    alt_path = Path(__file__).resolve().parents[4] / 'scripts' / 'check_dependencies.py'
                    if alt_path.exists():
                        spec = importlib.util.spec_from_file_location(
                            "check_dependencies", alt_path
                        )
                        dep_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(dep_module)
                        dep_module.quick_check(warn_only=True)

        except Exception as e:
            # Dependency check is optional - don't fail on errors
            self.log(ctx, f"Dependency check skipped: {e}", "debug")

        return ctx

    def validate(self, ctx: PhaseContext) -> List[str]:
        """No validation needed for dependency check."""
        return []
