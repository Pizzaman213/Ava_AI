"""
Test CUDA graph cleanup in TrainingLoopManager.

Tests proper resource cleanup to prevent GPU memory leaks.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add code/src to path for src.ava imports
code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir))

import torch


def test_training_loop_manager_import():
    """Test that TrainingLoopManager imports correctly."""
    from src.ava.training.loop import TrainingLoopManager
    print("✓ TrainingLoopManager imports successfully")


def test_cleanup_cuda_graph_exists():
    """Test that _cleanup_cuda_graph method exists."""
    from src.ava.training.loop import TrainingLoopManager

    # Check method exists
    assert hasattr(TrainingLoopManager, '_cleanup_cuda_graph'), \
        "_cleanup_cuda_graph method should exist"

    print("✓ _cleanup_cuda_graph method exists")


def test_cleanup_sets_attributes_to_none():
    """Test that cleanup properly nullifies graph attributes."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    # Create a mock model and context
    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    # Set some graph state
    manager._cuda_graph = MagicMock()
    manager._cuda_graph_captured = True
    manager._graph_static_input = {'input_ids': torch.zeros(1)}
    manager._graph_static_loss = torch.zeros(1)
    manager._graph_batch_size = 32
    manager._graph_seq_len = 512

    # Call cleanup
    manager._cleanup_cuda_graph()

    # Verify all attributes are reset
    assert manager._cuda_graph is None, "_cuda_graph should be None"
    assert manager._cuda_graph_captured is False, "_cuda_graph_captured should be False"
    assert manager._graph_static_input is None, "_graph_static_input should be None"
    assert manager._graph_static_loss is None, "_graph_static_loss should be None"
    assert manager._graph_batch_size is None, "_graph_batch_size should be None"
    assert manager._graph_seq_len is None, "_graph_seq_len should be None"

    print("✓ Cleanup properly nullifies all graph attributes")


def test_cleanup_calls_graph_reset():
    """Test that cleanup calls reset() on CUDA graph."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    # Create a mock CUDA graph
    mock_graph = MagicMock()
    manager._cuda_graph = mock_graph
    manager._cuda_graph_captured = True

    # Call cleanup
    manager._cleanup_cuda_graph()

    # Verify reset was called
    mock_graph.reset.assert_called_once()

    print("✓ Cleanup calls reset() on CUDA graph")


def test_cleanup_handles_none_graph():
    """Test that cleanup handles None graph gracefully."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    # Ensure graph is None
    manager._cuda_graph = None
    manager._cuda_graph_captured = False

    # This should not raise any exception
    try:
        manager._cleanup_cuda_graph()
        print("✓ Cleanup handles None graph gracefully")
    except Exception as e:
        assert False, f"Cleanup should not raise exception: {e}"


def test_cleanup_handles_reset_exception():
    """Test that cleanup handles exceptions from graph.reset()."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    # Create a mock CUDA graph that raises on reset
    mock_graph = MagicMock()
    mock_graph.reset.side_effect = RuntimeError("Graph already invalid")
    manager._cuda_graph = mock_graph
    manager._cuda_graph_captured = True

    # This should not raise any exception
    try:
        manager._cleanup_cuda_graph()
        print("✓ Cleanup handles reset() exception gracefully")
    except Exception as e:
        assert False, f"Cleanup should catch reset exceptions: {e}"


def test_public_cleanup_calls_cuda_graph_cleanup():
    """Test that public cleanup() method calls _cleanup_cuda_graph()."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    # Set graph state
    manager._cuda_graph = MagicMock()
    manager._cuda_graph_captured = True

    # Call public cleanup
    manager.cleanup()

    # Verify graph was cleaned up
    assert manager._cuda_graph is None, "cleanup() should call _cleanup_cuda_graph()"
    assert manager._cuda_graph_captured is False

    print("✓ Public cleanup() calls _cleanup_cuda_graph()")


def test_step_losses_list_exists():
    """Test that _step_losses list is initialized."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    assert hasattr(manager, '_step_losses'), "_step_losses should exist"
    assert isinstance(manager._step_losses, list), "_step_losses should be a list"

    print("✓ _step_losses list is initialized")


def test_reset_loss_accumulator_clears_step_losses():
    """Test that _reset_loss_accumulator clears _step_losses."""
    from src.ava.training.loop import TrainingLoopManager
    from src.ava.training.context import TrainingContext

    mock_model = MagicMock()
    context = TrainingContext(model=mock_model)
    context.device = torch.device('cpu')

    manager = TrainingLoopManager(context)

    # Add some step losses
    manager._step_losses = [1.0, 2.0, 3.0]

    # Reset
    manager._reset_loss_accumulator()

    assert len(manager._step_losses) == 0, "_step_losses should be empty after reset"

    print("✓ _reset_loss_accumulator clears _step_losses")


def run_all_tests():
    """Run all tests."""
    print("\n=== CUDA Graph Cleanup Tests ===\n")

    test_training_loop_manager_import()
    test_cleanup_cuda_graph_exists()
    test_cleanup_sets_attributes_to_none()
    test_cleanup_calls_graph_reset()
    test_cleanup_handles_none_graph()
    test_cleanup_handles_reset_exception()
    test_public_cleanup_calls_cuda_graph_cleanup()
    test_step_losses_list_exists()
    test_reset_loss_accumulator_clears_step_losses()

    print("\n=== All CUDA Graph Cleanup tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
