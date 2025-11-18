"""
Evaluation Manager for Ava Training Pipeline

Handles all model evaluation including:
- Generation quality testing
- Loss and perplexity computation
- Resume validation (smoke tests)
"""

from typing import Any, List, Optional, Tuple

import torch

from .base import TrainingComponent, TrainingContext


class EvaluationManager(TrainingComponent):
    """Manages model evaluation and quality testing."""

    def __init__(self, context: TrainingContext):
        """Initialize evaluation manager.

        Args:
            context: Training context with configuration
        """
        super().__init__(context)

    def test_generation_quality(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        test_prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.2,
    ) -> dict:
        """Test generation quality with sample prompts.

        Args:
            model: Model to test
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run on
            test_prompts: List of prompts to generate from
            max_length: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Dictionary with generation results and metrics
        """
        from src.Ava.utils.logging import get_logger

        model.eval()
        results = {
            "generated_texts": [],
            "repetition_scores": [],
            "avg_length": 0,
            "perplexity": None,
            "coherence": None,
        }

        total_length = 0
        all_generated_tokens = []

        with torch.no_grad():
            for prompt in test_prompts:
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                )
                input_ids = inputs["input_ids"].to(device, non_blocking=True)
                attention_mask = inputs["attention_mask"].to(device, non_blocking=True)
                prompt_len = input_ids.shape[1]

                try:
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=min(prompt_len + max_length, 512),
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                    generated_text = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )
                    results["generated_texts"].append(generated_text)

                    # Extract generated tokens (exclude prompt)
                    tokens = generated_ids[0][prompt_len:].tolist()

                    # Calculate repetition score (legacy metric)
                    if len(tokens) > 4:
                        trigrams = [
                            tuple(tokens[i : i + 3])
                            for i in range(len(tokens) - 2)
                        ]
                        repetition = (
                            1.0 - (len(set(trigrams)) / len(trigrams))
                            if len(trigrams) > 0
                            else 0.0
                        )
                    else:
                        repetition = 0.0

                    results["repetition_scores"].append(repetition)
                    total_length += len(tokens)
                    all_generated_tokens.append(tokens)

                except Exception as e:
                    get_logger().error(
                        f"    ⚠️  Generation failed for prompt '{prompt[:30]}...': {e}"
                    )
                    results["generated_texts"].append("[GENERATION FAILED]")
                    results["repetition_scores"].append(1.0)

        if len(test_prompts) > 0:
            results["avg_length"] = total_length / len(test_prompts)

        # Calculate coherence metrics if available
        if all_generated_tokens:
            try:
                from src.Ava.evaluation.coherence import quick_coherence_test

                coherence_metrics = quick_coherence_test(all_generated_tokens)
                results["coherence"] = coherence_metrics
            except Exception as e:
                get_logger().error(f"    ⚠️  Coherence calculation failed: {e}")
                results["coherence"] = None

        model.train()
        return results

    @torch.compile(
        mode="reduce-overhead", fullgraph=False, disable=False
    )
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: Any,
        device: torch.device,
        use_bf16: bool = False,
        max_batches: Optional[int] = None,
        training_config: Optional[Any] = None,
        stream_manager: Optional[Any] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Evaluate model and return loss and perplexity.

        Args:
            model: Model to evaluate
            dataloader: Validation dataloader
            device: Device to run on
            use_bf16: Whether to use BF16 precision
            max_batches: Maximum batches to evaluate
            training_config: Training configuration
            stream_manager: Optional stream manager for async GPU transfers

        Returns:
            Tuple of (avg_loss, perplexity)
        """
        from src.Ava.utils.logging import get_logger

        # Get max_batches from config if not provided
        if max_batches is None:
            if training_config and hasattr(training_config, "evaluation"):
                max_batches = getattr(
                    training_config.evaluation,
                    "default_max_validation_batches",
                    50,
                )
            else:
                max_batches = 50

        # Get cleanup setting from config
        cleanup_enabled = False
        if training_config and hasattr(training_config, "performance"):
            cleanup_enabled = getattr(
                training_config.performance, "enable_gpu_memory_cleanup", False
            )

        if cleanup_enabled and torch.cuda.is_available():
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        model.eval()

        # Reset expert counts if model has MoE
        if hasattr(model, "moe_layer") and hasattr(
            model.moe_layer, "reset_expert_counts"
        ):
            model.moe_layer.reset_expert_counts()
        elif hasattr(model, "reset_expert_counts"):
            model.reset_expert_counts()

        total_loss = 0.0
        num_valid_batches = 0
        num_invalid_batches = 0
        num_nan_losses = 0
        num_inf_losses = 0
        total_batches_processed = 0
        total_tokens = 0

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    max_batches_safe = max_batches if max_batches is not None else 100
                    if batch_idx >= max_batches_safe:
                        break

                    total_batches_processed += 1

                    # Move batch to device
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch.get("labels", input_ids)

                    if input_ids.device != device:
                        if stream_manager is not None:
                            input_ids = stream_manager.to_gpu_async(
                                input_ids, non_blocking=True
                            )
                            attention_mask = stream_manager.to_gpu_async(
                                attention_mask, non_blocking=True
                            )
                            labels = stream_manager.to_gpu_async(
                                labels, non_blocking=True
                            )
                            stream_manager.wait_for_transfers()
                        else:
                            input_ids = input_ids.to(device, non_blocking=True)
                            attention_mask = attention_mask.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)

                    # Mark CUDA graph boundary
                    if (
                        hasattr(torch, "compiler")
                        and hasattr(torch.compiler, "cudagraph_mark_step_begin")
                    ):
                        torch.compiler.cudagraph_mark_step_begin()

                    # Forward pass with autocast
                    autocast_dtype = (
                        torch.bfloat16 if use_bf16 else torch.float16
                    )
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        dtype=autocast_dtype if device.type == "cuda" else torch.float32,
                    ):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )

                    if "loss" not in outputs:
                        get_logger().info(
                            f"WARNING: Model outputs missing 'loss' at batch {total_batches_processed}"
                        )
                        continue

                    loss = outputs["loss"]

                    # Reduce loss if needed
                    if loss.dim() > 0:
                        loss = loss.mean()

                    # Get logging limits from config
                    max_nan_logs = (
                        getattr(training_config.evaluation, "max_nan_loss_logs", 5)
                        if training_config
                        else 5
                    )
                    max_inf_logs = (
                        getattr(training_config.evaluation, "max_inf_loss_logs", 5)
                        if training_config
                        else 5
                    )

                    # Handle loss validity
                    if torch.isnan(loss):
                        num_nan_losses += 1
                        num_invalid_batches += 1
                        if num_nan_losses <= max_nan_logs:
                            get_logger().info(
                                f"WARNING: NaN loss in evaluation (batch {total_batches_processed})"
                            )
                    elif torch.isinf(loss):
                        num_inf_losses += 1
                        num_invalid_batches += 1
                        if num_inf_losses <= max_inf_logs:
                            get_logger().info(
                                f"WARNING: Infinite loss in evaluation (batch {total_batches_processed})"
                            )
                    elif torch.isfinite(loss):
                        total_loss += loss.item()
                        num_valid_batches += 1
                        if attention_mask is not None:
                            total_tokens += attention_mask.sum().item()
                        else:
                            total_tokens += input_ids.numel()
                    else:
                        num_invalid_batches += 1

        finally:
            if cleanup_enabled and torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.train()

        # Calculate metrics
        if num_valid_batches == 0:
            if total_batches_processed == 0:
                get_logger().info(
                    "⚠️  WARNING: Validation dataloader is empty"
                )
                return (None, None)
            else:
                get_logger().info(
                    f"CRITICAL: All {total_batches_processed} validation batches had invalid losses!"
                )
                return (float("inf"), float("inf"))

        avg_valid_loss = total_loss / num_valid_batches

        # Calculate perplexity
        import math

        try:
            perplexity_threshold = (
                getattr(training_config.evaluation, "perplexity_overflow_threshold", 20)
                if training_config
                else 20
            )
            perplexity = (
                math.exp(avg_valid_loss)
                if avg_valid_loss < perplexity_threshold
                else float("inf")
            )
        except (ValueError, OverflowError, AttributeError):
            perplexity = None

        # Report validation health
        if num_invalid_batches > 0:
            invalid_rate = num_invalid_batches / total_batches_processed
            get_logger().info(
                f"⚠️  Validation health: {num_invalid_batches}/{total_batches_processed} "
                f"batches had invalid losses ({invalid_rate:.1%})"
            )
            get_logger().info(
                f"    Valid batches: {num_valid_batches}, Avg loss: {avg_valid_loss:.4f}"
            )
            get_logger().info(
                f"    NaN losses: {num_nan_losses}, Infinite losses: {num_inf_losses}"
            )

        return avg_valid_loss, perplexity

    def resume_smoke_test(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        device: torch.device,
        num_steps: int = 3,
    ) -> bool:
        """Quick smoke test to verify checkpoint resume works.

        Args:
            model: Model to test
            train_loader: Training dataloader
            device: Device to run on
            num_steps: Number of steps to run

        Returns:
            True if test passes, False otherwise
        """
        from src.Ava.utils.logging import get_logger

        get_logger().info(f"Running resume smoke test ({num_steps} steps)...")
        try:
            model.train()
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch.get("labels", input_ids)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                if "loss" not in outputs:
                    get_logger().error("Model did not return loss!")
                    return False

            get_logger().info("✓ Resume smoke test passed")
            return True

        except Exception as e:
            get_logger().error(f"✗ Resume smoke test failed: {e}")
            return False
