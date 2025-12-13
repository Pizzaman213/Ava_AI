"""
Generation manager for the Ava pipeline.

Handles sample generation for quality monitoring during training,
with support for async CPU generation and coherence evaluation.
"""

import copy
import io
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .context import ManagerInterface, TrainingContext

logger = logging.getLogger(__name__)

# Maximum history entries to prevent memory leak
MAX_HISTORY_SIZE = 100


class GenerationManager(ManagerInterface):
    """
    Manages text generation for quality monitoring during training.

    Handles:
        - Synchronous sample generation
        - Asynchronous generation on CPU (non-blocking)
        - Generation history tracking (capped to prevent memory leak)
        - Coherence measurement integration

    Thread Safety:
        Uses a lock to protect shared state (_pending_future, _generation_history)
        from race conditions when accessed from multiple threads.

    Example:
        >>> context = TrainingContext(model=model, device=device)
        >>> gen_mgr = GenerationManager(context)
        >>> gen_mgr.initialize()
        >>> text = gen_mgr.generate_sample(model, tokenizer, config)
        >>> gen_mgr.generate_async(model, step, config)  # Non-blocking
    """

    def __init__(self, context: TrainingContext):
        """
        Initialize the generation manager.

        Args:
            context: Training context with shared state
        """
        super().__init__(context)
        self.executor: Optional[ThreadPoolExecutor] = None
        self.coherence_measurer = None
        self._pending_future: Optional[Future] = None
        self._generation_history: List[Dict[str, Any]] = []
        self._generation_queue: Queue = Queue()
        self._log_dir: Optional[Path] = None
        # FIX: Add lock to prevent TOCTOU race conditions on shared state
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize thread pool executor for async generation."""
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="AsyncGeneration"
        )
        self._initialized = True
        self.logger.debug("GenerationManager initialized")

    def cleanup(self) -> None:
        """Shutdown executor and wait for pending generation."""
        if self._pending_future is not None and not self._pending_future.done():
            try:
                self._pending_future.result(timeout=30)
            except Exception as e:
                self.logger.warning(f"Generation cleanup timeout: {e}")

        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def set_log_dir(self, log_dir: Path) -> None:
        """Set directory for generation output files."""
        self._log_dir = Path(log_dir)

    def generate_sample(
        self,
        model: nn.Module,
        tokenizer: Any,
        vocab_size: int,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        prompt: Optional[str] = None,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Generate sample text synchronously.

        Args:
            model: Model to generate from
            tokenizer: Tokenizer for encoding/decoding
            vocab_size: Vocabulary size
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            prompt: Optional prompt text
            skip_special_tokens: Skip special tokens in output

        Returns:
            Generated text string
        """
        self.assert_initialized()

        model.eval()
        device = self.device

        with torch.no_grad():
            # Prepare input
            if prompt is not None and tokenizer is not None:
                try:
                    encoded = tokenizer.encode(prompt, return_tensors='pt')
                    generated_ids = encoded.to(device)
                except Exception as e:
                    self.logger.warning(f"Failed to encode prompt: {e}")
                    generated_ids = torch.randint(0, vocab_size, (1, 1)).to(device)
            else:
                generated_ids = torch.randint(0, vocab_size, (1, 1)).to(device)

            # Get max position embeddings
            max_pos = getattr(model, 'max_position_embeddings', 256)
            effective_max = min(max_length, max_pos)

            # Generate tokens
            for _ in range(effective_max - 1):
                if generated_ids.shape[1] >= max_pos:
                    break

                outputs = model(generated_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                # Get next token logits
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)

                # Apply repetition penalty
                if repetition_penalty > 1.0 and generated_ids.shape[1] > 0:
                    recent = generated_ids[0, -50:]
                    next_logits[:, recent] /= repetition_penalty

                # Top-k filtering
                probs = torch.softmax(next_logits, dim=-1)
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
                    probs_filtered = torch.zeros_like(probs)
                    probs_filtered.scatter_(-1, top_k_indices, top_k_probs)
                    probs = probs_filtered

                # Top-p (nucleus) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs > top_p
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # Sample
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, next_token)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # GPU SYNC FIX: Batch EOS checks to reduce sync frequency
                # Check for EOS every 8 tokens instead of every token
                # This reduces syncs from N to N/8 during generation
                seq_len = generated_ids.shape[1]
                if seq_len % 8 == 0 or seq_len >= effective_max - 1:
                    # Check if any of the last 8 tokens (or fewer) is EOS
                    check_start = max(0, seq_len - 8)
                    if (generated_ids[0, check_start:] == 0).any().item():
                        break

        # Decode
        # GPU SYNC FIX: Use single sync instead of double sync from .cpu().tolist()
        # .cpu() triggers a sync, then .tolist() triggers another sync
        # Instead: transfer to CPU with explicit sync, then tolist() is just CPU operation
        generated_cpu = generated_ids[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        generated_list = generated_cpu.tolist()

        if tokenizer is not None:
            try:
                text = tokenizer.decode(generated_list, skip_special_tokens=skip_special_tokens)
                if prompt:
                    return f"Prompt: {prompt}\nGenerated: {text}"
                return text
            except Exception as e:
                self.logger.warning(f"Decode failed: {e}")

        return f"Token IDs: {generated_list[:50]}..."

    def generate_async(
        self,
        model: nn.Module,
        global_step: int,
        vocab_size: int,
        tokenizer: Any = None,
        **generation_kwargs
    ) -> Optional[Future]:
        """
        Launch async generation on CPU (non-blocking).

        Args:
            model: Model to generate from
            global_step: Current training step
            vocab_size: Vocabulary size
            tokenizer: Optional tokenizer
            **generation_kwargs: Generation parameters

        Returns:
            Future object, or None if previous generation still running
        """
        self.assert_initialized()

        # FIX: Use lock to prevent TOCTOU race condition
        # Without lock, another thread could start generation between check and assignment
        with self._lock:
            # Check if previous generation is still running
            if self._pending_future is not None and not self._pending_future.done():
                return None

            def _run_generation():
                try:
                    # Create CPU copy
                    # GPU SYNC FIX: Use non_blocking=True transfers to overlap GPU->CPU DMA
                    # with CPU work. All transfers are enqueued first, then we sync once.
                    with torch.no_grad():
                        # First, transfer all tensors with non_blocking=True
                        # This enqueues all GPU->CPU DMA transfers in parallel
                        model_state = {}
                        for k, v in model.state_dict().items():
                            if v.is_cuda:
                                # non_blocking=True allows GPU to continue while DMA runs
                                model_state[k] = v.to('cpu', non_blocking=True)
                            else:
                                model_state[k] = v.clone()

                        # Single sync after all transfers are enqueued
                        # This waits for all non_blocking transfers to complete
                        if torch.cuda.is_available():
                            torch.cuda.current_stream().synchronize()

                        # Get model class
                        if hasattr(model, 'module'):
                            model_class = type(model.module)
                            config = getattr(model.module, 'config', None)
                        else:
                            model_class = type(model)
                            config = getattr(model, 'config', None)

                        # Create CPU model
                        if config is not None:
                            cpu_model = model_class(config)
                            cpu_model.load_state_dict(model_state)
                        else:
                            cpu_model = copy.deepcopy(model)
                            cpu_model.cpu()

                        cpu_model.eval()

                        # Generate
                        text = self.generate_sample(
                            cpu_model,
                            tokenizer,
                            vocab_size,
                            **generation_kwargs
                        )

                        # Record result
                        result = {
                            'step': global_step,
                            'generated_text': text,
                            'temperature': generation_kwargs.get('temperature', 1.0),
                            'top_p': generation_kwargs.get('top_p', 1.0),
                            'top_k': generation_kwargs.get('top_k', 50),
                            'prompt': generation_kwargs.get('prompt', ''),
                        }

                        # FIX: Use lock when modifying shared state
                        with self._lock:
                            self._add_to_history(result)
                        self._generation_queue.put(result)

                        # Save to file
                        if self._log_dir:
                            gen_dir = self._log_dir / "async_generation"
                            gen_dir.mkdir(exist_ok=True, parents=True)
                            output_file = gen_dir / f"generation_step_{global_step}.txt"
                            with open(output_file, 'w') as f:
                                f.write(f"=== Generation at Step {global_step} ===\n")
                                f.write(text)
                                f.write("\n=== End ===\n")

                        # Cleanup
                        del cpu_model
                        del model_state

                        return text

                except Exception as e:
                    self.logger.error(f"Async generation failed: {e}")
                    return None

            self._pending_future = self.executor.submit(_run_generation)
            return self._pending_future

    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """Add result to history with size cap to prevent memory leak."""
        self._generation_history.append(result)

        # Cap history size
        if len(self._generation_history) > MAX_HISTORY_SIZE:
            self._generation_history = self._generation_history[-MAX_HISTORY_SIZE:]

    def process_completed_generations(self) -> List[Dict[str, Any]]:
        """
        Process completed async generations.

        Returns:
            List of completed generation results
        """
        results = []
        while not self._generation_queue.empty():
            try:
                result = self._generation_queue.get_nowait()
                results.append(result)
            except Exception as e:
                logger.debug(f"Queue get_nowait failed (queue may be empty): {e}")
                break
        return results

    def get_generation_history(self) -> List[Dict[str, Any]]:
        """Get generation history (capped to MAX_HISTORY_SIZE)."""
        return self._generation_history.copy()

    def measure_coherence(
        self,
        model: nn.Module,
        global_step: int,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """
        Measure generation coherence.

        Args:
            model: Model to evaluate
            global_step: Current training step
            config: Coherence configuration

        Returns:
            Dictionary of coherence metrics, or None if unavailable
        """
        if not config.get('enabled', False):
            return None

        try:
            from ava.eval.coherence import CoherenceMeasurer, CoherenceConfig

            if self.coherence_measurer is None:
                coherence_config = CoherenceConfig(
                    num_samples=config.get('num_samples', 5),
                    max_length=config.get('max_length', 100),
                    temperature=config.get('temperature', 0.8),
                )
                self.coherence_measurer = CoherenceMeasurer(coherence_config)

            metrics = self.coherence_measurer.measure(model, self.device)

            return {
                'coherence_score': metrics.overall_score,
                'fluency': metrics.fluency,
                'repetition_ratio': metrics.repetition_ratio,
            }

        except ImportError:
            self.logger.debug("Coherence measurement not available")
            return None
        except Exception as e:
            self.logger.warning(f"Coherence measurement failed: {e}")
            return None

    def is_generation_pending(self) -> bool:
        """Check if async generation is still running."""
        return self._pending_future is not None and not self._pending_future.done()

    def get_status(self) -> Dict[str, Any]:
        """Return current generation status."""
        return {
            'total_generations': len(self._generation_history),
            'pending': self.is_generation_pending(),
            'queue_size': self._generation_queue.qsize(),
        }

    def on_error(self, error: Exception) -> None:
        """Handle generation errors."""
        self.logger.error(f"Generation error: {error}", exc_info=True)
