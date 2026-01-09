"""
Generation manager for the Ava pipeline.

Handles sample generation for quality monitoring during training,
with support for async CPU generation and coherence evaluation.

Async Generation Pattern:
    To avoid blocking the training loop, generation runs in a background thread:

    Training Thread                    Background Thread (ThreadPoolExecutor)
    ──────────────────────────────     ─────────────────────────────────────
    train_step()
    |
    +-> generate_async(model, step)
    |   |
    |   +-> submit(_async_generate)    -> _async_generate() starts
    |       (returns immediately)         |
    |                                     +-> copy model to CPU (slow, ~1-5s)
    train_step()  (continues)             |
    |                                     +-> generate tokens (slow)
    train_step()                          |
    |                                     +-> decode & return result
    +-> process_completed_generations()
        (checks if Future is done)

    Key design decisions:
    1. ThreadPoolExecutor with max_workers=1 ensures only one generation at a time
    2. Model is deep-copied to CPU for generation to avoid blocking GPU
    3. History is capped at MAX_HISTORY_SIZE=100 to prevent memory leaks
    4. Lock protects _pending_future and _generation_history from race conditions

Coherence Measurement:
    Uses CoherenceMeasurer (if available) to evaluate:
    - Repetition ratio: Fraction of repeated n-grams
    - Semantic coherence: Cross-entropy consistency across windows
    - Overall coherence score: Weighted combination

History Capping:
    Generation history is limited to MAX_HISTORY_SIZE entries.
    Oldest entries are discarded when limit is reached.
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

from ava.core.checkpoint import load_state_dict_with_remapping
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
        # Lock to prevent race conditions on shared state
        self._lock = threading.Lock()
        # Generation config storage
        self._generation_config: Dict[str, Any] = {}

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

    def set_generation_config(self, config: Dict[str, Any]) -> None:
        """Set generation configuration for async generation."""
        self._generation_config = config or {}
        self.logger.debug(f"Generation config set: {list(self._generation_config.keys())}")

    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation configuration."""
        return self._generation_config.copy()

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
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        no_repeat_ngram_size: int = 0,
        banned_tokens: Optional[List[int]] = None,
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
            eos_token_id: End-of-sequence token ID (auto-detected from model/tokenizer if None)
            bos_token_id: Beginning-of-sequence token ID (auto-detected from model/tokenizer if None)
            pad_token_id: Padding token ID (auto-detected from model/tokenizer if None)
            no_repeat_ngram_size: Block repeated n-grams of this size (0 = disabled)
            banned_tokens: List of token IDs to never generate (e.g., [23] blocks '/')

        Returns:
            Generated text string
        """
        # Default banned tokens: block rare punctuation not in typical training data
        if banned_tokens is None:
            banned_tokens = [23]  # Block '/' by default
        self.assert_initialized()

        # Get special token IDs with fallback chain: explicit param -> model config -> tokenizer -> default
        # This ensures generation uses the same token IDs the model was trained with
        base_model = model.module if hasattr(model, 'module') else model
        model_config = getattr(base_model, 'config', None)

        # EOS token ID: param -> model config -> tokenizer -> default (1)
        if eos_token_id is None:
            if model_config is not None and hasattr(model_config, 'eos_token_id'):
                eos_token_id = model_config.eos_token_id
            elif tokenizer is not None:
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = 102  # [SEP] token (BERT-style)
                self.logger.debug(f"Using default eos_token_id={eos_token_id}")

        # BOS token ID: param -> model config -> tokenizer -> default (2)
        if bos_token_id is None:
            if model_config is not None and hasattr(model_config, 'bos_token_id'):
                bos_token_id = model_config.bos_token_id
            elif tokenizer is not None:
                bos_token_id = getattr(tokenizer, 'bos_token_id', None)
            if bos_token_id is None:
                bos_token_id = 101  # [CLS] token (BERT-style)
                self.logger.debug(f"Using default bos_token_id={bos_token_id}")

        # PAD token ID: param -> model config -> tokenizer -> default (0)
        if pad_token_id is None:
            if model_config is not None and hasattr(model_config, 'pad_token_id'):
                pad_token_id = model_config.pad_token_id
            elif tokenizer is not None:
                pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = 0
                self.logger.debug(f"Using default pad_token_id={pad_token_id}")

        model.eval()

        # Set hybrid cache to generation mode (enables KV cache eviction for long sequences)
        if hasattr(base_model, '_hybrid_cache_manager') and base_model._hybrid_cache_manager:
            base_model._hybrid_cache_manager.set_mode('generation')

        # Get device from model, not self.device (supports CPU generation)
        device = next(model.parameters()).device

        with torch.no_grad():
            # Prepare input - ALWAYS start with BOS token for coherent generation
            # The model was trained with BOS at position 0, so we must include it
            if prompt is not None and tokenizer is not None:
                try:
                    # Handle both HuggingFace tokenizers and tokenizers library
                    if hasattr(tokenizer, 'encode'):
                        enc_result = tokenizer.encode(prompt)
                        # tokenizers library returns Encoding object with .ids attribute
                        if hasattr(enc_result, 'ids'):
                            token_ids = enc_result.ids
                        # HuggingFace returns tensor or list directly
                        elif isinstance(enc_result, torch.Tensor):
                            token_ids = enc_result.squeeze(0).tolist()
                        else:
                            token_ids = list(enc_result)
                    else:
                        token_ids = [bos_token_id]

                    # CRITICAL FIX: Strip trailing EOS/SEP token if present
                    # BERT-style tokenizers automatically add [SEP] at the end of encoded text.
                    # For autoregressive generation, we must remove it so the model can
                    # generate new content instead of treating the sequence as "complete".
                    if token_ids and token_ids[-1] == eos_token_id:
                        token_ids = token_ids[:-1]
                        self.logger.debug(f"Stripped trailing EOS token from prompt encoding")

                    # Prepend BOS token if not already present
                    if token_ids[0] != bos_token_id:
                        token_ids = [bos_token_id] + token_ids
                    generated_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
                except Exception as e:
                    self.logger.warning(f"Failed to encode prompt: {e}")
                    # Start with just BOS token
                    generated_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)
            else:
                # No prompt: start with BOS token (not random!)
                generated_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(device)

            # Get max position embeddings
            max_pos = getattr(model, 'max_position_embeddings', 256)
            effective_max = min(max_length, max_pos)

            # Initialize attention mask (all 1s = attend to all tokens)
            attention_mask = torch.ones_like(generated_ids, dtype=torch.long)

            # Generate tokens
            for _ in range(effective_max - 1):
                if generated_ids.shape[1] >= max_pos:
                    break

                # Create position_ids for generation (sequential: 0, 1, 2, ...)
                # This ensures RoPE embeddings are applied correctly
                current_len = generated_ids.shape[1]
                position_ids = torch.arange(current_len, device=device).unsqueeze(0)

                # Pass attention mask and position_ids to model for proper masking
                outputs = model(generated_ids, attention_mask=attention_mask, position_ids=position_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                # Get next token logits
                next_logits = logits[:, -1, :] / max(temperature, 1e-5)

                # Apply repetition penalty (proper implementation for both positive and negative logits)
                # GPU SYNC FIX: Use single .tolist() instead of per-token .item() calls
                if repetition_penalty > 1.0 and generated_ids.shape[1] > 0:
                    recent_tokens = generated_ids[0, -50:].tolist()  # Single GPU sync
                    for token_id_int in recent_tokens:
                        if next_logits[0, token_id_int] < 0:
                            next_logits[0, token_id_int] *= repetition_penalty
                        else:
                            next_logits[0, token_id_int] /= repetition_penalty

                # Apply n-gram blocking to prevent repeated patterns
                if no_repeat_ngram_size > 0 and generated_ids.shape[1] >= no_repeat_ngram_size:
                    gen_list = generated_ids[0].tolist()
                    ngram_prefix = gen_list[-(no_repeat_ngram_size - 1):]
                    ngram_banned = set()
                    for j in range(len(gen_list) - no_repeat_ngram_size + 1):
                        if gen_list[j:j + no_repeat_ngram_size - 1] == ngram_prefix:
                            banned_token = gen_list[j + no_repeat_ngram_size - 1]
                            ngram_banned.add(banned_token)
                    for token_id in ngram_banned:
                        next_logits[0, token_id] = float('-inf')

                # Apply banned tokens filter (blocks rare/problematic tokens like '/')
                if banned_tokens:
                    for token_id in banned_tokens:
                        if token_id < next_logits.shape[-1]:
                            next_logits[0, token_id] = float('-inf')

                # Compute softmax probabilities
                probs = torch.softmax(next_logits, dim=-1)

                # Top-k filtering: keep only top_k highest probability tokens
                if top_k > 0 and top_k < probs.shape[-1]:
                    top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)

                    # Apply top-p (nucleus) sampling within top-k
                    sorted_probs, sort_idx = torch.sort(top_k_probs, descending=True, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum_probs > top_p
                    mask[..., 0] = False  # Always keep at least one token
                    sorted_probs[mask] = 0.0

                    # Normalize with fallback for invalid distributions
                    prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                    # Fallback to uniform if all probs are zero (can happen with aggressive filtering)
                    is_invalid = (prob_sum < 1e-10) | torch.isnan(prob_sum)
                    if is_invalid.any():
                        uniform_probs = torch.ones_like(sorted_probs) / sorted_probs.shape[-1]
                        sorted_probs = torch.where(is_invalid.expand_as(sorted_probs), uniform_probs, sorted_probs)
                        prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                    sorted_probs = sorted_probs / (prob_sum + 1e-10)

                    # Sample from sorted probs, then map back to original indices
                    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                    # Map back: sampled_idx -> sort_idx -> top_k_indices -> vocab
                    local_idx = sort_idx.gather(-1, sampled_idx)
                    next_token = top_k_indices.gather(-1, local_idx)
                else:
                    # No top-k, just use top-p on full distribution
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum_probs > top_p
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0

                    # Normalize with fallback for invalid distributions
                    prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                    is_invalid = (prob_sum < 1e-10) | torch.isnan(prob_sum)
                    if is_invalid.any():
                        uniform_probs = torch.ones_like(sorted_probs) / sorted_probs.shape[-1]
                        sorted_probs = torch.where(is_invalid.expand_as(sorted_probs), uniform_probs, sorted_probs)
                        prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                    sorted_probs = sorted_probs / (prob_sum + 1e-10)

                    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, sampled_idx)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Update attention mask for the new token
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)

                # EOS check: stop generation immediately when EOS token is produced
                # This uses a single .item() call per token, which is acceptable for
                # generation (not training) since we want to stop ASAP
                next_token_val = next_token[0, 0].item()
                if next_token_val == eos_token_id:
                    self.logger.debug(f"Generation stopped at EOS token ({eos_token_id}) after {generated_ids.shape[1]} tokens")
                    break

        # Decode with single sync point
        generated_cpu = generated_ids[0].cpu()
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        generated_list = generated_cpu.tolist()

        # COHERENCE FIX: Detect and warn about repetitive/degenerate generation
        # This helps diagnose coherence issues early
        if len(generated_list) > 10:
            unique_tokens = len(set(generated_list))
            unique_ratio = unique_tokens / len(generated_list)
            if unique_ratio < 0.1:
                self.logger.warning(
                    f"Degenerate generation detected: unique_ratio={unique_ratio:.3f} "
                    f"({unique_tokens}/{len(generated_list)} unique tokens). "
                    f"This indicates the model may not have learned properly. "
                    f"Check: 1) position_ids are being passed, 2) attention_mask is 2D for packed sequences, "
                    f"3) training data quality."
                )

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
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **generation_kwargs
    ) -> Optional[Future]:
        """
        Launch async generation on CPU (non-blocking).

        Args:
            model: Model to generate from
            global_step: Current training step
            vocab_size: Vocabulary size
            tokenizer: Optional tokenizer
            eos_token_id: End-of-sequence token ID (auto-detected from model/tokenizer if None)
            bos_token_id: Beginning-of-sequence token ID (auto-detected from model/tokenizer if None)
            pad_token_id: Padding token ID (auto-detected from model/tokenizer if None)
            **generation_kwargs: Generation parameters

        Returns:
            Future object, or None if previous generation still running
        """
        self.assert_initialized()

        # Get special token IDs with fallback chain: explicit param -> model config -> tokenizer -> default
        base_model = model.module if hasattr(model, 'module') else model
        model_config = getattr(base_model, 'config', None)

        # Auto-detect EOS token ID from model config/tokenizer if not provided
        if eos_token_id is None:
            if model_config is not None and hasattr(model_config, 'eos_token_id'):
                eos_token_id = model_config.eos_token_id
            elif tokenizer is not None:
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = 102  # [SEP] token (BERT-style)

        # Auto-detect BOS token ID from model config/tokenizer if not provided
        if bos_token_id is None:
            if model_config is not None and hasattr(model_config, 'bos_token_id'):
                bos_token_id = model_config.bos_token_id
            elif tokenizer is not None:
                bos_token_id = getattr(tokenizer, 'bos_token_id', None)
            if bos_token_id is None:
                bos_token_id = 101  # [CLS] token (BERT-style)

        # Auto-detect PAD token ID from model config/tokenizer if not provided
        if pad_token_id is None:
            if model_config is not None and hasattr(model_config, 'pad_token_id'):
                pad_token_id = model_config.pad_token_id
            elif tokenizer is not None:
                pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = 0

        # Use lock to prevent race conditions
        with self._lock:
            # Check if previous generation is still running
            if self._pending_future is not None and not self._pending_future.done():
                return None

            # CRITICAL: Capture state_dict on main thread to avoid race conditions
            # during deepcopy. The model's internal caches (RoPE, causal mask) can
            # change during training, causing "dictionary keys changed during iteration"
            # errors if we deepcopy from the background thread.
            with torch.no_grad():
                base_model = model.module if hasattr(model, 'module') else model
                # Unwrap torch.compile's OptimizedModule to get the actual model
                # torch.compile wraps models in OptimizedModule with original at _orig_mod
                if hasattr(base_model, '_orig_mod'):
                    base_model = base_model._orig_mod
                # Copy state dict to CPU immediately on main thread
                state_dict_cpu = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                # Also capture the model class and config for reconstruction
                model_class = type(base_model)
                model_config = getattr(base_model, 'config', None)

            def _run_generation():
                try:
                    # Reconstruct model on CPU using captured state_dict
                    # This avoids deepcopy race conditions with model caches
                    with torch.no_grad():
                        if model_config is not None:
                            cpu_model = model_class(model_config)
                        else:
                            # Fallback to deepcopy if no config available
                            cpu_model = copy.deepcopy(base_model)

                        # CRITICAL: Apply hybrid caching wrapper if checkpoint uses it
                        # The checkpoint has _wrapped_attn keys, so model must have same structure
                        try:
                            from ava.optimizations.hybrid_cache import (
                                apply_hybrid_caching,
                                HybridCacheConfig,
                                ActivationCacheConfig,
                                KVCacheConfig,
                            )
                            # Check if training config has hybrid caching enabled
                            hybrid_cfg = self.context.config.get('hybrid_caching', {})
                            if hybrid_cfg.get('enabled', False):
                                # Build proper config from YAML structure
                                act_cfg = hybrid_cfg.get('activation_cache', {})
                                kv_cfg = hybrid_cfg.get('kv_cache', {})
                                cache_config = HybridCacheConfig(
                                    enabled=True,
                                    mode=hybrid_cfg.get('mode', 'generation'),
                                    activation_cache=ActivationCacheConfig(
                                        enabled=act_cfg.get('enabled', True),
                                        max_size_gb=act_cfg.get('max_size_gb', 2.0),
                                        eviction_policy=act_cfg.get('eviction_policy', 'lru'),
                                    ),
                                    kv_cache=KVCacheConfig(
                                        enabled=kv_cfg.get('enabled', True),
                                        max_size_gb=kv_cfg.get('max_size_gb', 4.0),
                                        eviction_policy=kv_cfg.get('eviction_policy', 'hybrid'),
                                        sink_tokens=kv_cfg.get('sink_tokens', 4),
                                        recent_tokens=kv_cfg.get('recent_tokens', 256),
                                    ),
                                )
                                cpu_model, _ = apply_hybrid_caching(cpu_model, cache_config)
                                self.logger.debug("Applied hybrid caching wrapper for checkpoint compatibility")
                        except ImportError:
                            self.logger.debug("Hybrid caching not available, skipping wrapper")
                        except Exception as e:
                            self.logger.warning(f"Failed to apply hybrid caching wrapper: {e}")

                        # Load state dict with automatic remapping
                        # Handles _wrapped_attn keys and other architecture mismatches
                        success, incompatible = load_state_dict_with_remapping(
                            cpu_model, state_dict_cpu, strict=True
                        )

                        if not success:
                            raise RuntimeError("Failed to load checkpoint into CPU model for generation")

                        cpu_model = cpu_model.to('cpu')
                        cpu_model.eval()

                        # Generate with explicit token IDs for consistency
                        text = self.generate_sample(
                            cpu_model,
                            tokenizer,
                            vocab_size,
                            eos_token_id=eos_token_id,
                            bos_token_id=bos_token_id,
                            pad_token_id=pad_token_id,
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

                        # Use lock when modifying shared state
                        with self._lock:
                            self._add_to_history(result)
                        self._generation_queue.put(result)

                        # Save to single JSONL file (append mode for WandB upload)
                        if self._log_dir:
                            import json
                            gen_dir = self._log_dir / "async_generation"
                            gen_dir.mkdir(exist_ok=True, parents=True)
                            output_file = gen_dir / "generations.jsonl"
                            entry = {
                                "step": global_step,
                                "generated_text": text,
                                "temperature": generation_kwargs.get('temperature', 1.0),
                                "top_p": generation_kwargs.get('top_p', 1.0),
                                "top_k": generation_kwargs.get('top_k', 50),
                                "prompt": generation_kwargs.get('prompt', ''),
                            }
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(entry) + '\n')

                        # Cleanup
                        del cpu_model
                        return text

                except Exception as e:
                    # Log error (logger handles formatting)
                    self.logger.error(f"Async generation failed at step {global_step}: {e}")

                    # Record error in history for monitoring
                    with self._lock:
                        error_result = {
                            'step': global_step,
                            'error': str(e),
                            'error_type': type(e).__name__,
                        }
                        self._add_to_history(error_result)
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
        use_fast: bool = None,  # Now reads from config
        use_fp16: bool = None,  # Now reads from config
        use_bf16: bool = None,  # Now reads from config
        micro_batch_size: int = None,  # Now reads from config
    ) -> Optional[Dict[str, float]]:
        """
        Measure generation coherence.

        Args:
            model: Model to evaluate
            global_step: Current training step
            config: Coherence configuration
            use_fast: Use FastCoherenceMeasurer (default: from config or True)
            use_fp16: Use FP16 precision (default: from config or False)
            use_bf16: Use BF16 precision (default: from config or True)
            micro_batch_size: Batch size for micro-batching (default: from config or 8)

        Returns:
            Dictionary of coherence metrics, or None if unavailable
        """
        if not config.get('enabled', False):
            return None

        # Read settings from config (with fallback to parameters, then defaults)
        use_fast = config.get('use_fast', use_fast if use_fast is not None else True)
        use_fp16 = config.get('use_fp16', use_fp16 if use_fp16 is not None else False)
        use_bf16 = config.get('use_bf16', use_bf16 if use_bf16 is not None else True)
        micro_batch_size = config.get('micro_batch_size', micro_batch_size if micro_batch_size is not None else 8)

        try:
            from ava.training.coherence import (
                CoherenceMeasurer,
                FastCoherenceMeasurer,
                CoherenceConfig,
            )

            # Build config with correct parameter names
            # Check for both 'max_generation_length' (YAML) and 'max_length' (fallback)
            coherence_config = CoherenceConfig(
                num_samples=config.get('num_samples', 5),
                max_generation_length=config.get('max_generation_length', config.get('max_length', 100)),
                temperature=config.get('temperature', 0.8),
                top_p=config.get('top_p', 0.9),
                top_k=config.get('top_k', 50),
            )

            if use_fast:
                # Use optimized FastCoherenceMeasurer
                measurer = FastCoherenceMeasurer(
                    model=model,
                    tokenizer=self.context.tokenizer,
                    config=coherence_config,
                    device=self.device,
                    micro_batch_size=micro_batch_size,
                )

                # Generate samples first (use original measurer for generation)
                orig_measurer = CoherenceMeasurer(
                    model=model,
                    tokenizer=self.context.tokenizer,
                    config=coherence_config,
                    device=self.device,
                )
                input_ids, _ = orig_measurer._generate_samples()

                # Measure with fast path
                metrics = measurer.measure_fast(
                    input_ids,
                    use_fp16=use_fp16,
                    use_bf16=use_bf16,
                )
            else:
                # Use original CoherenceMeasurer
                measurer = CoherenceMeasurer(
                    model=model,
                    tokenizer=self.context.tokenizer,
                    config=coherence_config,
                    device=self.device,
                )
                metrics = measurer.measure(generate_samples=True)

            return {
                'coherence_score': metrics.coherence_score,
                'perplexity': metrics.perplexity,
                'repetition_score': metrics.repetition_score,
                'sentence_flow': metrics.sentence_flow_score,
                'topic_consistency': metrics.topic_consistency,
                'num_samples': metrics.num_samples,
                'avg_sequence_length': metrics.avg_sequence_length,
                'unique_token_ratio': metrics.unique_token_ratio,
            }

        except ImportError as e:
            # Track first occurrence to avoid spam
            if not hasattr(self, '_coherence_import_error_logged'):
                self._coherence_import_error_logged = True
                self.logger.error(
                    "Coherence measurement not available: missing 'ava.eval.coherence' module. "
                    "This will disable coherence tracking for the entire run. "
                    f"Import error: {e}"
                )
            else:
                self.logger.info("Coherence measurement skipped (module not available)")
            return None
        except Exception as e:
            # Full context for debugging
            self.logger.warning(
                f"Coherence measurement failed at step {global_step}: {e}"
            )
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
