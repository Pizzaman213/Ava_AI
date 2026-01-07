"""
Coherence Measurement Module for LLM Training.

Measures text coherence through multiple metrics:
- Perplexity-based coherence (lower perplexity = more coherent)
- Repetition penalty (excessive repetition indicates incoherence)
- Sentence flow consistency (adjacent sentence similarity)
- Topic consistency (semantic drift detection)

These metrics are computed during training evaluation to track
model quality over time.

Optimizations:
- Single forward pass for all hidden-state-based metrics
- Micro-batching for large batch sizes (prevents OOM)
- Optional half-precision computation
- Vectorized n-gram repetition scoring
- Cached hidden states across metric computations
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CoherenceConfig from central config location (avoid duplication)
from ..config.training_config import CoherenceConfig

logger = logging.getLogger(__name__)

# Default micro-batch size for memory efficiency
DEFAULT_MICRO_BATCH_SIZE = 8


@dataclass
class CoherenceMetrics:
    """Container for coherence measurement results."""

    # Core metrics
    perplexity: float = 0.0
    repetition_score: float = 0.0  # 0-1, lower is better (less repetition)
    sentence_flow_score: float = 0.0  # 0-1, higher is better
    topic_consistency: float = 0.0  # 0-1, higher is better

    # Aggregate score (weighted combination)
    coherence_score: float = 0.0  # 0-1, higher is better

    # Additional details
    num_samples: int = 0
    avg_sequence_length: float = 0.0
    unique_token_ratio: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'coherence/perplexity': self.perplexity,
            'coherence/repetition_score': self.repetition_score,
            'coherence/sentence_flow': self.sentence_flow_score,
            'coherence/topic_consistency': self.topic_consistency,
            'coherence/overall_score': self.coherence_score,
            'coherence/num_samples': float(self.num_samples),
            'coherence/avg_seq_length': self.avg_sequence_length,
            'coherence/unique_token_ratio': self.unique_token_ratio,
        }


class CoherenceMeasurer:
    """
    Measures text coherence for LLM outputs.

    Computes multiple coherence metrics:
    1. Perplexity-based: Uses model's own loss as coherence indicator
    2. Repetition: Detects excessive n-gram repetition
    3. Sentence flow: Measures semantic consistency between adjacent parts
    4. Topic consistency: Detects semantic drift across the text
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        config: Optional[CoherenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize coherence measurer.

        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            config: Coherence measurement configuration
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CoherenceConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Cache for hidden states (used for flow/topic metrics)
        self._hidden_cache: List[torch.Tensor] = []

    @torch.no_grad()
    def measure(
        self,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        generate_samples: bool = False,
        prompts: Optional[List[str]] = None,
    ) -> CoherenceMetrics:
        """
        Measure coherence of provided texts or generated samples.

        Args:
            texts: List of text strings to evaluate
            input_ids: Pre-tokenized input (alternative to texts)
            generate_samples: If True, generate samples from model
            prompts: Optional prompts for generation

        Returns:
            CoherenceMetrics with all computed scores
        """
        self.model.eval()

        # Get input_ids from texts or generate samples
        if generate_samples:
            input_ids, texts = self._generate_samples(prompts)
        elif texts is not None and self.tokenizer is not None:
            input_ids = self._tokenize_texts(texts)
        elif input_ids is None:
            raise ValueError("Must provide texts, input_ids, or set generate_samples=True")

        # Ensure input_ids is on correct device
        if input_ids is not None:
            input_ids = input_ids.to(self.device)

        # Compute unique token ratio first (needed for validation)
        unique_ratio = self._compute_unique_ratio(input_ids)

        # Compute actual perplexity
        perplexity = self._compute_perplexity(input_ids)

        # Detect repetitive/degenerate generations (e.g., all PAD tokens)
        # If unique ratio < 10%, perplexity may be artificially low (~1.0)
        # This happens when model generates "[PAD] [PAD] [PAD]..." and predicts it perfectly
        # Apply this check to ALL inputs, not just generated samples
        if unique_ratio < 0.10:
            if generate_samples:
                logger.warning(
                    f"Detected repetitive generation (unique_ratio={unique_ratio:.4f}). "
                    f"Perplexity measurement may be unreliable (actual perplexity={perplexity:.2f})."
                )
            else:
                logger.warning(
                    f"Detected repetitive sequences (unique_ratio={unique_ratio:.4f}). "
                    f"This may indicate corrupted data, tokenizer issues, or degenerate generation. "
                    f"Perplexity measurement unreliable (actual perplexity={perplexity:.2f})."
                )

        # Compute other metrics
        repetition_score = self._compute_repetition_score(input_ids)
        flow_score = self._compute_sentence_flow(input_ids)
        topic_score = self._compute_topic_consistency(input_ids)

        # Compute aggregate score
        # Normalize perplexity to 0-1 using log-scale (lower perplexity = higher score)
        # Log-scale is more appropriate for exponential metrics like perplexity
        if perplexity >= self.config.max_perplexity:
            ppl_normalized = 0.0
        elif perplexity <= 1.0:
            ppl_normalized = 1.0  # Perfect perplexity
        else:
            log_ppl = math.log(perplexity)
            log_max = math.log(self.config.max_perplexity)
            ppl_normalized = 1.0 - (log_ppl / log_max)

        # Invert repetition (lower repetition = higher score)
        rep_inverted = 1.0 - repetition_score

        # FIX: Normalize by weight sum to ensure coherence_score is in [0, 1]
        # Without normalization, non-uniform weights produce unbounded scores
        weight_sum = (
            self.config.perplexity_weight +
            self.config.repetition_weight +
            self.config.flow_weight +
            self.config.topic_weight
        )

        coherence_score = (
            self.config.perplexity_weight * ppl_normalized +
            self.config.repetition_weight * rep_inverted +
            self.config.flow_weight * flow_score +
            self.config.topic_weight * topic_score
        ) / max(weight_sum, 1e-8)

        return CoherenceMetrics(
            perplexity=perplexity,
            repetition_score=repetition_score,
            sentence_flow_score=flow_score,
            topic_consistency=topic_score,
            coherence_score=coherence_score,
            num_samples=input_ids.shape[0] if input_ids is not None else 0,
            avg_sequence_length=input_ids.shape[1] if input_ids is not None else 0,
            unique_token_ratio=unique_ratio,
        )

    def _tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """Tokenize list of texts."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text input")

        encoded = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_generation_length,
        )
        return encoded['input_ids']

    def _generate_samples(
        self,
        prompts: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Generate text samples from model."""
        num_samples = self.config.num_samples

        if prompts is None:
            # Use default prompts or start tokens
            if self.tokenizer is not None:
                # Use BOS token or common start
                start_token = self.tokenizer.bos_token_id or 0
                input_ids = torch.full(
                    (num_samples, 1),
                    start_token,
                    dtype=torch.long,
                    device=self.device
                )
            else:
                input_ids = torch.zeros(
                    (num_samples, 1),
                    dtype=torch.long,
                    device=self.device
                )
        else:
            # Tokenize prompts
            input_ids = self._tokenize_texts(prompts[:num_samples]).to(self.device)

        # Generate using model
        generated = self._sample_from_model(
            input_ids,
            max_length=self.config.max_generation_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        # Decode to texts
        texts = []
        if self.tokenizer is not None:
            for seq in generated:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                texts.append(text)

        return generated, texts

    def _sample_from_model(
        self,
        input_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Sample tokens from model autoregressively."""
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Track which sequences have finished (hit EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Get EOS and PAD token IDs if available
        eos_token_id = None
        pad_token_id = None
        if self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else eos_token_id

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.model(generated)

            # Get logits for last position
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits'][:, -1, :]
            elif isinstance(outputs, tuple):
                logits = outputs[0][:, -1, :]
            else:
                logits = outputs[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)

            # Safety: ensure valid probability distribution for multinomial
            # If all probs are zero (e.g., all tokens filtered), use uniform distribution
            prob_sum = probs.sum(dim=-1, keepdim=True)
            invalid_mask = (prob_sum == 0) | torch.isnan(prob_sum)
            if invalid_mask.any():
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs = torch.where(invalid_mask.expand_as(probs), uniform, probs)

            next_token = torch.multinomial(probs, num_samples=1)

            # For finished sequences, use pad token instead of sampled token
            if eos_token_id is not None and pad_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, pad_token_id),
                    next_token
                )

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Update finished status
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

                # Stop if all sequences are finished
                if finished.all():
                    break

        return generated

    def _validate_hidden_states(
        self,
        hidden_states: Optional[torch.Tensor],
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Validate and normalize hidden state dimensions.

        Handles both [batch, seq, hidden] and [seq, batch, hidden] formats.

        Args:
            hidden_states: Hidden states from model output
            input_ids: Input token IDs for dimension validation

        Returns:
            Normalized hidden states in [batch, seq, hidden] format, or None if invalid
        """
        if hidden_states is None or hidden_states.dim() != 3:
            return None

        batch_size, seq_len = input_ids.shape

        # Check [batch, seq, hidden] format
        if hidden_states.shape[0] == batch_size and hidden_states.shape[1] == seq_len:
            hidden_dim = hidden_states.shape[2]

            # Sanity check: hidden_dim should be reasonable (64-8192)
            if not (64 <= hidden_dim <= 8192):
                logger.warning(
                    f"Unusual hidden dimension: {hidden_dim}. "
                    f"Expected 64-8192. Shape: {hidden_states.shape}"
                )

            return hidden_states

        # Check [seq, batch, hidden] format - transpose
        elif hidden_states.shape[0] == seq_len and hidden_states.shape[1] == batch_size:
            logger.debug(
                f"Transposing hidden states from [seq, batch, hidden] to [batch, seq, hidden]. "
                f"Original shape: {hidden_states.shape}"
            )
            return hidden_states.transpose(0, 1)

        else:
            # Dimension mismatch
            logger.warning(
                f"Hidden states dimension mismatch. "
                f"Input: [batch={batch_size}, seq={seq_len}], "
                f"Hidden: {hidden_states.shape}. "
                f"Cannot compute hidden-state-based metrics."
            )
            return None

    def _compute_perplexity(self, input_ids: torch.Tensor) -> float:
        """Compute perplexity on input sequences.

        CRITICAL FIX: Now creates and uses attention mask to exclude padding tokens.
        Without this, padding tokens corrupt the perplexity calculation.
        """
        if input_ids.shape[1] < 2:
            logger.debug("Sequence too short for perplexity (<2 tokens), using neutral score")
            # Use geometric mean for neutral score instead of max penalty
            return math.sqrt(self.config.max_perplexity)

        # Create attention mask (1 for real tokens, 0 for padding)
        # Assume pad_token_id is 0 (common default)
        pad_token_id = 0
        if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
            if self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id

        attention_mask = (input_ids != pad_token_id).long()

        # Shift for causal LM loss
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        attention_mask_shifted = attention_mask[:, :-1].contiguous()

        # Forward pass with attention mask
        outputs = self.model(inputs, attention_mask=attention_mask_shifted)

        # Get logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Create loss mask: only compute loss on non-padding tokens
        # Shape: [batch, seq_len-1]
        loss_mask = attention_mask[:, 1:].contiguous()

        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss_mask_flat = loss_mask.view(-1)

        # Compute cross-entropy only on non-masked positions
        # Set labels to -100 (ignore_index) where mask is 0
        labels_masked = labels_flat.clone()
        labels_masked[loss_mask_flat == 0] = -100

        loss = F.cross_entropy(
            logits_flat,
            labels_masked,
            ignore_index=-100,
            reduction='mean'
        )

        # FIX: Clamp loss BEFORE exp to prevent perplexity > max_perplexity
        # exp(20) ≈ 485M but max_perplexity is typically 100
        # Clamp to log(max_perplexity) to ensure perplexity <= max_perplexity
        max_loss = math.log(self.config.max_perplexity)
        perplexity = torch.exp(torch.clamp(loss, max=max_loss)).item()

        return perplexity

    def _compute_repetition_score(self, input_ids: torch.Tensor) -> float:
        """
        Compute repetition score based on n-gram repetition.

        Returns:
            Score 0-1 where higher means more repetition (bad)
        """
        # GPU SYNC FIX: Single batch transfer instead of per-sequence .tolist()
        input_ids_cpu = input_ids.cpu()
        if input_ids.is_cuda:
            torch.cuda.synchronize()
        all_seqs = input_ids_cpu.tolist()  # Already on CPU, no additional sync

        total_rep_ratio = 0.0
        num_valid = 0

        for seq_list in all_seqs:
            for n in self.config.ngram_sizes:
                if len(seq_list) < n:
                    continue

                # Extract n-grams
                ngrams = [tuple(seq_list[i:i+n]) for i in range(len(seq_list) - n + 1)]

                if len(ngrams) == 0:
                    continue

                # Count unique vs total
                unique_ngrams = len(set(ngrams))
                total_ngrams = len(ngrams)

                # Repetition ratio (1 - unique/total)
                rep_ratio = 1.0 - (unique_ngrams / total_ngrams)
                total_rep_ratio += rep_ratio
                num_valid += 1

        if num_valid == 0:
            return 0.0

        return total_rep_ratio / num_valid

    def _compute_sentence_flow(self, input_ids: torch.Tensor) -> float:
        """
        Compute sentence flow score using hidden state similarity.

        Measures how smoothly the text flows by comparing adjacent
        hidden state representations.

        Returns:
            Score 0-1 where higher means better flow
        """
        seq_len = input_ids.shape[1]
        if seq_len < 10:
            if seq_len < 2:
                logger.debug("Sequence very short (<2 tokens), using neutral score")
                return 0.5  # Neutral score
            else:
                # Partial credit: linear scale from 0.3 to 0.5
                logger.debug(f"Sequence short ({seq_len} tokens), using scaled score")
                return 0.3 + (seq_len - 2) * (0.2 / 8)

        try:
            # Get hidden states from model
            outputs = self.model(input_ids)

            # Try to get hidden states - handle various output formats
            hidden_states = None

            if isinstance(outputs, dict):
                # EnhancedMoEModel returns {'hidden_states': tensor, 'last_hidden_state': tensor, ...}
                if 'hidden_states' in outputs:
                    hs = outputs['hidden_states']
                    # Check if it's a list of layers or the final hidden state
                    if isinstance(hs, (list, tuple)):
                        hidden_states = hs[-1]  # Last layer
                    elif isinstance(hs, torch.Tensor) and hs.dim() == 3:
                        hidden_states = hs  # Already the final hidden state [batch, seq, hidden]
                elif 'last_hidden_state' in outputs:
                    hidden_states = outputs['last_hidden_state']
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hs = outputs.hidden_states
                if isinstance(hs, (list, tuple)):
                    hidden_states = hs[-1]
                elif isinstance(hs, torch.Tensor) and hs.dim() == 3:
                    hidden_states = hs
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state

            # Validate and normalize hidden states dimensions
            hidden_states = self._validate_hidden_states(hidden_states, input_ids)

            if hidden_states is None:
                # Model doesn't output hidden states or dimensions invalid
                logger.debug("Hidden states unavailable or invalid, using logit-based flow")
                return self._compute_logit_flow(input_ids)

            # Compute cosine similarity between adjacent positions
            # Split sequence into chunks and compare
            chunk_size = max(input_ids.shape[1] // 4, 2)
            similarities = []

            for i in range(0, input_ids.shape[1] - chunk_size, chunk_size):
                chunk2_start = i + chunk_size
                chunk2_end = min(chunk2_start + chunk_size, hidden_states.shape[1])

                # Skip if second chunk would be empty or too small
                if chunk2_start >= hidden_states.shape[1] or chunk2_end - chunk2_start < 1:
                    break

                chunk1 = hidden_states[:, i:i+chunk_size, :].mean(dim=1)
                chunk2 = hidden_states[:, chunk2_start:chunk2_end, :].mean(dim=1)

                # Cosine similarity
                sim = F.cosine_similarity(chunk1, chunk2, dim=-1)
                similarities.append(sim.mean())  # Keep on GPU

            if len(similarities) == 0:
                logger.debug("No chunk comparisons computed for sentence flow")
                return 0.0  # No comparisons - return low score

            # Stack on GPU, single sync at the end
            avg_sim = torch.stack(similarities).mean().item()

            # Map from [-1, 1] to [0, 1]
            return (avg_sim + 1) / 2

        except Exception as e:
            logger.warning(
                f"Sentence flow computation failed: {e}. "
                f"Falling back to logit-based computation. "
                f"Input shape: {input_ids.shape}",
                exc_info=True
            )
            return self._compute_logit_flow(input_ids)

    def _compute_logit_flow(self, input_ids: torch.Tensor) -> float:
        """Fallback flow computation using logit distributions."""
        outputs = self.model(input_ids)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute all KL divergences on GPU, single sync at end
        # Compare adjacent token probability distributions
        if probs.shape[1] < 2:
            return 0.5

        # Compute all adjacent KL divergences at once on GPU
        p1 = probs[:, :-1, :]  # [batch, seq-1, vocab]
        p2 = probs[:, 1:, :]   # [batch, seq-1, vocab]

        # KL divergence per position (batch mean over batch dimension)
        # Using sum reduction over vocab, then mean over batch
        # Add epsilon to prevent log(0) = -inf which causes NaN
        eps = 1e-10
        p1_safe = p1.clamp(min=eps)
        p2_safe = p2.clamp(min=eps)
        kl_divs = (p2_safe * (p2_safe.log() - p1_safe.log())).sum(dim=-1).mean(dim=0)  # [seq-1]

        # Convert to similarity (inverse, capped) - all on GPU
        similarities = 1.0 / (1.0 + kl_divs)

        # Single sync here
        return similarities.mean().item()

    def _compute_topic_consistency(self, input_ids: torch.Tensor) -> float:
        """
        Compute topic consistency by comparing start vs end of sequence.

        Measures semantic drift - whether the text stays on topic
        throughout the sequence.

        Returns:
            Score 0-1 where higher means better topic consistency
        """
        seq_len = input_ids.shape[1]
        if seq_len < 20:
            if seq_len < 10:
                logger.debug("Sequence very short (<10 tokens), using neutral score")
                return 0.5  # Neutral score
            else:
                # Partial credit: linear scale from 0.3 to 0.5
                logger.debug(f"Sequence short ({seq_len} tokens), using scaled score")
                return 0.3 + (seq_len - 10) * (0.2 / 10)

        try:
            # Get hidden states
            outputs = self.model(input_ids)

            # Try to get hidden states - handle various output formats
            hidden_states = None

            if isinstance(outputs, dict):
                # EnhancedMoEModel returns {'hidden_states': tensor, 'last_hidden_state': tensor, ...}
                if 'hidden_states' in outputs:
                    hs = outputs['hidden_states']
                    if isinstance(hs, (list, tuple)):
                        hidden_states = hs[-1]
                    elif isinstance(hs, torch.Tensor) and hs.dim() == 3:
                        hidden_states = hs
                elif 'last_hidden_state' in outputs:
                    hidden_states = outputs['last_hidden_state']
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hs = outputs.hidden_states
                if isinstance(hs, (list, tuple)):
                    hidden_states = hs[-1]
                elif isinstance(hs, torch.Tensor) and hs.dim() == 3:
                    hidden_states = hs
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state

            # Validate and normalize hidden states dimensions
            hidden_states = self._validate_hidden_states(hidden_states, input_ids)

            if hidden_states is None:
                logger.debug("Cannot compute topic consistency: hidden states unavailable or invalid")
                return 0.5  # Neutral score (consistent with Issue 2 fix)

            # Compare first quarter vs last quarter
            # Ensure at least 1 token per quarter to avoid empty tensor NaN
            quarter = max(input_ids.shape[1] // 4, 1)

            start_repr = hidden_states[:, :quarter, :].mean(dim=1)
            end_repr = hidden_states[:, -quarter:, :].mean(dim=1)

            # Cosine similarity
            sim = F.cosine_similarity(start_repr, end_repr, dim=-1)

            # Map from [-1, 1] to [0, 1]
            return ((sim.mean().item() + 1) / 2)

        except Exception as e:
            logger.warning(
                f"Topic consistency computation failed: {e}. "
                f"Returning neutral score 0.5 (not 0.0). "
                f"Input shape: {input_ids.shape}",
                exc_info=True
            )
            return 0.5  # Neutral score (consistent with Issue 2 fix)

    def _compute_unique_ratio(self, input_ids: torch.Tensor) -> float:
        """Compute ratio of unique tokens in sequences.

        GPU SYNC FIX: Count unique tokens on GPU, single sync at end.

        FIX: Computes batch-wide unique ratio, not per-sequence average.
        Example: [[1,2,3], [1,2,3]] should be 3/6=0.5, not (3+3)/6=1.0
        """
        total_tokens = input_ids.numel()

        if total_tokens == 0:
            return 0.0

        # FIX: Count unique tokens across entire batch (not per-sequence)
        # Flatten to 1D and count unique values
        unique_count = torch.unique(input_ids).numel()

        return unique_count / total_tokens


class FastCoherenceMeasurer:
    """
    Optimized coherence measurer for large batch sizes.

    Key optimizations:
    - Single forward pass for all hidden-state metrics
    - Micro-batching to prevent OOM on large batches
    - Optional half-precision (FP16/BF16) computation
    - Vectorized n-gram repetition scoring
    - Reuses cached hidden states across metrics

    Example:
        >>> measurer = FastCoherenceMeasurer(model, tokenizer, micro_batch_size=16)
        >>> metrics = measurer.measure_fast(input_ids, use_fp16=True)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        config: Optional[CoherenceConfig] = None,
        device: Optional[torch.device] = None,
        micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
    ):
        """
        Initialize fast coherence measurer.

        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            config: Coherence measurement configuration
            device: Device for computation
            micro_batch_size: Batch size for micro-batching (prevents OOM)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CoherenceConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.micro_batch_size = micro_batch_size

        # Cached outputs from single forward pass
        self._cached_logits: Optional[torch.Tensor] = None
        self._cached_hidden: Optional[torch.Tensor] = None
        self._cached_input_ids: Optional[torch.Tensor] = None

    def _clear_cache(self) -> None:
        """Clear cached tensors to free memory."""
        self._cached_logits = None
        self._cached_hidden = None
        self._cached_input_ids = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def measure_fast(
        self,
        input_ids: torch.Tensor,
        use_fp16: bool = False,
        use_bf16: bool = False,
    ) -> CoherenceMetrics:
        """
        Fast coherence measurement with single forward pass.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            use_fp16: Use FP16 for reduced memory
            use_bf16: Use BF16 for reduced memory (preferred on Ampere+)

        Returns:
            CoherenceMetrics with all computed scores
        """
        self.model.eval()
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]

        # Determine precision
        autocast_dtype = None
        if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif use_fp16:
            autocast_dtype = torch.float16

        # Micro-batch processing for large batches
        if batch_size > self.micro_batch_size:
            return self._measure_with_microbatching(input_ids, autocast_dtype)

        # Single batch processing
        return self._measure_single_batch(input_ids, autocast_dtype)

    def _measure_single_batch(
        self,
        input_ids: torch.Tensor,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> CoherenceMetrics:
        """Measure coherence for a single batch with single forward pass."""
        try:
            # Create attention mask (1 for real tokens, 0 for padding)
            pad_token_id = 0
            if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
                if self.tokenizer.pad_token_id is not None:
                    pad_token_id = self.tokenizer.pad_token_id

            attention_mask = (input_ids != pad_token_id).long()

            # Single forward pass to get both logits and hidden states
            # Check if DeepSpeed is managing the model (avoid autocast conflict)
            is_deepspeed = hasattr(self.model, 'module') and hasattr(self.model, 'deepspeed_io')

            if autocast_dtype is not None and not is_deepspeed:
                # Only use torch.amp.autocast if DeepSpeed is not handling mixed precision
                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    outputs = self.model(input_ids, attention_mask=attention_mask)
            else:
                # Let DeepSpeed handle mixed precision via its own config
                outputs = self.model(input_ids, attention_mask=attention_mask)

            # Extract logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Extract hidden states
            hidden_states = self._extract_hidden_states(outputs)

            # Compute unique ratio first (needed for validation)
            unique_ratio = self._compute_unique_ratio_fast(input_ids)

            # Compute actual perplexity
            perplexity = self._compute_perplexity_from_logits(logits, input_ids, attention_mask)

            # Detect repetitive/degenerate sequences
            # If unique ratio < 10%, perplexity may be artificially low (~1.0)
            # This applies to all inputs (generated or validation data)
            if unique_ratio < 0.10:
                logger.warning(
                    f"Detected repetitive sequences (unique_ratio={unique_ratio:.4f}). "
                    f"This may indicate corrupted data, tokenizer issues, or degenerate generation. "
                    f"Perplexity measurement unreliable (actual perplexity={perplexity:.2f})."
                )

            repetition_score = self._compute_repetition_vectorized(input_ids)

            if hidden_states is not None:
                flow_score = self._compute_flow_from_hidden(hidden_states)
                topic_score = self._compute_topic_from_hidden(hidden_states)
            else:
                # Fallback to logit-based metrics
                flow_score = self._compute_flow_from_logits(logits)
                topic_score = 0.0

            # Compute aggregate score
            # Use log-scale normalization (consistent with CoherenceMeasurer)
            if perplexity >= self.config.max_perplexity:
                ppl_normalized = 0.0
            elif perplexity <= 1.0:
                ppl_normalized = 1.0
            else:
                log_ppl = math.log(perplexity)
                log_max = math.log(self.config.max_perplexity)
                ppl_normalized = 1.0 - (log_ppl / log_max)
            rep_inverted = 1.0 - repetition_score

            # FIX: Normalize by weight sum to ensure coherence_score is in [0, 1]
            weight_sum = (
                self.config.perplexity_weight +
                self.config.repetition_weight +
                self.config.flow_weight +
                self.config.topic_weight
            )

            coherence_score = (
                self.config.perplexity_weight * ppl_normalized +
                self.config.repetition_weight * rep_inverted +
                self.config.flow_weight * flow_score +
                self.config.topic_weight * topic_score
            ) / max(weight_sum, 1e-8)

            return CoherenceMetrics(
                perplexity=perplexity,
                repetition_score=repetition_score,
                sentence_flow_score=flow_score,
                topic_consistency=topic_score,
                coherence_score=coherence_score,
                num_samples=input_ids.shape[0],
                avg_sequence_length=float(input_ids.shape[1]),
                unique_token_ratio=unique_ratio,
            )

        finally:
            self._clear_cache()

    def _measure_with_microbatching(
        self,
        input_ids: torch.Tensor,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> CoherenceMetrics:
        """Process large batches in micro-batches to prevent OOM."""
        batch_size = input_ids.shape[0]
        num_microbatches = (batch_size + self.micro_batch_size - 1) // self.micro_batch_size

        # Accumulators
        total_ppl = 0.0
        total_rep = 0.0
        total_flow = 0.0
        total_topic = 0.0
        total_unique = 0.0
        total_samples = 0

        for i in range(num_microbatches):
            start_idx = i * self.micro_batch_size
            end_idx = min(start_idx + self.micro_batch_size, batch_size)
            micro_batch = input_ids[start_idx:end_idx]

            # Measure this micro-batch
            metrics = self._measure_single_batch(micro_batch, autocast_dtype)

            # Accumulate weighted by sample count
            n = metrics.num_samples
            total_ppl += metrics.perplexity * n
            total_rep += metrics.repetition_score * n
            total_flow += metrics.sentence_flow_score * n
            total_topic += metrics.topic_consistency * n
            total_unique += metrics.unique_token_ratio * n
            total_samples += n

            # Free memory between micro-batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if total_samples == 0:
            return CoherenceMetrics()

        # Compute weighted averages
        avg_ppl = total_ppl / total_samples
        avg_rep = total_rep / total_samples
        avg_flow = total_flow / total_samples
        avg_topic = total_topic / total_samples
        avg_unique = total_unique / total_samples

        # Recompute aggregate score
        # Use log-scale normalization (consistent with CoherenceMeasurer)
        if avg_ppl >= self.config.max_perplexity:
            ppl_normalized = 0.0
        elif avg_ppl <= 1.0:
            ppl_normalized = 1.0
        else:
            log_ppl = math.log(avg_ppl)
            log_max = math.log(self.config.max_perplexity)
            ppl_normalized = 1.0 - (log_ppl / log_max)
        rep_inverted = 1.0 - avg_rep

        # FIX: Normalize by weight sum to ensure coherence_score is in [0, 1]
        weight_sum = (
            self.config.perplexity_weight +
            self.config.repetition_weight +
            self.config.flow_weight +
            self.config.topic_weight
        )

        coherence_score = (
            self.config.perplexity_weight * ppl_normalized +
            self.config.repetition_weight * rep_inverted +
            self.config.flow_weight * avg_flow +
            self.config.topic_weight * avg_topic
        ) / max(weight_sum, 1e-8)

        return CoherenceMetrics(
            perplexity=avg_ppl,
            repetition_score=avg_rep,
            sentence_flow_score=avg_flow,
            topic_consistency=avg_topic,
            coherence_score=coherence_score,
            num_samples=total_samples,
            avg_sequence_length=float(input_ids.shape[1]),
            unique_token_ratio=avg_unique,
        )

    def _extract_hidden_states(self, outputs) -> Optional[torch.Tensor]:
        """Extract hidden states from model outputs."""
        hidden_states = None

        if isinstance(outputs, dict):
            if 'hidden_states' in outputs:
                hs = outputs['hidden_states']
                if isinstance(hs, (list, tuple)):
                    hidden_states = hs[-1]
                elif isinstance(hs, torch.Tensor) and hs.dim() == 3:
                    hidden_states = hs
            elif 'last_hidden_state' in outputs:
                hidden_states = outputs['last_hidden_state']
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            if isinstance(hs, (list, tuple)):
                hidden_states = hs[-1]
            elif isinstance(hs, torch.Tensor) and hs.dim() == 3:
                hidden_states = hs
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state

        return hidden_states

    def _compute_perplexity_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Compute perplexity from pre-computed logits.

        CRITICAL FIX: Now uses attention mask to exclude padding tokens.
        Without this, padding tokens corrupt the perplexity calculation.

        Args:
            logits: Model logits [batch, seq, vocab]
            input_ids: Input token IDs [batch, seq]
            attention_mask: Attention mask [batch, seq] (1 for real tokens, 0 for padding)

        Returns:
            Perplexity value (clamped to max_perplexity)
        """
        if input_ids.shape[1] < 2:
            return self.config.max_perplexity

        # Shift for causal LM loss
        # logits: [batch, seq, vocab], input_ids: [batch, seq]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Create loss mask: only compute loss on non-padding tokens
        # Shift attention mask to align with labels
        loss_mask = attention_mask[:, 1:].contiguous()

        # Flatten tensors
        logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        labels_flat = shift_labels.view(-1)
        loss_mask_flat = loss_mask.view(-1)

        # Mask labels: set to -100 (ignore_index) where attention mask is 0
        labels_masked = labels_flat.clone()
        labels_masked[loss_mask_flat == 0] = -100

        loss = F.cross_entropy(
            logits_flat,
            labels_masked,
            ignore_index=-100,
            reduction='mean'
        )

        # FIX: Clamp loss BEFORE exp to prevent perplexity > max_perplexity
        max_loss = math.log(self.config.max_perplexity)
        perplexity = torch.exp(torch.clamp(loss, max=max_loss)).item()
        return perplexity

    def _compute_repetition_vectorized(self, input_ids: torch.Tensor) -> float:
        """
        Vectorized repetition scoring - faster than loop-based.

        Uses GPU-accelerated unique counting for bigrams/trigrams.
        """
        # Transfer to CPU once
        input_ids_cpu = input_ids.cpu()
        if input_ids.is_cuda:
            torch.cuda.synchronize()

        batch_size, seq_len = input_ids_cpu.shape

        if seq_len < 2:
            return 0.0

        total_rep = 0.0
        count = 0

        # Vectorized bigram extraction for entire batch
        for n in self.config.ngram_sizes:
            if seq_len < n:
                continue

            # Create n-gram keys by shifting and combining
            # For bigrams: key = token[i] * vocab_size + token[i+1]
            # This allows vectorized unique counting
            ngram_keys = input_ids_cpu[:, :seq_len - n + 1].clone()
            for offset in range(1, n):
                # Use large multiplier to create unique keys
                ngram_keys = ngram_keys * 100000 + input_ids_cpu[:, offset:seq_len - n + 1 + offset]

            # Count unique per sequence
            for b in range(batch_size):
                seq_ngrams = ngram_keys[b]
                unique_count = torch.unique(seq_ngrams).numel()
                total_count = seq_ngrams.numel()

                if total_count > 0:
                    rep_ratio = 1.0 - (unique_count / total_count)
                    total_rep += rep_ratio
                    count += 1

        return total_rep / count if count > 0 else 0.0

    def _compute_flow_from_hidden(self, hidden_states: torch.Tensor) -> float:
        """Compute flow score from cached hidden states."""
        if hidden_states.dim() != 3:
            logger.debug(f"Invalid hidden states: {hidden_states.dim()}D, expected 3D")
            return 0.5  # Neutral score

        seq_len = hidden_states.shape[1]
        if seq_len < 10:
            if seq_len < 2:
                return 0.5  # Neutral score
            else:
                # Partial credit: linear scale from 0.3 to 0.5
                return 0.3 + (seq_len - 2) * (0.2 / 8)
        chunk_size = max(seq_len // 4, 2)

        # Vectorized chunk comparison
        num_chunks = seq_len // chunk_size
        if num_chunks < 2:
            return 0.5

        # Reshape into chunks and compute means
        # Truncate to fit exact chunks
        truncated_len = num_chunks * chunk_size
        reshaped = hidden_states[:, :truncated_len, :].view(
            hidden_states.shape[0], num_chunks, chunk_size, hidden_states.shape[2]
        )
        chunk_means = reshaped.mean(dim=2)  # [batch, num_chunks, hidden]

        # Compare adjacent chunks
        chunk1 = chunk_means[:, :-1, :]  # [batch, num_chunks-1, hidden]
        chunk2 = chunk_means[:, 1:, :]   # [batch, num_chunks-1, hidden]

        # Batch cosine similarity
        sim = F.cosine_similarity(chunk1, chunk2, dim=-1)  # [batch, num_chunks-1]
        avg_sim = sim.mean().item()

        return (avg_sim + 1) / 2

    def _compute_topic_from_hidden(self, hidden_states: torch.Tensor) -> float:
        """Compute topic consistency from cached hidden states."""
        if hidden_states.dim() != 3:
            logger.debug(f"Invalid hidden states: {hidden_states.dim()}D, expected 3D")
            return 0.5  # Neutral score

        seq_len = hidden_states.shape[1]
        if seq_len < 20:
            if seq_len < 10:
                return 0.5  # Neutral score
            else:
                # Partial credit: linear scale from 0.3 to 0.5
                return 0.3 + (seq_len - 10) * (0.2 / 10)
        quarter = max(seq_len // 4, 1)

        # Compare first quarter to last quarter
        start_repr = hidden_states[:, :quarter, :].mean(dim=1)
        end_repr = hidden_states[:, -quarter:, :].mean(dim=1)

        sim = F.cosine_similarity(start_repr, end_repr, dim=-1)
        return (sim.mean().item() + 1) / 2

    def _compute_flow_from_logits(self, logits: torch.Tensor) -> float:
        """Fallback flow computation from logits."""
        if logits.shape[1] < 2:
            return 0.5

        # Use top-k probabilities for efficiency (don't need full vocab)
        probs = F.softmax(logits, dim=-1)

        # Compare adjacent distributions using JS divergence (symmetric, bounded)
        p1 = probs[:, :-1, :]
        p2 = probs[:, 1:, :]
        m = (p1 + p2) / 2

        # JS divergence = 0.5 * (KL(p1||m) + KL(p2||m))
        eps = 1e-10
        kl1 = (p1 * (torch.log(p1 + eps) - torch.log(m + eps))).sum(dim=-1)
        kl2 = (p2 * (torch.log(p2 + eps) - torch.log(m + eps))).sum(dim=-1)
        js_div = 0.5 * (kl1 + kl2)

        # Convert to similarity
        similarity = 1.0 / (1.0 + js_div)
        return similarity.mean().item()

    def _compute_unique_ratio_fast(self, input_ids: torch.Tensor) -> float:
        """Fast unique ratio computation.

        FIX: Computes batch-wide unique ratio (already correct implementation).
        """
        total_tokens = input_ids.numel()
        if total_tokens == 0:
            return 0.0

        # Flatten and count unique across entire batch
        unique_count = torch.unique(input_ids).numel()
        return unique_count / total_tokens


def measure_coherence_fast(
    model: nn.Module,
    input_ids: torch.Tensor,
    tokenizer: Optional[Any] = None,
    config: Optional[CoherenceConfig] = None,
    device: Optional[torch.device] = None,
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE,
    use_fp16: bool = False,
    use_bf16: bool = False,
) -> CoherenceMetrics:
    """
    Fast coherence measurement optimized for large batches.

    Args:
        model: Language model
        input_ids: Token IDs to evaluate
        tokenizer: Optional tokenizer
        config: Coherence configuration
        device: Computation device
        micro_batch_size: Size for micro-batching (prevents OOM)
        use_fp16: Use FP16 precision
        use_bf16: Use BF16 precision (preferred on Ampere+ GPUs)

    Returns:
        CoherenceMetrics with all scores
    """
    measurer = FastCoherenceMeasurer(
        model, tokenizer, config, device, micro_batch_size
    )
    return measurer.measure_fast(input_ids, use_fp16=use_fp16, use_bf16=use_bf16)


def measure_coherence(
    model: nn.Module,
    input_ids: torch.Tensor,
    tokenizer: Optional[Any] = None,
    config: Optional[CoherenceConfig] = None,
    device: Optional[torch.device] = None,
) -> CoherenceMetrics:
    """
    Convenience function to measure coherence on input sequences.

    Args:
        model: Language model
        input_ids: Token IDs to evaluate
        tokenizer: Optional tokenizer
        config: Coherence configuration
        device: Computation device

    Returns:
        CoherenceMetrics with all scores
    """
    measurer = CoherenceMeasurer(model, tokenizer, config, device)
    return measurer.measure(input_ids=input_ids)


def measure_batch_coherence(
    model: nn.Module,
    dataloader,
    tokenizer: Optional[Any] = None,
    config: Optional[CoherenceConfig] = None,
    device: Optional[torch.device] = None,
    max_batches: int = 10,
) -> CoherenceMetrics:
    """
    Measure coherence across multiple batches from a dataloader.

    Args:
        model: Language model
        dataloader: DataLoader with evaluation data
        tokenizer: Optional tokenizer
        config: Coherence configuration
        device: Computation device
        max_batches: Maximum batches to evaluate

    Returns:
        Aggregated CoherenceMetrics
    """
    measurer = CoherenceMeasurer(model, tokenizer, config, device)

    # Aggregate metrics
    total_ppl = 0.0
    total_rep = 0.0
    total_flow = 0.0
    total_topic = 0.0
    total_samples = 0
    total_seq_len = 0.0
    total_unique = 0.0
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            # Get input_ids from batch
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('inputs'))
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            else:
                input_ids = batch

            if input_ids is None:
                continue

            # Move to device
            if device is not None:
                input_ids = input_ids.to(device)

            # Measure this batch
            metrics = measurer.measure(input_ids=input_ids)

            # Skip batches with invalid perplexity (defense-in-depth)
            if math.isinf(metrics.perplexity) or math.isnan(metrics.perplexity):
                logger.warning(f"Batch {batch_idx} returned invalid perplexity, skipping")
                continue

            # Clamp perplexity to max for safety
            clamped_ppl = min(metrics.perplexity, config.max_perplexity if config else 100.0)

            # Accumulate
            total_ppl += clamped_ppl * metrics.num_samples
            total_rep += metrics.repetition_score * metrics.num_samples
            total_flow += metrics.sentence_flow_score * metrics.num_samples
            total_topic += metrics.topic_consistency * metrics.num_samples
            total_samples += metrics.num_samples
            total_seq_len += metrics.avg_sequence_length * metrics.num_samples
            total_unique += metrics.unique_token_ratio * metrics.num_samples
            num_batches += 1

    if total_samples == 0:
        return CoherenceMetrics()

    # Compute averages
    avg_ppl = total_ppl / total_samples
    avg_rep = total_rep / total_samples
    avg_flow = total_flow / total_samples
    avg_topic = total_topic / total_samples

    # Recompute aggregate score
    config = config or CoherenceConfig()
    # Use log-scale normalization (consistent with CoherenceMeasurer)
    if avg_ppl >= config.max_perplexity:
        ppl_normalized = 0.0
    elif avg_ppl <= 1.0:
        ppl_normalized = 1.0
    else:
        log_ppl = math.log(avg_ppl)
        log_max = math.log(config.max_perplexity)
        ppl_normalized = 1.0 - (log_ppl / log_max)
    rep_inverted = 1.0 - avg_rep

    # FIX: Normalize by weight sum to ensure coherence_score is in [0, 1]
    weight_sum = (
        config.perplexity_weight +
        config.repetition_weight +
        config.flow_weight +
        config.topic_weight
    )

    coherence_score = (
        config.perplexity_weight * ppl_normalized +
        config.repetition_weight * rep_inverted +
        config.flow_weight * avg_flow +
        config.topic_weight * avg_topic
    ) / max(weight_sum, 1e-8)

    return CoherenceMetrics(
        perplexity=avg_ppl,
        repetition_score=avg_rep,
        sentence_flow_score=avg_flow,
        topic_consistency=avg_topic,
        coherence_score=coherence_score,
        num_samples=total_samples,
        avg_sequence_length=total_seq_len / total_samples,
        unique_token_ratio=total_unique / total_samples,
    )


# Re-export CoherenceConfig for convenience
__all__ = [
    'CoherenceConfig',
    'CoherenceMetrics',
    'CoherenceMeasurer',
    'FastCoherenceMeasurer',
    'measure_coherence',
    'measure_coherence_fast',
    'measure_batch_coherence',
    'DEFAULT_MICRO_BATCH_SIZE',
]
