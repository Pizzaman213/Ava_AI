"""
Coherence Measurement Module for LLM Training.

Measures text coherence through multiple metrics:
- Perplexity-based coherence (lower perplexity = more coherent)
- Repetition penalty (excessive repetition indicates incoherence)
- Sentence flow consistency (adjacent sentence similarity)
- Topic consistency (semantic drift detection)

These metrics are computed during training evaluation to track
model quality over time.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CoherenceConfig from central config location (avoid duplication)
from ..config.training_config import CoherenceConfig

logger = logging.getLogger(__name__)


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

        # Compute individual metrics
        perplexity = self._compute_perplexity(input_ids)
        repetition_score = self._compute_repetition_score(input_ids)
        flow_score = self._compute_sentence_flow(input_ids)
        topic_score = self._compute_topic_consistency(input_ids)

        # Compute unique token ratio
        unique_ratio = self._compute_unique_ratio(input_ids)

        # Compute aggregate score
        # Normalize perplexity to 0-1 (lower perplexity = higher score)
        ppl_normalized = 1.0 - min(perplexity / self.config.max_perplexity, 1.0)

        # Invert repetition (lower repetition = higher score)
        rep_inverted = 1.0 - repetition_score

        coherence_score = (
            self.config.perplexity_weight * ppl_normalized +
            self.config.repetition_weight * rep_inverted +
            self.config.flow_weight * flow_score +
            self.config.topic_weight * topic_score
        )

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
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).all():
                    break

        return generated

    def _compute_perplexity(self, input_ids: torch.Tensor) -> float:
        """Compute perplexity on input sequences."""
        if input_ids.shape[1] < 2:
            return float('inf')

        # Shift for causal LM loss
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        # Forward pass
        outputs = self.model(inputs)

        # Get logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )

        # GPU SYNC FIX: Compute perplexity on GPU, then transfer single value
        # This avoids separate .item() call followed by math.exp on CPU
        # torch.exp and clamp run on GPU, single .item() at end
        perplexity = torch.exp(torch.clamp(loss, max=20.0)).item()

        return perplexity

    def _compute_repetition_score(self, input_ids: torch.Tensor) -> float:
        """
        Compute repetition score based on n-gram repetition.

        Returns:
            Score 0-1 where higher means more repetition (bad)
        """
        total_rep_ratio = 0.0
        num_valid = 0

        for seq in input_ids:
            seq_list = seq.tolist()

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
        if input_ids.shape[1] < 10:
            return 0.5  # Not enough context

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

            if hidden_states is None:
                # Model doesn't output hidden states, use logits similarity instead
                return self._compute_logit_flow(input_ids)

            # Ensure we have 3D tensor [batch, seq, hidden]
            if hidden_states.dim() != 3:
                return self._compute_logit_flow(input_ids)

            # Compute cosine similarity between adjacent positions
            # Split sequence into chunks and compare
            chunk_size = max(input_ids.shape[1] // 4, 2)
            similarities = []

            for i in range(0, input_ids.shape[1] - chunk_size, chunk_size):
                end_chunk2 = min(i + 2 * chunk_size, hidden_states.shape[1])
                if i + chunk_size >= end_chunk2:
                    break

                chunk1 = hidden_states[:, i:i+chunk_size, :].mean(dim=1)
                chunk2 = hidden_states[:, i+chunk_size:end_chunk2, :].mean(dim=1)

                # Cosine similarity
                sim = F.cosine_similarity(chunk1, chunk2, dim=-1)
                similarities.append(sim.mean())  # GPU SYNC FIX: Keep on GPU

            if len(similarities) == 0:
                return 0.5

            # GPU SYNC FIX: Stack on GPU, single sync at the end
            avg_sim = torch.stack(similarities).mean().item()

            # Map from [-1, 1] to [0, 1]
            return (avg_sim + 1) / 2

        except Exception as e:
            logger.warning(f"Sentence flow computation failed: {e}, using fallback")
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

        # GPU SYNC FIX: Compute all KL divergences on GPU, single sync at end
        # Compare adjacent token probability distributions
        if probs.shape[1] < 2:
            return 0.5

        # Compute all adjacent KL divergences at once on GPU
        p1 = probs[:, :-1, :]  # [batch, seq-1, vocab]
        p2 = probs[:, 1:, :]   # [batch, seq-1, vocab]

        # KL divergence per position (batch mean over batch dimension)
        # Using sum reduction over vocab, then mean over batch
        kl_divs = (p2 * (p2.log() - p1.log())).sum(dim=-1).mean(dim=0)  # [seq-1]

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
        if input_ids.shape[1] < 20:
            return 0.5  # Not enough context

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

            if hidden_states is None or hidden_states.dim() != 3:
                return 0.5  # Can't compute without hidden states

            # Compare first quarter vs last quarter
            quarter = input_ids.shape[1] // 4

            start_repr = hidden_states[:, :quarter, :].mean(dim=1)
            end_repr = hidden_states[:, -quarter:, :].mean(dim=1)

            # Cosine similarity
            sim = F.cosine_similarity(start_repr, end_repr, dim=-1)

            # Map from [-1, 1] to [0, 1]
            return ((sim.mean().item() + 1) / 2)

        except Exception as e:
            logger.warning(f"Topic consistency computation failed: {e}")
            return 0.5

    def _compute_unique_ratio(self, input_ids: torch.Tensor) -> float:
        """Compute ratio of unique tokens in sequences."""
        total_unique = 0
        total_tokens = 0

        for seq in input_ids:
            seq_list = seq.tolist()
            total_unique += len(set(seq_list))
            total_tokens += len(seq_list)

        if total_tokens == 0:
            return 0.0

        return total_unique / total_tokens


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

            # Accumulate
            total_ppl += metrics.perplexity * metrics.num_samples
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
    ppl_normalized = 1.0 - min(avg_ppl / config.max_perplexity, 1.0)
    rep_inverted = 1.0 - avg_rep

    coherence_score = (
        config.perplexity_weight * ppl_normalized +
        config.repetition_weight * rep_inverted +
        config.flow_weight * avg_flow +
        config.topic_weight * avg_topic
    )

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
