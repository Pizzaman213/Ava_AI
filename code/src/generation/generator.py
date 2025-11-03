"""
Text generation utilities for Qwen MoE++ models.

This module provides advanced text generation capabilities including:
- Multiple decoding strategies (greedy, sampling, beam search)
- Temperature-based sampling with top-k and nucleus (top-p) filtering
- Repetition penalty and length penalty
- Batch generation support

The generator is designed to work with models trained on data from
/project/code/data/pretraining/processed/ and provides high-quality
text generation with various control parameters.
"""

import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Optional, List, Tuple, Union, TYPE_CHECKING
import numpy as np
from transformers import AutoTokenizer  # type: ignore[import]

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class TextGenerator:
    """
    Advanced text generator for Qwen MoE++ models.

    This generator implements multiple decoding strategies and provides
    fine-grained control over the generation process through various
    parameters like temperature, top-k, and top-p.

    Args:
        model: Trained Qwen MoE++ model
        tokenizer: Tokenizer for encoding/decoding text
        device (str): Device to run generation on

    Example:
        >>> generator = TextGenerator(model, tokenizer)
        >>> text = generator.generate("Once upon a time", max_length=100, temperature=0.8)
    """

    def __init__(self, model, tokenizer: "PreTrainedTokenizerBase", device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer: "PreTrainedTokenizerBase" = tokenizer
        self.device = device or next(model.parameters()).device

        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int]],
        max_length: int = 100,
        min_length: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_penalty: float = 1.0,
        length_penalty: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = True,
        early_stopping: bool = True,
        num_return_sequences: int = 1,
        use_ngram_blocking: bool = False,
        ngram_size: int = 3
    ) -> Union[str, List[str]]:
        """
        Generate text from a prompt using specified decoding strategy.

        Args:
            prompt (str or List[int]): Input prompt as text or token IDs
            max_length (int): Maximum length of generated sequence
            min_length (int): Minimum length of generated sequence
            temperature (float): Sampling temperature (higher = more random)
            top_k (int): Number of highest probability tokens to keep for sampling
            top_p (float): Cumulative probability threshold for nucleus sampling
            repetition_penalty (float): Penalty for repeating tokens
            eos_penalty (float): Penalty multiplier for EOS token (>1.0 discourages EOS)
            length_penalty (float): Exponential penalty to length (for beam search)
            num_beams (int): Number of beams for beam search (1 = no beam search)
            do_sample (bool): Whether to use sampling (vs greedy/beam search)
            early_stopping (bool): Whether to stop when all beams hit EOS
            num_return_sequences (int): Number of sequences to return
            use_ngram_blocking (bool): Block repeated n-grams to prevent repetition
            ngram_size (int): Size of n-grams to block (default: 3)

        Returns:
            Generated text(s) as string or list of strings
        """
        # Encode prompt
        if isinstance(prompt, str):
            encoded = self.tokenizer.encode(prompt, return_tensors='pt')
            input_ids = torch.tensor(encoded) if not isinstance(encoded, torch.Tensor) else encoded
        else:
            input_ids = torch.tensor([prompt]) if not isinstance(prompt, torch.Tensor) else prompt

        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]

        # Choose generation method
        if num_beams > 1:
            outputs = self._beam_search(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_penalty=eos_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                use_ngram_blocking=use_ngram_blocking,
                ngram_size=ngram_size
            )
        else:
            outputs = self._generate_no_beam(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_penalty=eos_penalty,
                do_sample=do_sample,
                use_ngram_blocking=use_ngram_blocking,
                ngram_size=ngram_size
            )

        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove input prompt from generated sequence
            generated_tokens = output[input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts[0] if num_return_sequences == 1 else generated_texts

    def _generate_no_beam(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        min_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_penalty: float,
        do_sample: bool,
        use_ngram_blocking: bool = False,
        ngram_size: int = 3
    ) -> List[torch.Tensor]:
        """
        Generate text without beam search (greedy or sampling).

        This method generates text token by token, either greedily selecting
        the most probable token or sampling based on the probability distribution.
        """
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]

        # Track generated sequences
        generated = input_ids.clone()
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)

        while cur_len < max_length:
            # Get model predictions
            outputs = self.model(generated)
            next_token_logits = outputs['logits'][:, -1, :]

            # Apply EOS penalty
            if eos_penalty != 1.0:
                self._apply_eos_penalty(next_token_logits, eos_penalty)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(
                    next_token_logits,
                    generated,
                    repetition_penalty
                )

            # Apply n-gram blocking
            if use_ngram_blocking:
                self._apply_ngram_blocking(
                    next_token_logits,
                    generated,
                    ngram_size
                )

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k and top-p filtering
            if do_sample:
                filtered_logits = self._top_k_top_p_filtering(
                    next_token_logits,
                    top_k=top_k,
                    top_p=top_p
                )
                probs = F.softmax(filtered_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Force minimum length
            if cur_len < min_length:
                # Create condition tensor for torch.where
                eos_mask = (next_tokens == self.eos_token_id)
                next_tokens = torch.where(
                    eos_mask,  # type: ignore[arg-type]
                    torch.tensor(self.pad_token_id, device=next_tokens.device, dtype=next_tokens.dtype),
                    next_tokens
                )

            # Update generated sequence
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            cur_len += 1

            # Check for completion
            not_eos_mask = (next_tokens != self.eos_token_id).long()  # type: ignore[attr-defined]
            unfinished_sequences = unfinished_sequences * not_eos_mask
            if unfinished_sequences.sum() == 0:
                break

        return [generated[i] for i in range(batch_size)]

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        min_length: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_penalty: float,
        length_penalty: float,
        early_stopping: bool,
        num_return_sequences: int,
        do_sample: bool,
        use_ngram_blocking: bool = False,
        ngram_size: int = 3
    ) -> List[torch.Tensor]:
        """
        Generate text using beam search.

        Beam search maintains multiple hypotheses (beams) and explores the most
        promising sequences in parallel, leading to higher quality outputs.
        """
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]

        # Expand input for beam search
        expanded_input = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        expanded_input = expanded_input.contiguous().view(batch_size * num_beams, -1)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially

        # Track generated sequences
        generated = expanded_input.clone()
        done = [False] * batch_size

        while cur_len < max_length:
            # Get model predictions
            outputs = self.model(generated)
            next_token_logits = outputs['logits'][:, -1, :]

            # Apply EOS penalty
            if eos_penalty != 1.0:
                self._apply_eos_penalty(next_token_logits, eos_penalty)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(
                    next_token_logits,
                    generated,
                    repetition_penalty
                )

            # Apply n-gram blocking
            if use_ngram_blocking:
                self._apply_ngram_blocking(
                    next_token_logits,
                    generated,
                    ngram_size
                )

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Calculate scores
            scores = F.log_softmax(next_token_logits, dim=-1)
            scores = scores.view(batch_size, num_beams, -1)

            # Add beam scores
            next_scores = scores + beam_scores.unsqueeze(-1)

            # Reshape for beam selection
            next_scores = next_scores.view(batch_size, -1)

            # Sample or select top 2*num_beams tokens
            if do_sample:
                probs = F.softmax(next_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
                next_scores = torch.gather(next_scores, -1, next_tokens)
            else:
                next_scores, next_tokens = torch.topk(
                    next_scores, 2 * num_beams, dim=-1, largest=True, sorted=True
                )

            # Select best beams
            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # Pad the batch
                    next_batch_beam.extend([(0, self.pad_token_id, 0)] * num_beams)
                    continue

                next_sent_beam = []

                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.tokenizer.vocab_size
                    token_id = beam_token_id % self.tokenizer.vocab_size

                    effective_score = beam_token_score

                    # Apply length penalty
                    if length_penalty != 1.0:
                        effective_score = effective_score / (cur_len ** length_penalty)

                    next_sent_beam.append((effective_score, token_id, beam_id))

                    if len(next_sent_beam) >= num_beams:
                        break

                next_batch_beam.extend(next_sent_beam)

                # Check if we're done
                if next_sent_beam[0][1] == self.eos_token_id and cur_len >= min_length:
                    done[batch_idx] = True

            # Update beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = generated.new([x[1] for x in next_batch_beam])
            beam_idx = generated.new([x[2] for x in next_batch_beam])

            # Reorder generated sequences
            generated = generated[beam_idx, :]
            generated = torch.cat([generated, beam_tokens.unsqueeze(1)], dim=-1)

            cur_len += 1

            # Check early stopping
            if early_stopping and all(done):
                break

        # Select best sequences
        sequences = generated.view(batch_size, num_beams, -1)

        if num_return_sequences == 1:
            # Return best sequence for each batch
            return [sequences[i, 0] for i in range(batch_size)]
        else:
            # Return top num_return_sequences
            return [sequences[i, :num_return_sequences].reshape(-1, sequences.shape[-1])
                   for i in range(batch_size)]

    def _apply_eos_penalty(
        self,
        logits: torch.Tensor,
        penalty: float
    ):
        """Apply penalty to EOS token to encourage/discourage ending generation."""
        if self.eos_token_id is None:
            return
        # Handle various types that eos_token_id might be (str, list[str], or int)
        if isinstance(self.eos_token_id, (list, str)):
            # If it's a list or string, try to convert to int, otherwise return
            try:
                eos_id: int = int(self.eos_token_id[0] if isinstance(self.eos_token_id, list) else self.eos_token_id)
            except (ValueError, IndexError, TypeError):
                return
        else:
            eos_id: int = int(self.eos_token_id)
        for i in range(logits.shape[0]):
            if logits[i, eos_id] < 0:
                logits[i, eos_id] *= penalty
            else:
                logits[i, eos_id] /= penalty

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float
    ):
        """Apply repetition penalty to discourage repeating tokens."""
        for i in range(generated.shape[0]):
            for token_id in set(generated[i].tolist()):
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty

    def _apply_ngram_blocking(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        ngram_size: int
    ):
        """
        Block n-grams that would repeat earlier sequences.

        This prevents patterns like "time time time" or repeating phrases.
        """
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            gen_tokens = generated[i].tolist()

            # Need at least (ngram_size - 1) tokens to check
            if len(gen_tokens) < ngram_size - 1:
                continue

            # Get the context (last ngram_size - 1 tokens)
            context = gen_tokens[-(ngram_size - 1):]

            # For each possible next token, check if it would create a repeated n-gram
            for token_id in range(vocab_size):
                # Construct the potential n-gram
                potential_ngram = context + [token_id]

                # Search for this n-gram in the already generated sequence
                for j in range(len(gen_tokens) - ngram_size + 1):
                    if gen_tokens[j:j + ngram_size] == potential_ngram:
                        # This n-gram already exists - block it
                        logits[i, token_id] = float('-inf')
                        break

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """
        Filter a distribution using top-k and/or nucleus (top-p) filtering.

        Args:
            logits: Logits distribution shape (batch_size, vocab_size)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= top_p
            filter_value: Value to assign to filtered tokens

        Returns:
            Filtered logits
        """
        batch_size = logits.size(0)

        # Clone logits to avoid in-place modifications
        logits = logits.clone()

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        # Top-p filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False

            # Scatter sorted indices to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits