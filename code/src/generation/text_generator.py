"""
Text generation with MoE++ models
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Dict, Any
from transformers import PreTrainedTokenizerBase
import warnings


class TextGenerator:
    """Text generation interface for MoE++ models"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[str] = None
    ):
        """
        Initialize text generator
        
        Args:
            model: MoE++ model
            tokenizer: Tokenizer for encoding/decoding
            device: Device to use (auto-detect if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 256,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        early_stopping: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s)
        
        Args:
            prompt: Input prompt(s)
            max_length: Maximum generation length
            min_length: Minimum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            repetition_penalty: Penalty for repetition
            do_sample: Whether to sample or use greedy decoding
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early when all beams finish
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated text(s)
        """
        # Handle single vs batch input
        if isinstance(prompt, str):
            prompts = [prompt]
            single_input = True
        else:
            prompts = prompt
            single_input = False
        
        # Encode prompts
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get pad and eos tokens
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Generate
        if num_beams > 1:
            output_ids = self._beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                early_stopping=early_stopping
            )
        else:
            output_ids = self._sample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(output_ids):
            # Remove input tokens
            generated_tokens = output[len(input_ids[i]):]
            # Decode
            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts.append(text)
        
        # Return single string if single input
        if single_input:
            return generated_texts[0]
        return generated_texts
    
    def _sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        min_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int
    ) -> torch.Tensor:
        """Sample generation"""
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Keep track of which sequences are done
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # Auto-regressive generation
        while cur_len < max_length:
            # Get model outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            
            # Get logits
            if isinstance(outputs, dict):
                next_token_logits = outputs["logits"][:, -1, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(
                    next_token_logits,
                    input_ids,
                    repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k and top-p filtering
            if do_sample:
                next_token_logits = self._top_k_top_p_filtering(
                    next_token_logits,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Handle numerical issues
                probs = torch.nan_to_num(probs, nan=1e-8, posinf=1.0, neginf=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Ensure valid probabilities
                if torch.any(probs < 0) or torch.any(torch.isnan(probs)):
                    # Fallback to uniform distribution
                    probs = torch.ones_like(probs) / probs.shape[-1]
                
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Force min length
            if cur_len < min_length and eos_token_id is not None:
                next_tokens = next_tokens.masked_fill(
                    next_tokens == eos_token_id,
                    pad_token_id
                )
            
            # Update finished sequences
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    (next_tokens != eos_token_id).long()
                )
            
            # Pad finished sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                unfinished_sequences.unsqueeze(-1)
            ], dim=-1)
            
            cur_len += 1
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
        min_length: int,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        pad_token_id: int,
        eos_token_id: int,
        early_stopping: bool
    ) -> torch.Tensor:
        """Beam search generation"""
        # Simplified beam search - for full implementation, 
        # consider using transformers.generation_utils
        warnings.warn("Beam search not fully implemented, falling back to sampling")
        return self._sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ):
        """Apply repetition penalty to logits"""
        for i in range(input_ids.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                # If score < 0, apply penalty by multiplying
                if logits[i, previous_token] < 0:
                    logits[i, previous_token] *= penalty
                # If score > 0, apply penalty by dividing
                else:
                    logits[i, previous_token] /= penalty
    
    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """Filter logits using top-k and/or top-p filtering"""
        batch_size = logits.shape[0]
        
        # Top-k filtering
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of top-k
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        # Top-p filtering
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def stream_generate(
        self,
        prompt: str,
        max_length: int = 256,
        **kwargs
    ):
        """
        Stream generation token by token
        
        Yields:
            Generated tokens as they are produced
        """
        # Encode prompt
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        cur_len = input_ids.shape[1]
        
        # Generate token by token
        while cur_len < max_length:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if isinstance(outputs, dict):
                    next_token_logits = outputs["logits"][:, -1, :]
                else:
                    next_token_logits = outputs.logits[:, -1, :]
                
                # Simple sampling
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Decode and yield
                token_text = self.tokenizer.decode(next_token)
                yield token_text
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append token
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device)
                ], dim=-1)
                
                cur_len += 1