"""
Text Generation with Advanced Features
Implements generation with MoE++, speculative decoding, and RAG
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
import logging
import time
from pathlib import Path
from transformers import PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
from .continuous_batching import ContinuousBatchingEngine, InferenceRequest, create_continuous_batching_engine

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    # Basic settings
    max_length: int = 512
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    
    # Sampling parameters
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Advanced sampling
    use_contrastive_search: bool = False
    penalty_alpha: float = 0.6
    
    # Beam search
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    early_stopping: bool = False
    
    # Generation strategy
    use_cache: bool = True
    use_speculative_decoding: bool = True
    use_rag: bool = False
    use_chain_of_thought: bool = False
    
    # Output settings
    num_return_sequences: int = 1
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    return_dict_in_generate: bool = True
    
    # Stopping criteria
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    # Performance
    use_flash_attention: bool = True
    batch_size: int = 1
    use_continuous_batching: bool = False
    continuous_batching_config: Optional[Dict[str, Any]] = None

class TextGenerator:
    """Advanced text generator with multiple strategies"""
    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        config: GenerationConfig,
        speculative_decoder: Optional[Any] = None,
        rag_module: Optional[Any] = None,
        continuous_batching_engine: Optional[ContinuousBatchingEngine] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.speculative_decoder = speculative_decoder
        self.rag_module = rag_module
        self.continuous_batching_engine = continuous_batching_engine
        
        # Device
        self.device = next(model.parameters()).device
        
        # Set token IDs from tokenizer if not provided
        if config.eos_token_id is None:
            config.eos_token_id = tokenizer.eos_token_id
        if config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        if config.bos_token_id is None:
            config.bos_token_id = tokenizer.bos_token_id
        
        # Initialize continuous batching if configured
        if config.use_continuous_batching and not continuous_batching_engine:
            self.continuous_batching_engine = create_continuous_batching_engine(
                model, tokenizer, config.continuous_batching_config
            )
    
    def generate(
        self,
        inputs: Union[str, List[str], torch.Tensor],
        generation_config: Optional[GenerationConfig] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable] = None,
        logits_processor: Optional[Any] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate text with advanced features
        
        Args:
            inputs: Input text(s) or token IDs
            generation_config: Override default config
            stopping_criteria: Custom stopping criteria
            prefix_allowed_tokens_fn: Function to constrain generation
            logits_processor: Custom logits processors
            **kwargs: Additional generation arguments
            
        Returns:
            Generated sequences or full generation output
        """
        # Use provided config or default
        config = generation_config or self.config
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Prepare inputs
        if isinstance(inputs, str):
            inputs = [inputs]
        
        if isinstance(inputs, list):
            # Tokenize
            encoded = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(self.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            # Assume tensor input
            input_ids = inputs.to(self.device)
            attention_mask = torch.ones_like(input_ids)
        
        # Apply RAG if enabled
        if config.use_rag and self.rag_module:
            input_ids, attention_mask = self._apply_rag(
                inputs if isinstance(inputs, list) else None,
                input_ids,
                attention_mask,
            )
        
        # Apply chain-of-thought if enabled
        if config.use_chain_of_thought:
            input_ids, attention_mask = self._apply_chain_of_thought(
                input_ids,
                attention_mask,
            )
        
        # Generate based on strategy
        if config.use_speculative_decoding and self.speculative_decoder:
            outputs = self._generate_with_speculative_decoding(
                input_ids,
                attention_mask,
                config,
                stopping_criteria,
            )
        elif config.num_beams > 1:
            outputs = self._beam_search(
                input_ids,
                attention_mask,
                config,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                logits_processor,
            )
        elif config.do_sample:
            outputs = self._sample(
                input_ids,
                attention_mask,
                config,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                logits_processor,
            )
        else:
            outputs = self._greedy_search(
                input_ids,
                attention_mask,
                config,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                logits_processor,
            )
        
        return outputs
    
    def _apply_rag(
        self,
        texts: Optional[List[str]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RAG to augment input"""
        if texts is None:
            # Decode input_ids to get texts
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Retrieve relevant documents
        retrieval_results = self.rag_module.retrieve(texts, k=5)
        
        # Format retrieved context
        contexts = self.rag_module.format_retrieved_context(
            retrieval_results["documents"]
        )
        
        # Prepend context to input
        augmented_texts = []
        for text, context in zip(texts, contexts):
            augmented_text = f"Context: {context}\n\nQuery: {text}\n\nResponse:"
            augmented_texts.append(augmented_text)
        
        # Re-tokenize
        encoded = self.tokenizer(
            augmented_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)
        
        return encoded["input_ids"], encoded["attention_mask"]
    
    def _apply_chain_of_thought(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply chain-of-thought prompting"""
        # Add CoT prompt
        cot_prompt = "Let's think step by step:\n"
        cot_tokens = self.tokenizer(
            cot_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        
        # Concatenate
        input_ids = torch.cat([input_ids, cot_tokens["input_ids"]], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(cot_tokens["input_ids"])
        ], dim=1)
        
        return input_ids, attention_mask
    
    def _generate_with_speculative_decoding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
    ) -> Dict[str, Any]:
        """Generate using speculative decoding"""
        logger.info("Using speculative decoding for generation")
        
        # Use speculative decoder
        outputs = self.speculative_decoder.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens or config.max_length - input_ids.shape[1],
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            return_stats=True,
        )
        
        return outputs
    
    def _sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        prefix_allowed_tokens_fn: Optional[Callable],
        logits_processor: Optional[Any],
    ) -> Dict[str, Any]:
        """Sample-based generation"""
        batch_size = input_ids.shape[0]
        max_length = config.max_length
        
        # Initialize
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        past_key_values = None
        scores = [] if config.output_scores else None
        
        # Generation loop
        while input_ids.shape[1] < max_length:
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                    return_dict=True,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Apply logits processors
            if logits_processor:
                logits = logits_processor(input_ids, logits)
            
            # Apply prefix constraints
            if prefix_allowed_tokens_fn:
                logits = self._apply_prefix_constraints(
                    input_ids, logits, prefix_allowed_tokens_fn
                )
            
            # Apply temperature
            if config.temperature > 0:
                logits = logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                logits = self._top_k_filtering(logits, config.top_k)
            
            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                logits = self._top_p_filtering(logits, config.top_p)
            
            # Apply typical-p filtering
            if config.typical_p < 1.0:
                logits = self._typical_p_filtering(logits, config.typical_p)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    input_ids, logits, config.repetition_penalty
                )
            
            # Update sequences
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                unfinished_sequences.unsqueeze(1)
            ], dim=1)
            
            # Update finished sequences
            if config.eos_token_id is not None:
                if isinstance(config.eos_token_id, int):
                    eos_token_id = [config.eos_token_id]
                else:
                    eos_token_id = config.eos_token_id
                
                for eos_id in eos_token_id:
                    unfinished_sequences = unfinished_sequences * (next_tokens != eos_id)
            
            # Store scores
            if scores is not None:
                scores.append(logits)
            
            # Check stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, scores):
                break
            
            # Check if all sequences are finished
            if unfinished_sequences.sum() == 0:
                break
        
        # Prepare output
        if config.return_dict_in_generate:
            return {
                "sequences": input_ids,
                "scores": scores,
                "past_key_values": past_key_values,
            }
        else:
            return input_ids
    
    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        prefix_allowed_tokens_fn: Optional[Callable],
        logits_processor: Optional[Any],
    ) -> Dict[str, Any]:
        """Greedy decoding"""
        # Similar to sampling but always takes argmax
        batch_size = input_ids.shape[0]
        max_length = config.max_length
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        past_key_values = None
        
        while input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                    return_dict=True,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Apply processors
            if logits_processor:
                logits = logits_processor(input_ids, logits)
            
            # Greedy selection
            next_tokens = logits.argmax(dim=-1)
            
            # Update sequences
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                unfinished_sequences.unsqueeze(1)
            ], dim=1)
            
            # Check EOS
            if config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences * (next_tokens != config.eos_token_id)
            
            if unfinished_sequences.sum() == 0:
                break
        
        if config.return_dict_in_generate:
            return {"sequences": input_ids}
        else:
            return input_ids
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        prefix_allowed_tokens_fn: Optional[Callable],
        logits_processor: Optional[Any],
    ) -> Dict[str, Any]:
        """Beam search generation"""
        # This is a simplified implementation
        # Full implementation would include beam scoring, diversity penalties, etc.
        
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        max_length = config.max_length
        
        # Expand inputs for beam search
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)
        
        # Initialize beams
        beam_scores = torch.zeros(batch_size * num_beams, device=self.device)
        beam_scores[1:num_beams] = -1e9  # Only first beam is active initially
        
        past_key_values = None
        
        for step in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                    return_dict=True,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add beam scores
            log_probs = log_probs + beam_scores.unsqueeze(1)
            
            # Reshape for beam search
            vocab_size = log_probs.shape[-1]
            log_probs = log_probs.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams candidates
            next_scores, next_tokens = torch.topk(
                log_probs, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Compute next beam indices
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Select top num_beams
            beam_scores = next_scores[:, :num_beams].flatten()
            beam_tokens = next_tokens[:, :num_beams].flatten()
            beam_indices = next_indices[:, :num_beams].flatten()
            
            # Update sequences
            input_ids = torch.cat([
                input_ids[beam_indices],
                beam_tokens.unsqueeze(1)
            ], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask[beam_indices],
                torch.ones(batch_size * num_beams, 1, device=self.device)
            ], dim=1)
        
        # Select best sequences
        best_sequences = []
        for i in range(batch_size):
            batch_beam_scores = beam_scores[i * num_beams:(i + 1) * num_beams]
            best_idx = batch_beam_scores.argmax()
            best_sequence = input_ids[i * num_beams + best_idx]
            best_sequences.append(best_sequence)
        
        sequences = torch.stack(best_sequences)
        
        if config.return_dict_in_generate:
            return {"sequences": sequences}
        else:
            return sequences
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering"""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _typical_p_filtering(self, logits: torch.Tensor, typical_p: float) -> torch.Tensor:
        """Apply typical-p filtering (locally typical sampling)"""
        # Compute entropy
        normalized = F.log_softmax(logits, dim=-1)
        p = normalized.exp()
        ent = -(normalized * p).sum(-1, keepdim=True)
        
        # Compute deviation from entropy
        deviation = (normalized + ent).abs()
        
        # Sort by deviation
        _, sorted_indices = torch.sort(deviation)
        sorted_logits = logits.gather(-1, sorted_indices)
        sorted_p = p.gather(-1, sorted_indices)
        cumulative_probs = sorted_p.cumsum(dim=-1)
        
        # Remove tokens
        sorted_indices_to_remove = cumulative_probs > typical_p
        sorted_indices_to_remove[..., 0] = False  # Keep at least one
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty"""
        for i in range(input_ids.shape[0]):
            for previous_token in set(input_ids[i].tolist()):
                if logits[i, previous_token] < 0:
                    logits[i, previous_token] *= penalty
                else:
                    logits[i, previous_token] /= penalty
        
        return logits
    
    def _apply_prefix_constraints(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        prefix_allowed_tokens_fn: Callable,
    ) -> torch.Tensor:
        """Apply prefix constraints to logits"""
        for i in range(input_ids.shape[0]):
            allowed_tokens = prefix_allowed_tokens_fn(i, input_ids[i].tolist())
            if allowed_tokens is not None:
                mask = torch.ones_like(logits[i]) * -float('inf')
                mask[allowed_tokens] = 0
                logits[i] = logits[i] + mask
        
        return logits
    
    def generate_with_continuous_batching(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        priority: Optional[List[int]] = None,
        streaming_callbacks: Optional[List[Callable[[str], None]]] = None,
        completion_callbacks: Optional[List[Callable[[str], None]]] = None,
    ) -> List[str]:
        """
        Generate text using continuous batching for multiple prompts
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            priority: Priority for each request (higher = processed first)
            streaming_callbacks: Callbacks for streaming tokens
            completion_callbacks: Callbacks for completed generations
            
        Returns:
            List of generated texts
        """
        if not self.continuous_batching_engine:
            raise ValueError("Continuous batching engine not initialized")
        
        # Create requests
        requests = []
        for i, prompt in enumerate(prompts):
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
            
            request = InferenceRequest(
                request_id=f"req_{i}_{time.time()}",
                prompt_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                priority=priority[i] if priority else 0,
                streaming_callback=streaming_callbacks[i] if streaming_callbacks else None,
                completion_callback=completion_callbacks[i] if completion_callbacks else None,
            )
            requests.append(request)
        
        # Submit requests
        request_ids = []
        for request in requests:
            request_id = self.continuous_batching_engine.submit_request(request)
            request_ids.append(request_id)
        
        # Wait for completions (in a real system, this would be async)
        # For now, we'll use a simple polling approach
        completed_texts = {}
        
        def completion_handler(request_id: str):
            def handler(text: str):
                completed_texts[request_id] = text
            return handler
        
        # Update requests with completion handlers
        for i, request in enumerate(requests):
            request.completion_callback = completion_handler(request.request_id)
            self.continuous_batching_engine.submit_request(request)
        
        # Wait for all completions
        while len(completed_texts) < len(requests):
            time.sleep(0.1)
        
        # Return in order
        return [completed_texts[req.request_id] for req in requests]

def create_text_generator(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    generation_config: Optional[GenerationConfig] = None,
    use_speculative_decoding: bool = True,
    use_rag: bool = False,
    device: Optional[str] = None,
) -> TextGenerator:
    """
    Create text generator with specified features
    
    Args:
        model_path: Path to model
        tokenizer_path: Path to tokenizer (uses model_path if None)
        generation_config: Generation configuration
        use_speculative_decoding: Whether to use speculative decoding
        use_rag: Whether to use RAG
        device: Device to use
        
    Returns:
        TextGenerator instance
    """
    from transformers import AutoTokenizer
    from ..model.moe_transformer import MoEForCausalLM
    
    # Load model
    model = MoEForCausalLM.from_pretrained(model_path)
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Create config
    config = generation_config or GenerationConfig()
    
    # Create speculative decoder if needed
    speculative_decoder = None
    if use_speculative_decoding:
        from ..model.speculative import SpeculativeDecoder, DraftModel
        draft_model = DraftModel(model.config.vocab_size, config)
        speculative_decoder = SpeculativeDecoder(model, draft_model)
    
    # Create RAG module if needed
    rag_module = None
    if use_rag:
        from ..model.rag_module import RAGModule, RAGConfig
        rag_config = RAGConfig()
        rag_module = RAGModule(rag_config, model.config)
    
    # Create generator
    generator = TextGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        speculative_decoder=speculative_decoder,
        rag_module=rag_module,
    )
    
    return generator