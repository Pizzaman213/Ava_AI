#!/usr/bin/env python3
"""
Enhanced Text Generation Script for Ava MoE Models

This script provides advanced text generation capabilities for models trained with the
comprehensive 8-phase enhanced training framework. It seamlessly integrates with the
run management system and supports multiple checkpoint formats and generation strategies.

 Key Features:

•  Smart Auto-Discovery: Automatically finds and loads the most recent trained model
•  Run Management Integration: Load checkpoints by run ID with full metadata
•  Multi-Format Support: Handles new framework, legacy, DeepSpeed, and raw formats
•  Checkpoint Selection: Choose between latest, best, or step-specific checkpoints
•  Multiple Generation Modes: Single prompt, interactive session, or batch processing
•   Advanced Sampling: Temperature, top-k, top-p, repetition penalty, beam search
•  Detailed Logging: Shows model info, training metrics, and generation parameters

 Quick Start:

    # Easiest: Auto-discover latest trained model
    python generate.py --prompt "Once upon a time"

    # List all available training runs
    python generate.py --list-runs

 Usage Examples:

    # Load from specific run (automatically uses latest checkpoint)
    python generate.py --run-id run_20250928_134034_3b295412 --prompt "The future of AI"

    # Use the best checkpoint from a specific run
    python generate.py --run-id run_20250928_134034_3b295412 \\
                       --checkpoint-type best \\
                       --prompt "Hello world"

    # Explicit checkpoint path (for legacy/custom checkpoints)
    python generate.py --model-path /path/to/checkpoint.pt --prompt "Once upon a time"

    # Interactive mode with custom sampling parameters
    python generate.py --interactive \\
                       --temperature 0.8 \\
                       --top-p 0.95 \\
                       --repetition-penalty 1.5

    # Batch generation from file
    python generate.py --input-file prompts.txt \\
                       --output-file responses.txt \\
                       --max-length 200

  Sampling Parameters:

  --temperature    Controls randomness (0.1=focused, 1.0=balanced, 2.0=creative)
  --top-p          Nucleus sampling threshold (0.9=default, 0.95=more diverse)
  --top-k          Limits vocabulary per step (50=default, higher=more options)
  --repetition-penalty  Reduces repetition (1.0=off, 1.2-2.0=recommended)
  --num-beams      Beam search width (1=greedy, 4-8=better quality)
  --max-length     Maximum tokens to generate (default: 100)

 Integration with Training:

This script is part of the comprehensive Ava training pipeline supporting:
   Phase 1-8: All training enhancements (stability, data pipeline, adaptive LR, etc.)
   Run Management: Organized checkpoint storage with full metadata
   Multi-Format: Backward compatible with all checkpoint formats
   Production Ready: Robust error handling and format detection
"""

import argparse
import logging
import sys
import warnings
import torch  # type: ignore[import-not-found]
import yaml
from pathlib import Path
from typing import Optional, List, Union
import json
import threading
import time
from tqdm import tqdm

# Suppress Pydantic field attribute warnings early (these come from dependencies)
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings('ignore', category=UnsupportedFieldAttributeWarning)
except ImportError:
    # Newer versions of Pydantic may not have this warning
    pass

# Add project root to path
sys.path.append('/root/Ava_AI/code')
sys.path.insert(0, '/root/Ava_AI/code/src')

from ava.models.moe import EnhancedMoEModel, EnhancedMoEConfig  # type: ignore[import-not-found]
from ava.core.logging import (
    Colors, Icons, supports_color,
    print_header, print_subheader, print_success, print_info,
    print_box, print_metric
)
from src.generation.generator import TextGenerator
from transformers import AutoTokenizer  # type: ignore[import-not-found]
from datetime import datetime


class Spinner:
    """Spinning indicator for long-running operations."""

    def __init__(self, message: str = ""):
        self.spinner_chars = ['|', '/', '-', '\\']
        self.message = message
        self.running = False
        self.thread = None
        self.idx = 0

    def _spin(self):
        while self.running:
            char = self.spinner_chars[self.idx % len(self.spinner_chars)]
            print(f'\r{self.message}{char}', end='', flush=True)
            self.idx += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        # Clear the spinner character
        print(f'\r{self.message}', end='', flush=True)


# ============================================================================
# Generation Display Helper Functions
# ============================================================================

def print_generation_header():
    """Print header for generation session."""
    if supports_color():
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  {Icons.BRAIN} Ava Text Generation{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")
    else:
        print("\n" + "=" * 70)
        print("  [MODEL] Ava Text Generation")
        print("=" * 70 + "\n")


def print_prompt(prompt_text: str, label: str = "Prompt"):
    """Print the input prompt with styling."""
    if supports_color():
        print(f"{Colors.BOLD}{Colors.LIGHT_BLUE}{Icons.ARROW_RIGHT} {label}:{Colors.RESET}")
        print(f"  {Colors.WHITE}{prompt_text}{Colors.RESET}")
    else:
        print(f"-> {label}:")
        print(f"  {prompt_text}")


def print_generated_text(text: str, label: str = "Generated"):
    """Print the generated text with styling."""
    if supports_color():
        print(f"\n{Colors.BOLD}{Colors.GREEN}{Icons.SUCCESS} {label}:{Colors.RESET}")
        print(f"  {Colors.LIME}{text}{Colors.RESET}")
    else:
        print(f"\n[OK] {label}:")
        print(f"  {text}")


def print_generation_separator():
    """Print a visual separator between generations."""
    if supports_color():
        print(f"\n{Colors.GRAY}{'-' * 50}{Colors.RESET}\n")
    else:
        print("\n" + "-" * 50 + "\n")


def print_generation_params(args):
    """Print generation parameters in a formatted way."""
    if supports_color():
        print(f"{Colors.DARK_CYAN}{Icons.GEAR} Generation Parameters:{Colors.RESET}")
        print(f"  {Colors.WHITE}Temperature:{Colors.RESET} {Colors.GOLD}{args.temperature}{Colors.RESET}")
        print(f"  {Colors.WHITE}Top-p:{Colors.RESET}       {Colors.GOLD}{args.top_p}{Colors.RESET}")
        print(f"  {Colors.WHITE}Top-k:{Colors.RESET}       {Colors.GOLD}{args.top_k}{Colors.RESET}")
        print(f"  {Colors.WHITE}Max Length:{Colors.RESET}  {Colors.GOLD}{args.max_length}{Colors.RESET}")
        if args.repetition_penalty != 1.0:
            print(f"  {Colors.WHITE}Rep. Penalty:{Colors.RESET} {Colors.GOLD}{args.repetition_penalty}{Colors.RESET}")
    else:
        print("[*] Generation Parameters:")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top-p:       {args.top_p}")
        print(f"  Top-k:       {args.top_k}")
        print(f"  Max Length:  {args.max_length}")
        if args.repetition_penalty != 1.0:
            print(f"  Rep. Penalty: {args.repetition_penalty}")


def print_batch_progress(current: int, total: int, prompt: str):
    """Print batch progress with truncated prompt."""
    truncated = prompt[:40] + "..." if len(prompt) > 40 else prompt
    if supports_color():
        print(f"{Colors.LIGHT_BLUE}[{current}/{total}]{Colors.RESET} {Colors.GRAY}{truncated}{Colors.RESET}")
    else:
        print(f"[{current}/{total}] {truncated}")


def print_interactive_banner():
    """Print banner for interactive mode."""
    if supports_color():
        print(f"\n{Colors.BOLD}{Colors.PURPLE}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.PURPLE}  {Icons.BRAIN} Interactive Generation Mode (Raw Mode){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.PURPLE}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.CYAN}Enter prompts for text completion (type 'quit' to exit){Colors.RESET}")
        print(f"\n{Colors.GRAY}Commands:{Colors.RESET}")
        print(f"  {Colors.WHITE}/settings{Colors.RESET}          {Colors.GRAY}- show current settings{Colors.RESET}")
        print(f"  {Colors.WHITE}/set NAME VALUE{Colors.RESET}    {Colors.GRAY}- update parameter{Colors.RESET}")
        print(f"  {Colors.WHITE}/chat{Colors.RESET}              {Colors.GRAY}- toggle chat mode (OpenOrca){Colors.RESET}")
        print(f"  {Colors.WHITE}/system <msg>{Colors.RESET}      {Colors.GRAY}- change system prompt{Colors.RESET}")
        print(f"  {Colors.WHITE}/clear{Colors.RESET}             {Colors.GRAY}- clear conversation history{Colors.RESET}")
        print(f"\n{Colors.GRAY}Parameters: temperature, top_p, top_k, max_length, repetition_penalty{Colors.RESET}")
        print(f"{Colors.GRAY}Example: /set temperature 0.8{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.PURPLE}{'=' * 60}{Colors.RESET}\n")
    else:
        print("\n" + "=" * 60)
        print("  [MODEL] Interactive Generation Mode (Raw Mode)")
        print("=" * 60)
        print("Enter prompts for text completion (type 'quit' to exit)")
        print("\nCommands:")
        print("  /settings          - show current settings")
        print("  /set NAME VALUE    - update parameter")
        print("  /chat              - toggle chat mode (OpenOrca)")
        print("  /system <msg>      - change system prompt")
        print("  /clear             - clear conversation history")
        print("\nParameters: temperature, top_p, top_k, max_length, repetition_penalty")
        print("Example: /set temperature 0.8")
        print("=" * 60 + "\n")


def print_assistant_response(response: str):
    """Print assistant response in interactive mode."""
    if supports_color():
        print(f"{Colors.BOLD}{Colors.GREEN}Assistant:{Colors.RESET} {Colors.LIME}{response}{Colors.RESET}")
    else:
        print(f"Assistant: {response}")


def print_system_prompt_display(prompt: str):
    """Print the system prompt."""
    if supports_color():
        print(f"{Colors.MAGENTA}System:{Colors.RESET} {Colors.GRAY}{prompt}{Colors.RESET}\n")
    else:
        print(f"System: {prompt}\n")


def print_interactive_settings(settings: dict, system_prompt: str, use_openorca_format: bool):
    """Print interactive mode settings with colors."""
    if supports_color():
        print(f"\n{Colors.DARK_CYAN}{Icons.GEAR} Current settings:{Colors.RESET}")
        for k, v in settings.items():
            print(f"  {Colors.WHITE}{k}:{Colors.RESET} {Colors.GOLD}{v}{Colors.RESET}")
        print(f"  {Colors.WHITE}system_prompt:{Colors.RESET} {Colors.GRAY}{system_prompt}{Colors.RESET}")
        print(f"  {Colors.WHITE}openorca_format:{Colors.RESET} {Colors.GOLD}{use_openorca_format}{Colors.RESET}")
    else:
        print("\nCurrent settings:")
        for k, v in settings.items():
            print(f"  {k}: {v}")
        print(f"  system_prompt: {system_prompt}")
        print(f"  openorca_format: {use_openorca_format}")


def get_user_prompt_label(use_openorca_format: bool) -> str:
    """Get the colored user prompt label."""
    if use_openorca_format:
        if supports_color():
            return f"{Colors.BOLD}{Colors.CYAN}User:{Colors.RESET} "
        return "User: "
    else:
        if supports_color():
            return f"{Colors.BOLD}{Colors.CYAN}>{Colors.RESET} "
        return "> "


def find_latest_run(base_dir: str = '/root/Ava_AI/code/outputs/runs') -> Optional[Path]:
    """Find the most recent training run directory."""
    runs_path = Path(base_dir)
    if not runs_path.exists():
        return None

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None

    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def list_available_runs(base_dir: str = '/root/Ava_AI/code/outputs/runs') -> List[Path]:
    """List all available training run directories."""
    runs_path = Path(base_dir)
    if not runs_path.exists():
        return []

    run_dirs = [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs


def get_checkpoint_path_from_run(run_dir: Path, checkpoint_type: str = 'latest') -> Path:
    """
    Get checkpoint path from a run directory.

    Args:
        run_dir: Path to run directory
        checkpoint_type: 'latest', 'best', or 'step_N'

    Returns:
        Path to checkpoint file
    """
    # Check if we're already in a step_N directory
    if run_dir.name.startswith('step_'):
        # We're in a step checkpoint directory, just return model.pt
        model_path = run_dir / 'model.pt'
        if model_path.exists():
            return model_path

    checkpoints_dir = run_dir / 'checkpoints'

    if checkpoint_type == 'latest':
        return checkpoints_dir / 'latest_model.pt'
    elif checkpoint_type == 'best':
        return checkpoints_dir / 'best_model.pt'
    elif checkpoint_type.startswith('step_'):
        step_num = checkpoint_type.split('_')[1]
        return checkpoints_dir / f'step_{step_num}' / 'model.pt'
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")


def strip_torch_compile_prefix(state_dict: dict) -> dict:
    """
    Strip '_orig_mod.' prefix from state dict keys.

    When a model is saved after torch.compile(), the state dict keys get
    prefixed with '_orig_mod.'. This function removes that prefix so the
    weights can be loaded into a non-compiled model.

    Args:
        state_dict: Model state dictionary (potentially with _orig_mod. prefix)

    Returns:
        State dictionary with cleaned keys
    """
    fixed_state_dict = {}
    prefix = '_orig_mod.'
    had_prefix = False

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            fixed_state_dict[new_key] = value
            had_prefix = True
        else:
            fixed_state_dict[key] = value

    if had_prefix:
        print(f" Stripped torch.compile prefix from {len(fixed_state_dict)} keys")

    return fixed_state_dict


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = 'cpu',
) -> str:
    """
    Generate text using the same proven logic as training generation.

    This function properly handles:
    - BOS/EOS token handling
    - Attention mask and position IDs
    - Repetition penalty for both positive and negative logits
    - Top-k + top-p (nucleus) sampling

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top-k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        Generated text (excluding prompt)
    """
    model.eval()

    # Get special token IDs from model config or use BERT defaults
    model_config = getattr(model, 'config', None)
    bos_token_id = getattr(model_config, 'bos_token_id', 101) if model_config else 101
    eos_token_id = getattr(model_config, 'eos_token_id', 102) if model_config else 102

    # Encode prompt
    token_ids = tokenizer.encode(prompt)

    # NOTE: Removed EOS stripping - models trained with BERT tokenizer expect [SEP] at context end
    # The model learned to predict continuations after [SEP], not after regular tokens.
    # If generation produces garbage, the training format might expect EOS at context end.

    # Prepend BOS if not present
    if token_ids[0] != bos_token_id:
        token_ids = [bos_token_id] + token_ids

    prompt_len = len(token_ids)
    generated_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(generated_ids, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_length):
            # Create position IDs
            current_len = generated_ids.shape[1]
            position_ids = torch.arange(current_len, device=device).unsqueeze(0)

            # Forward pass with attention mask and position IDs
            outputs = model(generated_ids, attention_mask=attention_mask, position_ids=position_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Get next token logits and apply temperature
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty (handles both positive and negative logits)
            if repetition_penalty > 1.0 and generated_ids.shape[1] > 0:
                recent_tokens = generated_ids[0, -50:].tolist()
                for token_id_int in recent_tokens:
                    if next_logits[0, token_id_int] < 0:
                        next_logits[0, token_id_int] *= repetition_penalty
                    else:
                        next_logits[0, token_id_int] /= repetition_penalty

            # Compute probabilities
            probs = torch.softmax(next_logits, dim=-1)

            # Top-k filtering
            if top_k > 0 and top_k < probs.shape[-1]:
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)

                # Apply top-p within top-k
                sorted_probs, sort_idx = torch.sort(top_k_probs, descending=True, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs > top_p
                mask[..., 0] = False  # Always keep at least one token
                sorted_probs[mask] = 0.0

                # Normalize
                prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                if prob_sum < 1e-10:
                    sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.shape[-1]
                else:
                    sorted_probs = sorted_probs / (prob_sum + 1e-10)

                # Sample
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                local_idx = sort_idx.gather(-1, sampled_idx)
                next_token = top_k_indices.gather(-1, local_idx)
            else:
                # Just use top-p on full distribution
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs > top_p
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_idx)

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=-1)

            # Stop on EOS
            if next_token[0, 0].item() == eos_token_id:
                break

    # Decode only the generated part (excluding prompt)
    generated_tokens = generated_ids[0, prompt_len:].tolist()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def generate_text_stream(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = 'cpu',
):
    """
    Stream text generation token by token using proven training logic.

    Yields tokens as they are generated for real-time display.
    Handles WordPiece tokenization properly by decoding incrementally.

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Keep only top-k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        device: Device to run on

    Yields:
        str: Each generated token/word as text with proper spacing
    """
    model.eval()

    # Get special token IDs from model config or use BERT defaults
    model_config = getattr(model, 'config', None)
    bos_token_id = getattr(model_config, 'bos_token_id', 101) if model_config else 101
    eos_token_id = getattr(model_config, 'eos_token_id', 102) if model_config else 102

    # Encode prompt
    token_ids = tokenizer.encode(prompt)

    # NOTE: Removed EOS stripping - models trained with BERT tokenizer expect [SEP] at context end
    # The model learned to predict continuations after [SEP], not after regular tokens.

    # Prepend BOS if not present
    if token_ids[0] != bos_token_id:
        token_ids = [bos_token_id] + token_ids

    prompt_len = len(token_ids)
    generated_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(generated_ids, dtype=torch.long)

    # Track previously decoded text to yield only new content
    prev_decoded = ""

    with torch.no_grad():
        for _ in range(max_length):
            # Create position IDs
            current_len = generated_ids.shape[1]
            position_ids = torch.arange(current_len, device=device).unsqueeze(0)

            # Forward pass
            outputs = model(generated_ids, attention_mask=attention_mask, position_ids=position_ids)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Get next token logits and apply temperature
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty
            if repetition_penalty > 1.0 and generated_ids.shape[1] > 0:
                recent_tokens = generated_ids[0, -50:].tolist()
                for token_id_int in recent_tokens:
                    if next_logits[0, token_id_int] < 0:
                        next_logits[0, token_id_int] *= repetition_penalty
                    else:
                        next_logits[0, token_id_int] /= repetition_penalty

            # Compute probabilities
            probs = torch.softmax(next_logits, dim=-1)

            # Top-k + top-p sampling
            if top_k > 0 and top_k < probs.shape[-1]:
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
                sorted_probs, sort_idx = torch.sort(top_k_probs, descending=True, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs > top_p
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                if prob_sum < 1e-10:
                    sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.shape[-1]
                else:
                    sorted_probs = sorted_probs / (prob_sum + 1e-10)
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                local_idx = sort_idx.gather(-1, sampled_idx)
                next_token = top_k_indices.gather(-1, local_idx)
            else:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs > top_p
                mask[..., 0] = False
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-10)
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_idx)

            # Append token first
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=-1)

            # Decode full generated sequence and yield the new part
            # This handles WordPiece properly (spaces are added correctly)
            generated_tokens = generated_ids[0, prompt_len:].tolist()
            current_decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Yield only the new part
            if len(current_decoded) > len(prev_decoded):
                new_text = current_decoded[len(prev_decoded):]
                if new_text:
                    yield new_text
                prev_decoded = current_decoded

            # Stop on EOS
            if next_token[0, 0].item() == eos_token_id:
                break


class GenerationPipeline:
    """
    Complete generation pipeline for Qwen MoE++ models.

    This pipeline handles model loading, tokenization, and various generation
    strategies for producing high-quality text outputs.

    Args:
        model_path (str): Path to trained model checkpoint
        config_path (str, optional): Path to model configuration file
        device (str): Device to run inference on ('cuda', 'cpu')

    Example:
        >>> pipeline = GenerationPipeline("outputs/model.pt")
        >>> text = pipeline.generate("The meaning of life is", max_length=100)
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None, device: str = 'cuda', cpu: bool = False):
        if cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")

        # Determine checkpoint format and load
        checkpoint_path = Path(model_path)

        # Check if this is a DeepSpeed checkpoint
        is_deepspeed = 'mp_rank_00_model_states.pt' in model_path or model_path.endswith('/step_257000')

        # Handle DeepSpeed checkpoint path
        if model_path.endswith('/step_257000'):
            deepspeed_path = Path(model_path) / 'step_257000' / 'mp_rank_00_model_states.pt'
            meta_path = Path(model_path) / 'model.pt'
        else:
            deepspeed_path = Path(model_path) if is_deepspeed else None
            meta_path = Path(model_path).parent.parent / 'model.pt' if is_deepspeed else None

        # Load checkpoint to detect format
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        print(f" Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Detect checkpoint format
        is_new_framework = 'run_id' in checkpoint  # New framework has run metadata
        is_old_framework = 'model_state_dict' in checkpoint and 'run_id' not in checkpoint
        is_train_100m = 'epoch' in checkpoint and 'model_state_dict' in checkpoint  # train_100m_full.py format

        if is_new_framework:
            print(" Detected new framework checkpoint format")
        elif is_train_100m:
            print(" Detected train_100m_full.py checkpoint format")
        elif is_old_framework:
            print(" Detected old framework checkpoint format")
        elif is_deepspeed:
            print(" Detected DeepSpeed checkpoint format")
        else:
            print(" Detected raw state dict format")

        # Load configuration
        model_config = None

        if config_path:
            # User-provided config takes priority
            print(f" Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            model_config = config_dict.get('model', {})
        elif is_train_100m and 'config' in checkpoint:
            # train_100m_full.py format: config is stored in checkpoint
            print(" Loading config from checkpoint (train_100m_full.py)")
            config_data = checkpoint['config']
            if isinstance(config_data, dict) and 'model' in config_data:
                model_config = config_data['model']
            else:
                model_config = config_data
        elif is_new_framework and 'config' in checkpoint:
            # New framework: config is stored in checkpoint
            print(" Loading config from checkpoint (new framework)")
            config_data = checkpoint['config']
            if isinstance(config_data, dict) and 'model' in config_data:
                model_config = config_data['model']
            else:
                model_config = config_data
        elif 'config' in checkpoint:
            # Old framework or other format
            print(" Loading config from checkpoint")
            config_data = checkpoint['config']
            if isinstance(config_data, dict) and 'model' in config_data:
                model_config = config_data['model']
            else:
                model_config = config_data
        elif is_deepspeed and meta_path and meta_path.exists():
            # Load config from metadata checkpoint for DeepSpeed
            print(f" Loading config from DeepSpeed metadata: {meta_path}")
            meta_checkpoint = torch.load(meta_path, map_location=self.device, weights_only=False)
            if 'config' in meta_checkpoint:
                model_config = meta_checkpoint['config'].get('model', meta_checkpoint['config'])

        if model_config is None:
            raise ValueError(
                "No model configuration found. Please provide --config-path or ensure "
                "the checkpoint contains configuration information."
            )

        # Sanitize model config - fix type issues and filter to valid fields
        if 'layer_norm_eps' in model_config and isinstance(model_config['layer_norm_eps'], str):
            model_config['layer_norm_eps'] = float(model_config['layer_norm_eps'])
        if 'rope_theta' in model_config and isinstance(model_config['rope_theta'], str):
            model_config['rope_theta'] = float(model_config['rope_theta'])

        # Filter to only valid EnhancedMoEConfig fields
        valid_keys = set(EnhancedMoEConfig.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}

        # FIX: Detect expert format from checkpoint state dict and override activation if needed
        # This handles checkpoints trained with non-gated experts (gelu) despite config saying swiglu
        ckpt_state_dict = None
        if 'model_state_dict' in checkpoint:
            ckpt_state_dict = checkpoint['model_state_dict']
        elif 'module' in checkpoint:
            ckpt_state_dict = checkpoint['module']

        if ckpt_state_dict is not None:
            # Check for gated vs non-gated expert weights
            has_gated = any('gate_up_weights' in k for k in ckpt_state_dict.keys())
            has_non_gated = any('.experts.up_weights' in k for k in ckpt_state_dict.keys())

            if has_non_gated and not has_gated:
                # Checkpoint uses non-gated experts (trained with gelu despite config claiming swiglu)
                filtered_config['activation'] = 'gelu'
                print("  Note: Detected non-gated expert weights, using activation='gelu'")
            elif has_gated and not has_non_gated:
                # Checkpoint uses gated experts (swiglu/geglu)
                if filtered_config.get('activation', 'gelu') == 'gelu':
                    filtered_config['activation'] = 'swiglu'
                    print("  Note: Detected gated expert weights, using activation='swiglu'")

            # FIX: Detect tied vs untied embeddings mismatch
            # If checkpoint has separate lm_head.weight and token_embedding.weight,
            # don't tie them even if config says to (checkpoint may have been saved incorrectly)
            has_separate_lm_head = 'lm_head.weight' in ckpt_state_dict and 'token_embedding.weight' in ckpt_state_dict
            if has_separate_lm_head and filtered_config.get('tie_word_embeddings', False):
                # Check if they're actually different (torch is already imported at module level)
                lm_w = ckpt_state_dict['lm_head.weight']
                emb_w = ckpt_state_dict['token_embedding.weight']
                # Use simple comparison to avoid torch import issue
                if lm_w.shape == emb_w.shape and not (lm_w == emb_w).all().item():
                    filtered_config['tie_word_embeddings'] = False
                    print("  Note: Checkpoint has separate lm_head weights, disabling tie_word_embeddings")

        # Initialize model
        print(" Initializing model...")
        self.config = EnhancedMoEConfig(**filtered_config)
        self.model = EnhancedMoEModel(self.config)

        # Load model state
        if is_deepspeed and deepspeed_path and deepspeed_path.exists():
            print(f" Loading DeepSpeed model state from {deepspeed_path}")
            ds_checkpoint = torch.load(deepspeed_path, map_location=self.device, weights_only=False)
            if 'module' in ds_checkpoint:
                state_dict = strip_torch_compile_prefix(ds_checkpoint['module'])
                self.model.load_state_dict(state_dict, strict=False)
                print(f" DeepSpeed model loaded successfully")
            else:
                raise ValueError("Invalid DeepSpeed checkpoint format")
        elif is_train_100m and 'model_state_dict' in checkpoint:
            # train_100m_full.py format
            print(" Loading model state (train_100m_full.py)")
            state_dict = strip_torch_compile_prefix(checkpoint['model_state_dict'])
            self.model.load_state_dict(state_dict, strict=False)
            print(f" Model loaded successfully")
            print(f"  Epoch: {checkpoint.get('epoch', '?')}, Step: {checkpoint.get('step', '?')}")
        elif is_new_framework and 'model_state_dict' in checkpoint:
            # New framework format
            print(" Loading model state (new framework)")
            state_dict = strip_torch_compile_prefix(checkpoint['model_state_dict'])
            self.model.load_state_dict(state_dict, strict=False)
            print(f" Model loaded from run: {checkpoint.get('run_id', 'unknown')}")
            print(f"  Epoch: {checkpoint.get('epoch', '?')}, Step: {checkpoint.get('step', '?')}, Loss: {checkpoint.get('loss', '?'):.4f}")
        elif 'model_state_dict' in checkpoint:
            # Old framework format
            print(" Loading model state (old framework)")
            state_dict = strip_torch_compile_prefix(checkpoint['model_state_dict'])
            self.model.load_state_dict(state_dict, strict=False)
            print(f" Model loaded successfully")
        elif 'module' in checkpoint:
            # DeepSpeed format
            print(" Loading model state (DeepSpeed)")
            state_dict = strip_torch_compile_prefix(checkpoint['module'])
            self.model.load_state_dict(state_dict, strict=False)
            print(f" Model loaded successfully")
        else:
            # Raw state dict
            print(" Loading model state (raw state dict)")
            state_dict = strip_torch_compile_prefix(checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            print(f" Model loaded successfully")

        # FIX: Verify expert weights loaded correctly (defensive check)
        if ckpt_state_dict is not None:
            model_keys = set(self.model.state_dict().keys())
            ckpt_keys = set(ckpt_state_dict.keys())
            missing_in_ckpt = model_keys - ckpt_keys
            expert_missing = [k for k in missing_in_ckpt if 'expert' in k and 'weight' in k]
            if len(expert_missing) > 10:
                print(f" WARNING: {len(expert_missing)} expert weight keys not loaded!")
                print(f"          This may cause garbage output. Check activation mismatch.")
                print(f"          Checkpoint has: {'gate_up_weights' if 'gate_up_weights' in str(ckpt_keys) else 'up_weights'}")
                print(f"          Model expects: {'gate_up_weights' if 'gate_up_weights' in str(model_keys) else 'up_weights'}")

        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer - use default path
        default_tokenizer_path = '/root/Ava_AI/pretokenized_data/tokenizers/vocab'
        tokenizer_path_resolved = default_tokenizer_path

        print(f" Loading tokenizer from {tokenizer_path_resolved}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_resolved)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f" Tokenizer loaded: vocab_size={len(self.tokenizer)}")

        # Initialize generator (kept for backward compatibility, but we use generate_text directly)
        self.generator = TextGenerator(self.model, self.tokenizer)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        num_beams: int = 1,
        repetition_penalty: float = 1.2,
        eos_penalty: float = 1.0,
        do_sample: bool = True,
        use_ngram_blocking: bool = False,
        ngram_size: int = 3
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s) using the proven training generation logic.

        Args:
            prompt: Input prompt(s) for generation
            max_length: Maximum length of generated text
            min_length: Minimum length before allowing EOS token
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search (not used, kept for API compat)
            repetition_penalty: Penalty for repeating tokens
            eos_penalty: Penalty multiplier for EOS token (not used, kept for API compat)
            do_sample: Whether to use sampling (not used, kept for API compat)

        Returns:
            Generated text(s)
        """
        single_prompt = isinstance(prompt, str)
        if single_prompt:
            prompt = [prompt]

        generated_texts = []
        device_str = str(self.device)

        for p in tqdm(prompt, desc="Generating", disable=len(prompt) == 1):
            # Use the new generate_text function with proven logic
            output = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=p,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device_str,
            )
            generated_texts.append(output)

        return generated_texts[0] if single_prompt else generated_texts

    def generate_stream(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        eos_penalty: float = 1.0,
        do_sample: bool = True,
        use_ngram_blocking: bool = False,
        ngram_size: int = 3
    ):
        """
        Stream generate text token by token using proven training logic.

        Yields:
            str: Each token as it's generated
        """
        device_str = str(self.device)
        for token in generate_text_stream(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device_str,
        ):
            yield token

    def interactive_generation(self):
        """
        Interactive generation mode for real-time text generation.
        Uses OpenOrca format (System/User/Assistant) for prompts.
        """
        print_interactive_banner()

        # Default settings
        settings = {
            'max_length': 150,
            'temperature': 0.8,
            'top_p': 0.92,
            'top_k': 50,
            'repetition_penalty': 1.15
        }

        # OpenOrca format settings
        system_prompt = "You are a helpful assistant that provides clear and accurate information."
        use_openorca_format = False  # Raw mode by default
        conversation_history = []

        while True:
            try:
                prompt_label = get_user_prompt_label(use_openorca_format)
                user_input = input(prompt_label).strip()

                if user_input.lower() == 'quit':
                    print_info("Goodbye!")
                    break

                if user_input.startswith('/settings'):
                    print_interactive_settings(settings, system_prompt, use_openorca_format)
                    continue

                if user_input.startswith('/set '):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        if param in settings:
                            try:
                                settings[param] = type(settings[param])(value)
                                print_success(f"Updated {param} to {value}")
                            except ValueError:
                                if supports_color():
                                    print(f"{Colors.RED}Invalid value for {param}{Colors.RESET}")
                                else:
                                    print(f"Invalid value for {param}")
                        else:
                            valid_params = ', '.join(settings.keys())
                            if supports_color():
                                print(f"{Colors.YELLOW}Unknown parameter: {param}{Colors.RESET}")
                                print(f"{Colors.GRAY}Valid parameters: {valid_params}{Colors.RESET}")
                            else:
                                print(f"Unknown parameter: {param}")
                                print(f"Valid parameters: {valid_params}")
                    else:
                        valid_params = ', '.join(settings.keys())
                        print_info(f"Usage: /set <parameter> <value>")
                        print_info(f"Parameters: {valid_params}")
                    continue

                if user_input.startswith('/system '):
                    system_prompt = user_input[8:].strip()
                    print_success("System prompt updated")
                    print_system_prompt_display(system_prompt)
                    conversation_history = []  # Clear history on system change
                    continue

                if user_input == '/clear':
                    conversation_history = []
                    print_success("Conversation history cleared")
                    continue

                if user_input == '/chat' or user_input == '/raw':
                    use_openorca_format = not use_openorca_format
                    mode = "Chat mode (OpenOrca)" if use_openorca_format else "Raw mode"
                    print_info(f"Switched to: {mode}")
                    continue

                if not user_input:
                    continue

                # Build the prompt
                if use_openorca_format:
                    # Build OpenOrca format prompt
                    full_prompt = f"System: {system_prompt}\n"

                    # Add conversation history (last few turns for context)
                    for turn in conversation_history[-4:]:  # Keep last 2 exchanges
                        full_prompt += f"User: {turn['user']}\n"
                        full_prompt += f"Assistant: {turn['assistant']}\n"

                    full_prompt += f"User: {user_input}\nAssistant:"
                else:
                    full_prompt = user_input

                # Stream tokens as they're generated
                # Show spinner until first token arrives
                if use_openorca_format:
                    output_prefix = "Assistant: "
                else:
                    # In raw mode, show the prompt being continued
                    output_prefix = f"{user_input}"
                spinner = Spinner(output_prefix)
                spinner.start()
                response_tokens = []
                stop_generation = False
                first_token = True

                for token in self.generate_stream(full_prompt, **settings):
                    # Stop spinner on first token
                    if first_token:
                        spinner.stop()
                        first_token = False
                    # Check for stop tokens
                    response_tokens.append(token)
                    response_so_far = ''.join(response_tokens)

                    # Check if we hit a stop sequence
                    for stop_token in ["\nUser:", "\nSystem:", "\n\nUser:", "\n\nSystem:"]:
                        if stop_token in response_so_far:
                            # Print only up to the stop token
                            clean_response = response_so_far.split(stop_token)[0]
                            # Calculate what we haven't printed yet
                            already_printed = ''.join(response_tokens[:-1])
                            remaining = clean_response[len(already_printed):]
                            print(remaining, end="", flush=True)
                            stop_generation = True
                            break

                    if stop_generation:
                        break

                    # Print the token
                    print(token, end="", flush=True)

                # Stop spinner if no tokens were generated
                if first_token:
                    spinner.stop()

                print()  # Newline after response
                response = ''.join(response_tokens)

                # Clean up response for history
                for stop_token in ["\nUser:", "\nSystem:", "\n\nUser:", "\n\nSystem:"]:
                    if stop_token in response:
                        response = response.split(stop_token)[0].strip()

                # Save to conversation history
                if use_openorca_format:
                    conversation_history.append({
                        'user': user_input,
                        'assistant': response
                    })

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate text using trained Qwen MoE++ model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest checkpoint from most recent run
  python generate.py --prompt "Once upon a time"

  # Use specific run ID
  python generate.py --run-id run_20250928_124122_4499b1de --prompt "Hello"

  # Use best checkpoint from specific run
  python generate.py --run-id run_20250928_124122_4499b1de --checkpoint-type best --prompt "Hello"

  # Use specific checkpoint file
  python generate.py --model-path /path/to/checkpoint.pt --prompt "Hello"
"""
    )

    # Model arguments - make model-path optional when using run-id
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model-path', type=str,
                       help='Path to trained model checkpoint')
    model_group.add_argument('--run-id', type=str,
                       help='Run ID to load checkpoint from (e.g., run_20250928_124122_4499b1de)')
    model_group.add_argument('--list-runs', action='store_true',
                       help='List available training runs and exit')

    parser.add_argument('--checkpoint-type', type=str, default='latest',
                       choices=['latest', 'best'],
                       help='Which checkpoint to load from run (default: latest)')
    parser.add_argument('--config-path', type=str,
                       help='Path to model configuration (if not in checkpoint)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (overrides --device)')

    # Generation mode (not required if --list-runs is used)
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--prompt', type=str,
                           help='Single prompt for generation')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive generation mode')
    mode_group.add_argument('--input-file', type=str,
                           help='File containing prompts (one per line)')

    # Output
    parser.add_argument('--output-file', type=str,
                       help='File to save generated texts')

    # Generation parameters
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length of generated text')
    parser.add_argument('--min-length', type=int, default=0,
                       help='Minimum tokens to generate before allowing EOS')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling probability threshold')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--num-beams', type=int, default=1,
                       help='Number of beams for beam search')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Penalty for repeating tokens (default: 1.2)')
    parser.add_argument('--eos-penalty', type=float, default=1.0,
                       help='Penalty multiplier for EOS token (>1.0 = discourage EOS)')
    parser.add_argument('--use-ngram-blocking', action='store_true',
                       help='Enable n-gram blocking to prevent repetition')
    parser.add_argument('--ngram-size', type=int, default=3,
                       help='Size of n-grams to block (default: 3)')
    parser.add_argument('--no-sample', action='store_true',
                       help='Use greedy decoding instead of sampling')

    args = parser.parse_args()

    # Validate that generation mode is specified (unless --list-runs)
    if not args.list_runs and not any([args.prompt, args.interactive, args.input_file]):
        parser.error("One of --prompt, --interactive, --input-file, or --list-runs is required")

    # Handle --list-runs
    if args.list_runs:
        runs = list_available_runs()
        if not runs:
            print("No training runs found in /root/Ava_AI/code/outputs/runs/")
            return

        print("\nAvailable training runs:\n")
        for i, run_dir in enumerate(runs, 1):
            run_id = run_dir.name
            # Check for available checkpoints
            checkpoints = []
            if (run_dir / 'checkpoints/latest_model.pt').exists():
                checkpoints.append('latest')
            if (run_dir / 'checkpoints/best_model.pt').exists():
                checkpoints.append('best')

            # Get run metadata if available
            metadata_file = run_dir / 'configs/run_metadata.json'
            metadata_str = ""
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if 'created_at' in metadata:
                            created = datetime.fromisoformat(metadata['created_at'])
                            metadata_str = f" - Created: {created.strftime('%Y-%m-%d %H:%M:%S')}"
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                    logging.debug(f"Could not read metadata for {run_id}: {e}")

            print(f"{i}. {run_id}{metadata_str}")
            print(f"   Checkpoints: {', '.join(checkpoints) if checkpoints else 'none'}")
            print()

        return

    # Determine model path
    if args.run_id:
        # Load from specific run
        run_dir = Path('/root/Ava_AI/code/outputs/runs') / args.run_id
        if not run_dir.exists():
            print(f" Run not found: {args.run_id}")
            print("\nAvailable runs:")
            for run in list_available_runs()[:5]:
                print(f"  - {run.name}")
            return

        model_path = str(get_checkpoint_path_from_run(run_dir, args.checkpoint_type))
        if not Path(model_path).exists():
            print(f" Checkpoint not found: {model_path}")
            print(f"\nAvailable checkpoints in {args.run_id}:")
            checkpoints_dir = run_dir / 'checkpoints'
            if checkpoints_dir.exists():
                for item in checkpoints_dir.iterdir():
                    print(f"  - {item.name}")
            return

        print(f"Using checkpoint: {model_path}")

    elif args.model_path:
        # Use explicitly provided path
        model_path = args.model_path
    else:
        # Auto-discover latest run
        print("No --model-path or --run-id specified, searching for latest run...")
        latest_run = find_latest_run()
        if not latest_run:
            print(" No training runs found in /root/Ava_AI/code/outputs/runs/")
            print("\nPlease specify --model-path or --run-id, or train a model first.")
            print("Use --list-runs to see available runs.")
            return

        model_path = str(get_checkpoint_path_from_run(latest_run, args.checkpoint_type))
        if not Path(model_path).exists():
            print(f" Checkpoint not found: {model_path}")
            return

        print(f" Auto-discovered latest run: {latest_run.name}")
        print(f"  Using checkpoint: {model_path}")

    # Initialize pipeline
    pipeline = GenerationPipeline(
        model_path=model_path,
        config_path=args.config_path,
        device=args.device,
        cpu=args.cpu
    )

    # Handle different modes
    if args.interactive:
        pipeline.interactive_generation()

    elif args.prompt:
        # Single prompt generation
        print_generation_header()
        print_generation_params(args)
        print_generation_separator()

        output = pipeline.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            eos_penalty=args.eos_penalty,
            do_sample=not args.no_sample,
            use_ngram_blocking=args.use_ngram_blocking,
            ngram_size=args.ngram_size
        )

        print_prompt(args.prompt)
        print_generated_text(output)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                if isinstance(output, list):
                    f.write('\n'.join(output))
                else:
                    f.write(output)
            print_success(f"Saved to {args.output_file}")

    elif args.input_file:
        # Batch generation from file
        print_generation_header()

        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print_info(f"Loaded {len(prompts)} prompts from {args.input_file}")
        print_generation_params(args)
        print_generation_separator()

        outputs = pipeline.generate(
            prompt=prompts,
            max_length=args.max_length,
            min_length=args.min_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            eos_penalty=args.eos_penalty,
            do_sample=not args.no_sample,
            use_ngram_blocking=args.use_ngram_blocking,
            ngram_size=args.ngram_size
        )

        if args.output_file:
            with open(args.output_file, 'w') as f:
                for prompt, output in zip(prompts, outputs):
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Response: {output}\n")
                    f.write("-" * 50 + "\n")
            print_success(f"Saved {len(outputs)} responses to {args.output_file}")
        else:
            for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
                print_batch_progress(i, len(prompts), prompt)
                print_prompt(prompt, label=f"Prompt {i}")
                print_generated_text(output, label="Response")
                print_generation_separator()


if __name__ == "__main__":
    main()