#!/usr/bin/env python3
"""
Enhanced Text Generation Script for Ava MoE Models

This script provides advanced text generation capabilities for models trained with the
comprehensive 8-phase enhanced training framework. It seamlessly integrates with the
run management system and supports multiple checkpoint formats and generation strategies.

✨ Key Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 🔍 Smart Auto-Discovery: Automatically finds and loads the most recent trained model
• 📦 Run Management Integration: Load checkpoints by run ID with full metadata
• 🔄 Multi-Format Support: Handles new framework, legacy, DeepSpeed, and raw formats
• 🎯 Checkpoint Selection: Choose between latest, best, or step-specific checkpoints
• 🎨 Multiple Generation Modes: Single prompt, interactive session, or batch processing
• ⚙️  Advanced Sampling: Temperature, top-k, top-p, repetition penalty, beam search
• 📊 Detailed Logging: Shows model info, training metrics, and generation parameters

🚀 Quick Start:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Easiest: Auto-discover latest trained model
    python generate.py --prompt "Once upon a time"

    # List all available training runs
    python generate.py --list-runs

📋 Usage Examples:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

🎛️  Sampling Parameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --temperature    Controls randomness (0.1=focused, 1.0=balanced, 2.0=creative)
  --top-p          Nucleus sampling threshold (0.9=default, 0.95=more diverse)
  --top-k          Limits vocabulary per step (50=default, higher=more options)
  --repetition-penalty  Reduces repetition (1.0=off, 1.2-2.0=recommended)
  --num-beams      Beam search width (1=greedy, 4-8=better quality)
  --max-length     Maximum tokens to generate (default: 100)

🔗 Integration with Training:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This script is part of the comprehensive Ava training pipeline supporting:
  ✅ Phase 1-8: All training enhancements (stability, data pipeline, adaptive LR, etc.)
  ✅ Run Management: Organized checkpoint storage with full metadata
  ✅ Multi-Format: Backward compatible with all checkpoint formats
  ✅ Production Ready: Robust error handling and format detection
"""

import argparse
import sys
import warnings
import torch  # type: ignore[import-not-found]
import yaml
from pathlib import Path
from typing import Optional, List, Union
import json
from tqdm import tqdm

# Suppress Pydantic field attribute warnings early (these come from dependencies)
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings('ignore', category=UnsupportedFieldAttributeWarning)
except ImportError:
    # Newer versions of Pydantic may not have this warning
    pass

# Add project root to path
sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig  # type: ignore[import-not-found]
from src.generation.generator import TextGenerator
from transformers import AutoTokenizer  # type: ignore[import-not-found]
from datetime import datetime


def find_latest_run(base_dir: str = '/project/code/outputs/runs') -> Optional[Path]:
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


def list_available_runs(base_dir: str = '/project/code/outputs/runs') -> List[Path]:
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

        if is_new_framework:
            print(" Detected new framework checkpoint format")
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

        # Initialize model
        print(" Initializing model...")
        self.config = EnhancedMoEConfig(**filtered_config)
        self.model = EnhancedMoEModel(self.config)

        # Load model state
        if is_deepspeed and deepspeed_path and deepspeed_path.exists():
            print(f" Loading DeepSpeed model state from {deepspeed_path}")
            ds_checkpoint = torch.load(deepspeed_path, map_location=self.device, weights_only=False)
            if 'module' in ds_checkpoint:
                self.model.load_state_dict(ds_checkpoint['module'])
                print(f"✓ DeepSpeed model loaded successfully")
            else:
                raise ValueError("Invalid DeepSpeed checkpoint format")
        elif is_new_framework and 'model_state_dict' in checkpoint:
            # New framework format
            print(" Loading model state (new framework)")
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✓ Model loaded from run: {checkpoint.get('run_id', 'unknown')}")
            print(f"  Epoch: {checkpoint.get('epoch', '?')}, Step: {checkpoint.get('step', '?')}, Loss: {checkpoint.get('loss', '?'):.4f}")
        elif 'model_state_dict' in checkpoint:
            # Old framework format
            print(" Loading model state (old framework)")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded successfully")
        elif 'module' in checkpoint:
            # DeepSpeed format
            print(" Loading model state (DeepSpeed)")
            self.model.load_state_dict(checkpoint['module'])
            print(f"✓ Model loaded successfully")
        else:
            # Raw state dict
            print(" Loading model state (raw state dict)")
            self.model.load_state_dict(checkpoint)
            print(f"✓ Model loaded successfully")

        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer - try to get from config, fallback to vocab_size mapping
        tokenizer_path = None

        # First, try to get tokenizer path from checkpoint config
        if is_new_framework and 'config' in checkpoint:
            config_data = checkpoint['config']
            # Check if there's a data section with tokenizer_name
            if isinstance(config_data, dict) and 'data' in config_data:
                tokenizer_path = config_data['data'].get('tokenizer_name')

        # Fallback: auto-detect from model vocab size
        if not tokenizer_path:
            model_vocab_size = self.config.vocab_size
            tokenizer_map = {
                500: 'enhanced-500',
                1000: 'enhanced-1000',
                27000: 'enhanced-27000',
                57000: 'enhanced-57000',
                65536: 'enhanced-65536'
            }
            tokenizer_name = tokenizer_map.get(model_vocab_size, 'enhanced-65536')
            tokenizer_path = f'/project/code/models/tokenizer/{tokenizer_name}'
            print(f" Auto-detected vocab_size={model_vocab_size}, using tokenizer: {tokenizer_name}")
        else:
            print(f" Using tokenizer from checkpoint config: {tokenizer_path}")

        print(f" Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✓ Tokenizer loaded: vocab_size={len(self.tokenizer)}")

        # Initialize generator
        self.generator = TextGenerator(self.model, self.tokenizer)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 1.0,
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
        Generate text from prompt(s).

        Args:
            prompt: Input prompt(s) for generation
            max_length: Maximum length of generated text
            min_length: Minimum length before allowing EOS token
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens
            eos_penalty: Penalty multiplier for EOS token (>1.0 = discourage EOS)
            do_sample: Whether to use sampling (vs greedy decoding)

        Returns:
            Generated text(s)
        """
        single_prompt = isinstance(prompt, str)
        if single_prompt:
            prompt = [prompt]

        generated_texts = []

        for p in tqdm(prompt, desc="Generating", disable=len(prompt) == 1):
            output = self.generator.generate(
                prompt=p,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                eos_penalty=eos_penalty,
                do_sample=do_sample,
                use_ngram_blocking=use_ngram_blocking,
                ngram_size=ngram_size
            )
            generated_texts.append(output)

        return generated_texts[0] if single_prompt else generated_texts

    def interactive_generation(self):
        """
        Interactive generation mode for real-time text generation.
        """
        print("\n" + "="*60)
        print(" Interactive Generation Mode")
        print("="*60)
        print("Enter your prompts (type 'quit' to exit)")
        print("Commands: /settings - show settings, /set <param> <value> - update parameter")
        print("="*60 + "\n")

        # Default settings
        settings = {
            'max_length': 100,
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 2.0
        }

        while True:
            try:
                prompt = input("\n Prompt: ").strip()

                if prompt.lower() == 'quit':
                    break

                if prompt.startswith('/settings'):
                    print("\nCurrent settings:")
                    for k, v in settings.items():
                        print(f"  {k}: {v}")
                    continue

                if prompt.startswith('/set'):
                    parts = prompt.split()
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        if param in settings:
                            try:
                                settings[param] = type(settings[param])(value)
                                print(f" Updated {param} to {value}")
                            except ValueError:
                                print(f" Invalid value for {param}")
                        else:
                            print(f" Unknown parameter: {param}")
                    continue

                if not prompt:
                    continue

                print("\n Generating...\n")
                response = self.generate(prompt, **settings)
                print(f" Response:\n{response}")

            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break
            except Exception as e:
                print(f" Error: {e}")


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
    parser.add_argument('--repetition-penalty', type=float, default=2.0,
                       help='Penalty for repeating tokens')
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
            print("No training runs found in /project/code/outputs/runs/")
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
                except:
                    pass

            print(f"{i}. {run_id}{metadata_str}")
            print(f"   Checkpoints: {', '.join(checkpoints) if checkpoints else 'none'}")
            print()

        return

    # Determine model path
    if args.run_id:
        # Load from specific run
        run_dir = Path('/project/code/outputs/runs') / args.run_id
        if not run_dir.exists():
            print(f"❌ Run not found: {args.run_id}")
            print("\nAvailable runs:")
            for run in list_available_runs()[:5]:
                print(f"  - {run.name}")
            return

        model_path = str(get_checkpoint_path_from_run(run_dir, args.checkpoint_type))
        if not Path(model_path).exists():
            print(f"❌ Checkpoint not found: {model_path}")
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
            print("❌ No training runs found in /project/code/outputs/runs/")
            print("\nPlease specify --model-path or --run-id, or train a model first.")
            print("Use --list-runs to see available runs.")
            return

        model_path = str(get_checkpoint_path_from_run(latest_run, args.checkpoint_type))
        if not Path(model_path).exists():
            print(f"❌ Checkpoint not found: {model_path}")
            return

        print(f"✓ Auto-discovered latest run: {latest_run.name}")
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

        print(f"\n Prompt: {args.prompt}")
        print(f" Generated:\n{output}")

        if args.output_file:
            with open(args.output_file, 'w') as f:
                if isinstance(output, list):
                    f.write('\n'.join(output))
                else:
                    f.write(output)
            print(f"\n Saved to {args.output_file}")

    elif args.input_file:
        # Batch generation from file
        with open(args.input_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f" Loaded {len(prompts)} prompts from {args.input_file}")

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
            print(f" Saved {len(outputs)} responses to {args.output_file}")
        else:
            for prompt, output in zip(prompts, outputs):
                print(f"\n Prompt: {prompt}")
                print(f" Generated: {output}")
                print("-" * 50)


if __name__ == "__main__":
    main()