#!/usr/bin/env python3
"""
Automatically test all checkpoints from the latest training run.
Finds the most recent run and evaluates each checkpoint using generation.
"""

import sys
sys.path.insert(0, '/project/code/src')
sys.path.insert(0, '/project/code')

import os
import glob
import re
import torch
from transformers import AutoTokenizer
from Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
import inspect
from typing import Optional, List, Tuple, Dict, Any

class CheckpointTester:
    """Automatically test all checkpoints from latest training run"""

    def __init__(self, runs_dir: str = '/project/code/outputs/runs'):
        self.runs_dir = runs_dir
        self.tokenizer: Optional[Any] = None
        self.device = 'cpu'  # Use CPU for testing

        # Create backward compatibility symlink
        if not os.path.exists('/project/code/src/src'):
            try:
                os.symlink('/project/code/src', '/project/code/src/src')
            except:
                pass

    def find_latest_run(self):
        """Find the most recent training run"""
        runs = glob.glob(os.path.join(self.runs_dir, 'run_*'))
        if not runs:
            raise FileNotFoundError(f"No training runs found in {self.runs_dir}")

        # Sort by modification time (most recent first)
        latest_run = max(runs, key=os.path.getmtime)
        run_name = os.path.basename(latest_run)

        print(f"📁 Found latest run: {run_name}")
        return latest_run

    def find_all_checkpoints(self, run_dir: str) -> List[Tuple[int, str]]:
        """Find all checkpoint files in a training run"""
        checkpoints_dir = os.path.join(run_dir, 'checkpoints')

        # Find all step checkpoints
        step_checkpoints: List[Tuple[int, str]] = []
        for step_dir in glob.glob(os.path.join(checkpoints_dir, 'step_*')):
            checkpoint_file = os.path.join(step_dir, 'model.pt')
            if os.path.exists(checkpoint_file):
                # Extract step number
                match = re.search(r'step_(\d+)', step_dir)
                if match:
                    step_num = int(match.group(1))
                    step_checkpoints.append((step_num, checkpoint_file))

        # Check for latest_model.pt
        latest_checkpoint = os.path.join(checkpoints_dir, 'latest_model.pt')
        if os.path.exists(latest_checkpoint):
            step_checkpoints.append((999999, latest_checkpoint))  # Put latest at end

        # Sort by step number
        step_checkpoints.sort(key=lambda x: x[0])

        print(f"📊 Found {len(step_checkpoints)} checkpoints")
        return step_checkpoints

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a checkpoint and create model"""
        print(f"\n🔄 Loading: {os.path.basename(os.path.dirname(checkpoint_path))}/{os.path.basename(checkpoint_path)}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract model config
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        elif 'config' in checkpoint:
            full_config = checkpoint['config']
            if hasattr(full_config, 'model'):
                model_config = full_config.model
            elif isinstance(full_config, dict) and 'model' in full_config:
                model_config_dict = full_config['model']
                valid_params = inspect.signature(EnhancedMoEConfig.__init__).parameters
                filtered_dict = {k: v for k, v in model_config_dict.items() if k in valid_params}

                # Fix type conversions
                if 'layer_norm_eps' in filtered_dict and isinstance(filtered_dict['layer_norm_eps'], str):
                    filtered_dict['layer_norm_eps'] = float(filtered_dict['layer_norm_eps'])
                if 'rope_theta' in filtered_dict and isinstance(filtered_dict['rope_theta'], str):
                    filtered_dict['rope_theta'] = float(filtered_dict['rope_theta'])

                model_config = EnhancedMoEConfig(**filtered_dict)
            else:
                raise ValueError("Cannot extract model config from checkpoint")
        else:
            raise ValueError("No config found in checkpoint")

        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            print("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create and load model
        model = EnhancedMoEModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model, checkpoint

    def test_generation(self, model: Any, step_num: int, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test generation with a model"""
        if prompts is None:
            prompts = [
                "Once upon a time",
                "The quick brown fox",
                "In a distant galaxy",
                "Hello, my name is"
            ]

        print(f"\n{'='*70}")
        if step_num == 999999:
            print(f"🧪 Testing LATEST checkpoint")
        else:
            print(f"🧪 Testing Step {step_num}")
        print(f"{'='*70}")

        results: Dict[str, Any] = {}

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Prompt: \"{prompt}\"")

            try:
                # Tokenize
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer not loaded")
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_length=80,
                        min_length=20,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.5,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"✅ Generated: \"{generated}\"")

                # Calculate stats
                tokens = generated.split()
                if len(tokens) > 1:
                    unique_tokens = len(set(tokens))
                    repetition_rate = 1 - (unique_tokens / len(tokens))
                    print(f"📈 Stats: {len(tokens)} tokens, {unique_tokens} unique, {repetition_rate:.1%} repetition")

                    results[prompt] = {
                        'generated': generated,
                        'total_tokens': len(tokens),
                        'unique_tokens': unique_tokens,
                        'repetition_rate': repetition_rate,
                        'success': True
                    }
                else:
                    results[prompt] = {'success': False, 'error': 'Too few tokens'}

            except Exception as e:
                print(f"❌ Generation failed: {e}")
                results[prompt] = {'success': False, 'error': str(e)}

        return results

    def run_all_tests(self, max_checkpoints: Optional[int] = None, skip_interval: Optional[int] = None) -> Dict[int, Any]:
        """
        Test all checkpoints from the latest run

        Args:
            max_checkpoints: Maximum number of checkpoints to test (None = all)
            skip_interval: Test every Nth checkpoint (None = test all)
        """
        print("🚀 Starting automated checkpoint testing")
        print(f"🖥️  Device: {self.device}")

        # Find latest run
        latest_run = self.find_latest_run()

        # Find all checkpoints
        checkpoints = self.find_all_checkpoints(latest_run)

        if not checkpoints:
            print("❌ No checkpoints found!")
            return

        # Apply filtering
        if skip_interval and skip_interval > 1:
            # Keep every Nth checkpoint, plus the latest
            latest = checkpoints[-1]
            filtered = checkpoints[::skip_interval]
            if latest not in filtered:
                filtered.append(latest)
            checkpoints = sorted(filtered, key=lambda x: x[0])
            print(f"📉 Testing every {skip_interval} checkpoints: {len(checkpoints)} total")

        if max_checkpoints and len(checkpoints) > max_checkpoints:
            # Keep first, evenly spaced middle ones, and latest
            if max_checkpoints >= 3:
                first = [checkpoints[0]]
                latest = [checkpoints[-1]]
                middle_count = max_checkpoints - 2
                step = (len(checkpoints) - 2) // middle_count
                middle = checkpoints[1:-1:step][:middle_count]
                checkpoints = first + middle + latest
            else:
                checkpoints = checkpoints[:max_checkpoints]
            print(f"📉 Limited to {max_checkpoints} checkpoints")

        # Test each checkpoint
        all_results: Dict[int, Any] = {}

        for i, (step_num, checkpoint_path) in enumerate(checkpoints, 1):
            print(f"\n{'#'*70}")
            print(f"# Checkpoint {i}/{len(checkpoints)}")
            print(f"{'#'*70}")

            try:
                # Load model
                model, checkpoint = self.load_checkpoint(checkpoint_path)

                # Test generation
                results = self.test_generation(model, step_num)
                all_results[step_num] = results

                # Free memory
                del model
                del checkpoint
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

                print(f"✅ Completed checkpoint {step_num}")

            except Exception as e:
                print(f"❌ Failed to test checkpoint {step_num}: {e}")
                import traceback
                traceback.print_exc()
                all_results[step_num] = {'error': str(e)}

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, all_results: Dict[int, Any]) -> None:
        """Print a summary of all test results"""
        print("\n" + "="*70)
        print("📊 SUMMARY OF ALL CHECKPOINTS")
        print("="*70)

        for step_num in sorted(all_results.keys()):
            results = all_results[step_num]

            if 'error' in results:
                label = "LATEST" if step_num == 999999 else f"Step {step_num:6d}"
                print(f"\n{label}: ❌ ERROR - {results['error']}")
                continue

            # Calculate average repetition rate
            successful = [r for r in results.values() if r.get('success', False)]
            if successful:
                avg_repetition = sum(r['repetition_rate'] for r in successful) / len(successful)
                avg_tokens = sum(r['total_tokens'] for r in successful) / len(successful)
                avg_unique = sum(r['unique_tokens'] for r in successful) / len(successful)

                label = "LATEST" if step_num == 999999 else f"Step {step_num:6d}"
                print(f"\n{label}: ✅ {len(successful)}/{len(results)} prompts succeeded")
                print(f"  Avg: {avg_tokens:.1f} tokens, {avg_unique:.1f} unique, {avg_repetition:.1%} repetition")
            else:
                label = "LATEST" if step_num == 999999 else f"Step {step_num:6d}"
                print(f"\n{label}: ❌ All prompts failed")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test all checkpoints from latest training run')
    parser.add_argument('--runs-dir', default='/project/code/outputs/runs',
                       help='Directory containing training runs')
    parser.add_argument('--max-checkpoints', type=int, default=None,
                       help='Maximum number of checkpoints to test')
    parser.add_argument('--skip-interval', type=int, default=None,
                       help='Test every Nth checkpoint (e.g., 2 = every other)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                       help='Device to use for testing')

    args = parser.parse_args()

    tester = CheckpointTester(runs_dir=args.runs_dir)
    if args.device:
        tester.device = args.device

    tester.run_all_tests(
        max_checkpoints=args.max_checkpoints,
        skip_interval=args.skip_interval
    )


if __name__ == "__main__":
    main()
