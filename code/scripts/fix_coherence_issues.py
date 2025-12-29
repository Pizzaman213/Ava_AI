#!/usr/bin/env python3
"""
Fix Coherence Issues Script

This script diagnoses and fixes common coherence problems in the Ava LLM training framework.

Run: python code/scripts/fix_coherence_issues.py

Issues addressed:
1. Tokenizer special tokens not loading
2. MoE auxiliary losses dominating CE loss
3. Generation parameters not tuned for coherence
4. Data quality issues
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_tokenizer(tokenizer_path: str) -> dict:
    """Check if tokenizer loads special tokens correctly."""
    results = {"status": "unknown", "issues": [], "fixes": []}

    try:
        from transformers import PreTrainedTokenizerFast

        path = Path(tokenizer_path)
        if not path.exists():
            results["status"] = "error"
            results["issues"].append(f"Tokenizer path not found: {tokenizer_path}")
            results["fixes"].append("Run: python code/scripts/1_data_download/unified_download.py")
            return results

        # Try loading tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path))

        # Check special tokens
        issues = []
        if tokenizer.eos_token_id is None:
            issues.append("EOS token not loaded (causes infinite generation)")
        if tokenizer.bos_token_id is None:
            issues.append("BOS token not loaded (causes bad sequence starts)")
        if tokenizer.pad_token_id is None:
            issues.append("PAD token not loaded (causes attention mask issues)")

        if issues:
            results["status"] = "failed"
            results["issues"] = issues
            results["fixes"].append("Create tokenizer_config.json and special_tokens_map.json")
        else:
            results["status"] = "passed"
            results["details"] = {
                "eos": f"{tokenizer.eos_token} (ID: {tokenizer.eos_token_id})",
                "bos": f"{tokenizer.bos_token} (ID: {tokenizer.bos_token_id})",
                "pad": f"{tokenizer.pad_token} (ID: {tokenizer.pad_token_id})",
            }

    except ImportError:
        results["status"] = "skipped"
        results["issues"].append("transformers not installed")
    except Exception as e:
        results["status"] = "error"
        results["issues"].append(str(e))

    return results


def check_config(config_path: str) -> dict:
    """Check if config has good coherence settings."""
    results = {"status": "unknown", "issues": [], "recommendations": []}

    try:
        import yaml

        path = Path(config_path)
        if not path.exists():
            results["status"] = "error"
            results["issues"].append(f"Config not found: {config_path}")
            return results

        with open(path) as f:
            config = yaml.safe_load(f)

        model = config.get("model", {})
        training = config.get("training", {})

        # Check MoE loss coefficients
        router_z = model.get("router_z_loss_coef", 0.001)
        load_balance = model.get("load_balance_loss_coef", 0.01)

        if router_z > 0.001:
            results["issues"].append(f"router_z_loss_coef too high: {router_z} (should be <= 0.001)")
            results["recommendations"].append("Set router_z_loss_coef: 0.0001")

        if load_balance > 0.01:
            results["issues"].append(f"load_balance_loss_coef too high: {load_balance} (should be <= 0.01)")
            results["recommendations"].append("Set load_balance_loss_coef: 0.001")

        # Check coherence regularization
        entropy_reg = model.get("entropy_regularization", 0.005)
        if entropy_reg > 0.01:
            results["issues"].append(f"entropy_regularization too high: {entropy_reg} (can cause incoherence)")
            results["recommendations"].append("Set entropy_regularization: 0.001")

        output_div = model.get("output_diversity_weight", 0.0)
        if output_div > 0.001:
            results["issues"].append(f"output_diversity_weight too high: {output_div} (can cause incoherence)")
            results["recommendations"].append("Set output_diversity_weight: 0.0 or 0.0005")

        # Check label smoothing
        label_smooth = model.get("label_smoothing", 0.0)
        if label_smooth == 0.0:
            results["recommendations"].append("Enable label_smoothing: 0.1 (prevents overfitting)")

        # Check tie_word_embeddings
        tie_embed = model.get("tie_word_embeddings", False)
        if not tie_embed:
            results["recommendations"].append("Enable tie_word_embeddings: true (improves vocab learning)")

        # Check generation settings
        gen = training.get("generation", {})
        rep_penalty = gen.get("repetition_penalty", 1.0)
        if rep_penalty < 1.3:
            results["recommendations"].append(f"Increase repetition_penalty from {rep_penalty} to 1.5")

        no_repeat = gen.get("no_repeat_ngram_size", 0)
        if no_repeat == 0:
            results["recommendations"].append("Enable no_repeat_ngram_size: 3")

        if results["issues"]:
            results["status"] = "issues_found"
        elif results["recommendations"]:
            results["status"] = "ok_with_recommendations"
        else:
            results["status"] = "passed"

    except ImportError:
        results["status"] = "skipped"
        results["issues"].append("PyYAML not installed")
    except Exception as e:
        results["status"] = "error"
        results["issues"].append(str(e))

    return results


def fix_tokenizer_config(tokenizer_dir: str) -> bool:
    """Create tokenizer config files if missing."""
    path = Path(tokenizer_dir)
    path.mkdir(parents=True, exist_ok=True)

    # tokenizer_config.json
    config_file = path / "tokenizer_config.json"
    if not config_file.exists():
        config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 512,
            "padding_side": "right",
            "pad_token": "<|pad|>",
            "eos_token": "<|eos|>",
            "bos_token": "<|bos|>",
            "unk_token": "<|unk|>"
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created: {config_file}")

    # special_tokens_map.json
    tokens_file = path / "special_tokens_map.json"
    if not tokens_file.exists():
        tokens = {
            "pad_token": "<|pad|>",
            "eos_token": "<|eos|>",
            "bos_token": "<|bos|>",
            "unk_token": "<|unk|>"
        }
        with open(tokens_file, "w") as f:
            json.dump(tokens, f, indent=2)
        print(f"Created: {tokens_file}")

    return True


def print_coherence_guide():
    """Print coherence troubleshooting guide."""
    guide = """
================================================================================
                     COHERENCE TROUBLESHOOTING GUIDE
================================================================================

SYMPTOMS -> LIKELY CAUSE -> FIX

1. REPETITIVE OUTPUT ("the the the the")
   Cause: Tokenizer EOS/BOS not loaded
   Fix:   Create tokenizer_config.json with special tokens

   Cause: Early training (< 1000 steps)
   Fix:   Continue training - this is normal early behavior

2. GIBBERISH OUTPUT (random characters)
   Cause: MoE aux losses too high (dominating CE loss)
   Fix:   Reduce router_z_loss_coef to 0.0001
          Reduce load_balance_loss_coef to 0.001

   Cause: Learning rate too high
   Fix:   Reduce learning_rate by 2-5x

3. VERY SHORT OUTPUTS (stops after few words)
   Cause: EOS token learned too aggressively
   Fix:   Set eos_logit_bias: -0.5 (negative bias)
          Increase min_sequence_length: 20

4. LOW UNIQUE_TOKEN_RATIO (< 0.10)
   Cause: Degenerate generation loop
   Fix:   Verify tokenizer special tokens
          Increase repetition_penalty to 1.5
          Enable no_repeat_ngram_size: 3

5. HIGH PERPLEXITY (> 100)
   Cause: Model not learning
   Fix:   Check data quality (run unified_download.py)
          Verify data has BOS/EOS tokens
          Reduce learning rate if loss not decreasing

RECOMMENDED CONFIG FOR COHERENCE:
---------------------------------
model:
  # Reduced MoE losses
  router_z_loss_coef: 0.0001
  load_balance_loss_coef: 0.001
  diversity_loss_coef: 0.001

  # Anti-repetition
  entropy_regularization: 0.001
  output_diversity_weight: 0.0  # Can cause issues if > 0

  # Training stability
  label_smoothing: 0.1
  tie_word_embeddings: true

training:
  generation:
    repetition_penalty: 1.5
    no_repeat_ngram_size: 3
    temperature: 0.5
    top_k: 25

MONITORING (watch these in WandB):
----------------------------------
unique_token_ratio:  < 0.10 = bad, > 0.30 = good, > 0.50 = great
coherence_score:     < 0.30 = bad, > 0.50 = good, > 0.70 = great
perplexity:          > 100 = bad, 20-50 = ok, < 20 = good

================================================================================
"""
    print(guide)


def main():
    print("=" * 60)
    print("COHERENCE ISSUES DIAGNOSTIC")
    print("=" * 60)

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Check common tokenizer paths
    tokenizer_paths = [
        project_root / "code/data/Ava_Ai/tokenizer_v3/transformers",
        project_root / "code/data/Ava_Ai/tokenizer",
        project_root / "code/models/tokenizer",
    ]

    print("\n1. TOKENIZER CHECK")
    print("-" * 40)

    tokenizer_found = False
    for tok_path in tokenizer_paths:
        if tok_path.exists():
            print(f"Checking: {tok_path}")
            result = check_tokenizer(str(tok_path))
            tokenizer_found = True

            if result["status"] == "passed":
                print(f"  Status: PASSED")
                for k, v in result.get("details", {}).items():
                    print(f"    {k}: {v}")
            else:
                print(f"  Status: {result['status'].upper()}")
                for issue in result["issues"]:
                    print(f"    Issue: {issue}")
                for fix in result["fixes"]:
                    print(f"    Fix: {fix}")
            break

    if not tokenizer_found:
        print("  No tokenizer found. Run data download first:")
        print("  python code/scripts/1_data_download/unified_download.py")

    # Check config
    print("\n2. CONFIG CHECK")
    print("-" * 40)

    config_paths = [
        project_root / "code/configs/moe/large.yaml",
        project_root / "code/configs/moe/minimal_working.yaml",
    ]

    for cfg_path in config_paths:
        if cfg_path.exists():
            print(f"Checking: {cfg_path.name}")
            result = check_config(str(cfg_path))

            if result["status"] == "passed":
                print("  Status: PASSED")
            else:
                print(f"  Status: {result['status'].upper()}")
                for issue in result["issues"]:
                    print(f"    Issue: {issue}")
                for rec in result["recommendations"]:
                    print(f"    Recommend: {rec}")

    # Print guide
    print("\n3. TROUBLESHOOTING GUIDE")
    print("-" * 40)
    print_coherence_guide()

    print("\nDone! Review the issues above and apply fixes.")


if __name__ == "__main__":
    main()
