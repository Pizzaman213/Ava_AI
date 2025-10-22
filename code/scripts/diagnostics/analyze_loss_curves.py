#!/usr/bin/env python3
"""
Loss Curve Analysis Tool
Diagnoses overfitting and fast loss decrease issues
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import math


class LossAnalyzer:
    """Analyze training loss curves to detect overfitting"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.steps: List[int] = []

    def add_entry(self, step: int, train_loss: float, val_loss: float) -> None:
        """Add a training step entry"""
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def calculate_drop_rate(self, start_idx: int = 0, end_idx: Optional[int] = None) -> float:
        """Calculate loss drop rate over a range"""
        if end_idx is None:
            end_idx = len(self.train_losses)

        if start_idx >= end_idx or len(self.train_losses) <= start_idx:
            return 0.0

        start_loss = self.train_losses[start_idx]
        end_loss = self.train_losses[end_idx - 1]

        if start_loss <= 0:
            return 0.0

        return (start_loss - end_loss) / start_loss

    def calculate_divergence_ratio(self, start_idx: int = 0, end_idx: Optional[int] = None) -> float:
        """Calculate average val/train loss ratio"""
        if end_idx is None:
            end_idx = len(self.train_losses)

        if start_idx >= end_idx or len(self.train_losses) <= start_idx:
            return 1.0

        ratios = []
        for i in range(start_idx, end_idx):
            if self.train_losses[i] > 0:
                ratio = self.val_losses[i] / self.train_losses[i]
                ratios.append(ratio)

        return sum(ratios) / len(ratios) if ratios else 1.0

    def diagnose(self) -> Dict[str, any]:
        """Run diagnostic analysis"""
        if len(self.train_losses) < 2:
            return {"error": "Not enough data points"}

        # Overall statistics
        initial_loss = self.train_losses[0]
        final_loss = self.train_losses[-1]
        drop_rate = self.calculate_drop_rate()
        divergence = self.calculate_divergence_ratio()

        # Recent statistics (last 10% of training)
        recent_start = max(0, len(self.train_losses) - len(self.train_losses) // 10)
        recent_drop = self.calculate_drop_rate(recent_start)
        recent_div = self.calculate_divergence_ratio(recent_start)

        # Early statistics (first 10% of training)
        early_end = len(self.train_losses) // 10
        early_drop = self.calculate_drop_rate(0, early_end)

        # Find loss plateau point
        plateau_step = self._find_plateau()

        diagnosis = {
            "summary": {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "total_steps": len(self.train_losses),
                "plateau_at_step": plateau_step,
            },
            "overall_metrics": {
                "drop_rate": drop_rate,
                "val_train_ratio": divergence,
                "loss_reduction": initial_loss - final_loss,
            },
            "early_training": {
                "drop_rate": early_drop,
                "status": "memorization" if early_drop > 0.5 else "learning",
            },
            "recent_training": {
                "drop_rate": recent_drop,
                "val_train_ratio": recent_div,
                "status": "overfitting" if recent_div > 1.5 else "good",
            },
            "diagnosis": self._generate_diagnosis(drop_rate, divergence, early_drop),
        }

        return diagnosis

    def _find_plateau(self) -> Optional[int]:
        """Find the step at which loss plateaus"""
        if len(self.train_losses) < 5:
            return None

        # Check if recent changes are < 1% of loss value
        window = min(10, len(self.train_losses) // 5)
        for i in range(len(self.train_losses) - window, 0, -1):
            recent_change = abs(self.train_losses[i] - self.train_losses[i - 1])
            if recent_change < 0.01 * self.train_losses[i]:
                return self.steps[i] if i < len(self.steps) else i

        return None

    def _generate_diagnosis(
        self, drop_rate: float, divergence: float, early_drop: float
    ) -> Dict[str, any]:
        """Generate diagnostic report"""
        severity = "normal"
        causes = []
        solutions = []

        # Check for overfitting
        if divergence > 2.0:
            severity = "critical"
            causes.append("severe_divergence")
            solutions.append("Get larger dataset (10k+ samples minimum)")

        if drop_rate > 0.7:
            severity = "severe" if severity == "normal" else severity
            causes.append("very_fast_drop")
            solutions.append("Dataset may be too small or repetitive")

        if early_drop > 0.5:
            severity = "moderate" if severity == "normal" else severity
            causes.append("early_memorization")
            solutions.append("Increase regularization (dropout, weight_decay)")

        if divergence > 1.5:
            if severity == "normal":
                severity = "moderate"
            causes.append("train_val_divergence")
            solutions.append("Monitor validation performance carefully")

        # Check for slow convergence (good sign)
        if drop_rate < 0.2 and divergence < 1.1:
            severity = "healthy"
            causes = ["slow_steady_improvement"]
            solutions = ["Continue training - this is good!"]

        if not causes:
            severity = "normal"
            causes = ["normal_training"]
            solutions = ["Continue monitoring, looks okay"]

        return {
            "severity": severity,
            "causes": causes,
            "solutions": solutions,
            "interpretation": self._get_interpretation(severity),
        }

    def _get_interpretation(self, severity: str) -> str:
        """Get human-readable interpretation"""
        interpretations = {
            "healthy": "✅ Good training - steady improvement with minimal divergence",
            "normal": "✓ Normal training - proceed but monitor closely",
            "moderate": "⚠️  Possible overfitting - consider regularization or more data",
            "severe": "⚠️⚠️ Likely overfitting - loss dropping too fast, get more data",
            "critical": "❌ Critical overfitting - model not generalizing, need bigger dataset",
        }
        return interpretations.get(severity, "Unknown")

    def print_report(self) -> None:
        """Print formatted diagnostic report"""
        if not self.train_losses:
            print("❌ No loss data to analyze")
            return

        diagnosis = self.diagnose()

        if "error" in diagnosis:
            print(f"❌ Error: {diagnosis['error']}")
            return

        # Header
        print("\n" + "=" * 70)
        print("  LOSS CURVE DIAGNOSTIC REPORT")
        print("=" * 70)

        # Summary
        summary = diagnosis["summary"]
        print("\n📊 SUMMARY:")
        print(f"  Initial Loss: {summary['initial_loss']:.4f}")
        print(f"  Final Loss:   {summary['final_loss']:.4f}")
        print(f"  Total Steps:  {summary['total_steps']}")
        if summary["plateau_at_step"]:
            print(f"  Plateau At:   Step {summary['plateau_at_step']}")

        # Overall metrics
        overall = diagnosis["overall_metrics"]
        print("\n📈 OVERALL METRICS:")
        print(f"  Loss Drop Rate:     {overall['drop_rate']:.1%}")
        print(f"  Val/Train Ratio:    {overall['val_train_ratio']:.2f}x")
        print(f"  Total Reduction:    {overall['loss_reduction']:.4f}")

        # Early vs Recent
        early = diagnosis["early_training"]
        recent = diagnosis["recent_training"]
        print("\n⏱️  PROGRESSION:")
        print(f"  Early Drop Rate:    {early['drop_rate']:.1%} ({early['status']})")
        print(f"  Recent Drop Rate:   {recent['drop_rate']:.1%}")
        print(f"  Recent Val/Train:   {recent['val_train_ratio']:.2f}x ({recent['status']})")

        # Diagnosis
        diag = diagnosis["diagnosis"]
        print("\n🔍 DIAGNOSIS:")
        print(f"  Severity:           {diag['severity'].upper()}")
        print(f"  Interpretation:     {diag['interpretation']}")

        if diag["causes"]:
            print(f"\n  Likely Causes:")
            for cause in diag["causes"]:
                print(f"    • {cause.replace('_', ' ').title()}")

        if diag["solutions"]:
            print(f"\n  Recommended Solutions:")
            for i, solution in enumerate(diag["solutions"], 1):
                print(f"    {i}. {solution}")

        print("\n" + "=" * 70 + "\n")

    def save_json(self, filepath: str) -> None:
        """Save analysis as JSON"""
        diagnosis = self.diagnose()
        with open(filepath, "w") as f:
            json.dump(diagnosis, f, indent=2)
        if self.verbose:
            print(f"✓ Analysis saved to {filepath}")

    def plot_ascii(self) -> None:
        """Print ASCII plot of loss curves"""
        if len(self.train_losses) < 2:
            print("Not enough data to plot")
            return

        max_loss = max(max(self.train_losses), max(self.val_losses))
        min_loss = min(min(self.train_losses), min(self.val_losses))
        range_loss = max_loss - min_loss if max_loss > min_loss else 1

        # Sample every Nth point if too many
        sample_rate = max(1, len(self.train_losses) // 50)
        indices = range(0, len(self.train_losses), sample_rate)

        print("\n📉 LOSS CURVES (ASCII):\n")
        print(f"Loss^")
        print(f"     {max_loss:.2f}┤")

        # Plot each row
        height = 20
        for row in range(height, 0, -1):
            threshold = min_loss + (range_loss * row / height)
            line = f"     {threshold:.2f}┤"

            for idx in indices:
                if self.train_losses[idx] >= threshold:
                    line += "T"
                elif self.val_losses[idx] >= threshold:
                    line += "V"
                else:
                    line += " "

            print(line)

        print(f"     {min_loss:.2f}┤" + "─" * (len(list(indices)) + 1))
        print(f"        └" + "─" * len(list(indices)))
        print(f"\n     T = Training loss")
        print(f"     V = Validation loss\n")


def load_from_logs(log_file: str) -> Tuple[List[int], List[float], List[float]]:
    """Load loss data from training logs"""
    steps = []
    train_losses = []
    val_losses = []

    try:
        with open(log_file, "r") as f:
            for line in f:
                # Parse lines like: "Step 10: Loss=2.34 Val=2.45"
                if "Loss=" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.startswith("Step"):
                                step = int(parts[i + 1].rstrip(":"))
                            elif part.startswith("Loss="):
                                loss = float(part.split("=")[1])
                            elif part.startswith("Val="):
                                val = float(part.split("=")[1])

                        steps.append(step)
                        train_losses.append(loss)
                        val_losses.append(val)
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        print(f"❌ File not found: {log_file}")
        return [], [], []

    return steps, train_losses, val_losses


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_loss_curves.py <log_file> [output_json]")
        print("\nExample:")
        print("  python analyze_loss_curves.py training.log")
        print("  python analyze_loss_curves.py training.log analysis.json")
        sys.exit(1)

    log_file = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None

    # Load data
    steps, train_losses, val_losses = load_from_logs(log_file)

    if not train_losses:
        print(f"❌ Could not parse loss data from {log_file}")
        sys.exit(1)

    # Analyze
    analyzer = LossAnalyzer()
    for step, train_loss, val_loss in zip(steps, train_losses, val_losses):
        analyzer.add_entry(step, train_loss, val_loss)

    # Report
    analyzer.print_report()
    analyzer.plot_ascii()

    # Save JSON if requested
    if output_json:
        analyzer.save_json(output_json)


if __name__ == "__main__":
    main()
