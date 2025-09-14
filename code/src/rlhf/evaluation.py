"""
Comprehensive Evaluation Framework
Implements metrics for factual accuracy, logical consistency, calibration, and reasoning quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import json
from pathlib import Path
from collections import defaultdict, Counter
import re
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    # Factual accuracy
    fact_check_enabled: bool = True
    fact_check_sources: List[str] = field(default_factory=lambda: ["wikipedia", "wikidata"])
    min_fact_confidence: float = 0.7
    
    # Logical consistency
    consistency_check_enabled: bool = True
    contradiction_penalty: float = 0.5
    consistency_threshold: float = 0.8
    
    # Calibration
    calibration_bins: int = 10
    plot_calibration: bool = True
    
    # Hallucination detection
    hallucination_threshold: float = 0.3
    check_entity_hallucination: bool = True
    check_fact_hallucination: bool = True
    
    # Reasoning quality
    min_reasoning_steps: int = 2
    max_reasoning_steps: int = 20
    reasoning_coherence_threshold: float = 0.7
    
    # Cross-domain transfer
    test_domains: List[str] = field(default_factory=lambda: [
        "mathematics", "science", "history", "literature", "code"
    ])
    
    # Output settings
    save_results: bool = True
    results_dir: str = "evaluation_results"
    generate_report: bool = True

class FactualAccuracyMetric:
    """Evaluates factual accuracy of model outputs"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.fact_cache = {}
        self.entity_kb = self._load_entity_kb()
    
    def _load_entity_kb(self) -> Set[str]:
        """Load known entities (simplified)"""
        # In practice, load from knowledge base
        return {
            "Earth", "Sun", "Moon", "United States", "Europe",
            "Einstein", "Newton", "Darwin", "Shakespeare",
            # Add more entities
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate factual accuracy"""
        results = {
            "scores": [],
            "fact_checks": [],
            "entity_accuracy": [],
            "claim_verification": []
        }
        
        for i, pred in enumerate(tqdm(predictions, desc="Evaluating factual accuracy")):
            context = contexts[i] if contexts else None
            reference = references[i] if references else None
            
            # Extract and verify facts
            facts = self._extract_facts(pred)
            fact_scores = []
            
            for fact in facts:
                score = self._verify_fact(fact, context, reference)
                fact_scores.append(score)
                results["fact_checks"].append({
                    "fact": fact,
                    "score": score,
                    "verified": score > self.config.min_fact_confidence
                })
            
            # Check entity accuracy
            entities = self._extract_entities(pred)
            entity_score = self._check_entity_accuracy(entities, reference)
            results["entity_accuracy"].append(entity_score)
            
            # Overall score
            if fact_scores:
                overall_score = np.mean(fact_scores) * 0.7 + entity_score * 0.3
            else:
                overall_score = entity_score
            
            results["scores"].append(overall_score)
        
        # Aggregate metrics
        results["mean_accuracy"] = np.mean(results["scores"])
        results["fact_verification_rate"] = np.mean([
            fc["verified"] for fc in results["fact_checks"]
        ]) if results["fact_checks"] else 0.0
        
        return results
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        facts = []
        
        # Simple fact patterns
        fact_patterns = [
            r"(?:is|are|was|were)\s+([^.]+)",
            r"(?:has|have|had)\s+([^.]+)",
            r"(?:in|on|at)\s+(\d{4})",  # Years
            r"(\d+(?:\.\d+)?)\s*(?:percent|%|degrees|meters|kilometers)",  # Quantities
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            for pattern in fact_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                facts.extend([f"{sentence.strip()}." for m in matches if m])
        
        return facts[:10]  # Limit to top 10 facts
    
    def _verify_fact(
        self,
        fact: str,
        context: Optional[str] = None,
        reference: Optional[str] = None
    ) -> float:
        """Verify a single fact"""
        # Check cache
        if fact in self.fact_cache:
            return self.fact_cache[fact]
        
        score = 0.5  # Default uncertainty
        
        # Check against reference if available
        if reference:
            if fact.lower() in reference.lower():
                score = 1.0
            elif any(word in reference.lower() for word in fact.lower().split()):
                score = 0.7
        
        # Simple heuristics for common facts
        if "earth" in fact.lower() and "round" in fact.lower():
            score = 1.0
        elif "sun" in fact.lower() and "star" in fact.lower():
            score = 1.0
        
        self.fact_cache[fact] = score
        return score
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text"""
        entities = set()
        
        # Simple NER - look for capitalized words
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and i > 0:
                entities.add(word.strip('.,!?;:'))
                
                # Check for multi-word entities
                if i < len(words) - 1 and words[i + 1][0].isupper():
                    entities.add(f"{word} {words[i + 1]}".strip('.,!?;:'))
        
        return entities
    
    def _check_entity_accuracy(
        self,
        entities: Set[str],
        reference: Optional[str] = None
    ) -> float:
        """Check if entities are real/accurate"""
        if not entities:
            return 1.0
        
        valid_count = 0
        for entity in entities:
            # Check known entities
            if entity in self.entity_kb:
                valid_count += 1
            elif reference and entity.lower() in reference.lower():
                valid_count += 1
        
        return valid_count / len(entities)

class LogicalConsistencyMetric:
    """Evaluates logical consistency in model outputs"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        texts: List[str],
        reasoning_chains: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """Evaluate logical consistency"""
        results = {
            "consistency_scores": [],
            "contradictions": [],
            "logical_errors": [],
            "reasoning_validity": []
        }
        
        for i, text in enumerate(tqdm(texts, desc="Evaluating logical consistency")):
            # Check internal consistency
            consistency_score, contradictions = self._check_internal_consistency(text)
            results["consistency_scores"].append(consistency_score)
            results["contradictions"].extend(contradictions)
            
            # Check logical structure
            logical_errors = self._check_logical_structure(text)
            results["logical_errors"].extend(logical_errors)
            
            # Evaluate reasoning chain if provided
            if reasoning_chains and i < len(reasoning_chains):
                chain_validity = self._evaluate_reasoning_chain(reasoning_chains[i])
                results["reasoning_validity"].append(chain_validity)
        
        # Aggregate metrics
        results["mean_consistency"] = np.mean(results["consistency_scores"])
        results["contradiction_rate"] = len(results["contradictions"]) / len(texts)
        results["logical_error_rate"] = len(results["logical_errors"]) / len(texts)
        
        if results["reasoning_validity"]:
            results["mean_reasoning_validity"] = np.mean(results["reasoning_validity"])
        
        return results
    
    def _check_internal_consistency(self, text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Check for internal contradictions"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        contradictions = []
        
        # Check pairwise consistency
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if self._are_contradictory(sentences[i], sentences[j]):
                    contradictions.append({
                        "sentence1": sentences[i],
                        "sentence2": sentences[j],
                        "positions": (i, j)
                    })
        
        # Calculate consistency score
        if len(sentences) > 1:
            consistency_score = 1.0 - (len(contradictions) / (len(sentences) * (len(sentences) - 1) / 2))
        else:
            consistency_score = 1.0
        
        return consistency_score, contradictions
    
    def _are_contradictory(self, sent1: str, sent2: str) -> bool:
        """Check if two sentences are contradictory"""
        # Simple contradiction patterns
        negation_words = ["not", "never", "no", "none", "neither", "nor"]
        
        # Check for direct negation
        for neg in negation_words:
            if neg in sent1.lower() and neg not in sent2.lower():
                # Check if they're about the same subject
                words1 = set(sent1.lower().split())
                words2 = set(sent2.lower().split())
                common_words = words1 & words2
                if len(common_words) > 2:
                    return True
        
        # Check for opposite statements
        opposites = [
            ("increase", "decrease"), ("rise", "fall"), ("grow", "shrink"),
            ("always", "never"), ("all", "none"), ("true", "false")
        ]
        
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()
        
        for word1, word2 in opposites:
            if word1 in sent1_lower and word2 in sent2_lower:
                return True
            if word2 in sent1_lower and word1 in sent2_lower:
                return True
        
        return False
    
    def _check_logical_structure(self, text: str) -> List[Dict[str, Any]]:
        """Check for logical errors in structure"""
        errors = []
        
        # Check for circular reasoning
        sentences = text.split('.')
        for i, sent in enumerate(sentences):
            if "because" in sent.lower():
                parts = sent.lower().split("because")
                if len(parts) == 2:
                    # Check if conclusion appears in premise
                    if any(word in parts[1] for word in parts[0].split() if len(word) > 3):
                        errors.append({
                            "type": "circular_reasoning",
                            "sentence": sent,
                            "position": i
                        })
        
        # Check for non-sequiturs
        transition_words = ["therefore", "thus", "hence", "so", "consequently"]
        for i, sent in enumerate(sentences):
            for trans in transition_words:
                if trans in sent.lower() and i > 0:
                    # Check if conclusion relates to previous sentence
                    prev_words = set(sentences[i-1].lower().split())
                    curr_words = set(sent.lower().split())
                    if len(prev_words & curr_words) < 2:
                        errors.append({
                            "type": "non_sequitur",
                            "sentence": sent,
                            "position": i
                        })
        
        return errors
    
    def _evaluate_reasoning_chain(self, chain: List[str]) -> float:
        """Evaluate validity of reasoning chain"""
        if not chain or len(chain) < 2:
            return 0.0
        
        validity_scores = []
        
        # Check step transitions
        for i in range(1, len(chain)):
            prev_step = chain[i-1]
            curr_step = chain[i]
            
            # Check if steps are connected
            connection_score = self._measure_step_connection(prev_step, curr_step)
            validity_scores.append(connection_score)
        
        # Check overall coherence
        coherence = self._measure_chain_coherence(chain)
        
        return np.mean(validity_scores) * 0.7 + coherence * 0.3
    
    def _measure_step_connection(self, step1: str, step2: str) -> float:
        """Measure connection between reasoning steps"""
        # Simple word overlap measure
        words1 = set(step1.lower().split())
        words2 = set(step2.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at"}
        words1 = words1 - common_words
        words2 = words2 - common_words
        
        if not words1 or not words2:
            return 0.5
        
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return min(overlap * 2, 1.0)  # Scale up but cap at 1.0
    
    def _measure_chain_coherence(self, chain: List[str]) -> float:
        """Measure overall coherence of reasoning chain"""
        if len(chain) < 2:
            return 1.0
        
        # Check if chain progresses logically
        first_words = set(chain[0].lower().split())
        last_words = set(chain[-1].lower().split())
        
        # Should have some evolution from start to end
        overlap = len(first_words & last_words) / max(len(first_words), len(last_words))
        
        # Ideal overlap is moderate (not too high, not too low)
        coherence = 1.0 - abs(overlap - 0.3) * 2
        
        return max(0, coherence)

class CalibrationMetric:
    """Evaluates model calibration and confidence accuracy"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        predictions: List[str],
        confidence_scores: List[float],
        ground_truth: List[str],
        plot: bool = True
    ) -> Dict[str, Any]:
        """Evaluate calibration"""
        # Compute accuracy for each prediction
        accuracies = [
            1.0 if pred.lower().strip() == gt.lower().strip() else 0.0
            for pred, gt in zip(predictions, ground_truth)
        ]
        
        # Compute ECE (Expected Calibration Error)
        ece, bin_data = self._compute_ece(confidence_scores, accuracies)
        
        # Compute other calibration metrics
        mce = self._compute_mce(confidence_scores, accuracies)
        brier_score = self._compute_brier_score(confidence_scores, accuracies)
        
        # Reliability diagram
        if plot and self.config.plot_calibration:
            self._plot_reliability_diagram(bin_data)
        
        results = {
            "ece": ece,
            "mce": mce,
            "brier_score": brier_score,
            "bin_data": bin_data,
            "mean_confidence": np.mean(confidence_scores),
            "mean_accuracy": np.mean(accuracies),
            "overconfidence": np.mean(confidence_scores) - np.mean(accuracies)
        }
        
        return results
    
    def _compute_ece(
        self,
        confidences: List[float],
        accuracies: List[float]
    ) -> Tuple[float, List[Dict[str, float]]]:
        """Compute Expected Calibration Error"""
        n_bins = self.config.calibration_bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = [
                (conf, acc) for conf, acc in zip(confidences, accuracies)
                if bin_lower <= conf < bin_upper
            ]
            
            if in_bin:
                bin_confidence = np.mean([conf for conf, _ in in_bin])
                bin_accuracy = np.mean([acc for _, acc in in_bin])
                bin_count = len(in_bin)
                
                ece += (bin_count / len(confidences)) * abs(bin_accuracy - bin_confidence)
                
                bin_data.append({
                    "lower": bin_lower,
                    "upper": bin_upper,
                    "confidence": bin_confidence,
                    "accuracy": bin_accuracy,
                    "count": bin_count,
                    "gap": abs(bin_accuracy - bin_confidence)
                })
        
        return ece, bin_data
    
    def _compute_mce(self, confidences: List[float], accuracies: List[float]) -> float:
        """Compute Maximum Calibration Error"""
        _, bin_data = self._compute_ece(confidences, accuracies)
        
        if bin_data:
            return max(bd["gap"] for bd in bin_data)
        return 0.0
    
    def _compute_brier_score(self, confidences: List[float], accuracies: List[float]) -> float:
        """Compute Brier score"""
        return np.mean([(conf - acc) ** 2 for conf, acc in zip(confidences, accuracies)])
    
    def _plot_reliability_diagram(self, bin_data: List[Dict[str, float]]):
        """Plot reliability diagram"""
        if not bin_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        bin_centers = [(bd["lower"] + bd["upper"]) / 2 for bd in bin_data]
        bin_accuracies = [bd["accuracy"] for bd in bin_data]
        bin_counts = [bd["count"] for bd in bin_data]
        
        # Scale points by count
        sizes = [count / max(bin_counts) * 200 for count in bin_counts]
        
        ax1.scatter(bin_centers, bin_accuracies, s=sizes, alpha=0.7)
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Confidence')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.hist([bd["confidence"] for bd in bin_data for _ in range(bd["count"])],
                 bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "calibration_plot.png", dpi=150)
        plt.close()

class HallucinationDetector:
    """Detects hallucinations in model outputs"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.known_entities = self._load_known_entities()
    
    def _load_known_entities(self) -> Set[str]:
        """Load set of known entities"""
        # In practice, load from comprehensive KB
        return {
            "Albert Einstein", "Isaac Newton", "Charles Darwin",
            "United States", "Europe", "Asia", "Africa",
            "World War II", "Renaissance", "Industrial Revolution",
            # Add more
        }
    
    def evaluate(
        self,
        outputs: List[str],
        contexts: Optional[List[str]] = None,
        references: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect hallucinations in outputs"""
        results = {
            "hallucination_scores": [],
            "entity_hallucinations": [],
            "fact_hallucinations": [],
            "unsupported_claims": []
        }
        
        for i, output in enumerate(tqdm(outputs, desc="Detecting hallucinations")):
            context = contexts[i] if contexts else None
            reference = references[i] if references else None
            
            # Check entity hallucinations
            if self.config.check_entity_hallucination:
                entity_hall = self._detect_entity_hallucinations(output, context, reference)
                results["entity_hallucinations"].extend(entity_hall)
            
            # Check factual hallucinations
            if self.config.check_fact_hallucination:
                fact_hall = self._detect_fact_hallucinations(output, context, reference)
                results["fact_hallucinations"].extend(fact_hall)
            
            # Check unsupported claims
            unsupported = self._detect_unsupported_claims(output, context)
            results["unsupported_claims"].extend(unsupported)
            
            # Compute hallucination score
            hall_score = self._compute_hallucination_score(
                len(entity_hall), len(fact_hall), len(unsupported), len(output.split())
            )
            results["hallucination_scores"].append(hall_score)
        
        # Aggregate metrics
        results["mean_hallucination_score"] = np.mean(results["hallucination_scores"])
        results["hallucination_rate"] = np.mean([
            score > self.config.hallucination_threshold
            for score in results["hallucination_scores"]
        ])
        
        return results
    
    def _detect_entity_hallucinations(
        self,
        output: str,
        context: Optional[str] = None,
        reference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect hallucinated entities"""
        hallucinations = []
        
        # Extract entities from output
        entities = self._extract_entities(output)
        
        for entity in entities:
            # Check if entity is known
            if entity not in self.known_entities:
                # Check if mentioned in context or reference
                mentioned = False
                if context and entity.lower() in context.lower():
                    mentioned = True
                if reference and entity.lower() in reference.lower():
                    mentioned = True
                
                if not mentioned:
                    hallucinations.append({
                        "type": "entity",
                        "text": entity,
                        "confidence": 0.8  # High confidence it's hallucinated
                    })
        
        return hallucinations
    
    def _detect_fact_hallucinations(
        self,
        output: str,
        context: Optional[str] = None,
        reference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect hallucinated facts"""
        hallucinations = []
        
        # Extract quantitative claims
        number_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:percent|%)',
            r'(\d{4})\s*(?:year|AD|BC|CE|BCE)',
            r'(\d+(?:\.\d+)?)\s*(?:million|billion|thousand)',
            r'(\d+(?:\.\d+)?)\s*(?:meters|kilometers|miles|feet)',
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                # Check if number appears in context/reference
                if context and match not in context:
                    if reference and match not in reference:
                        hallucinations.append({
                            "type": "quantitative",
                            "text": match,
                            "pattern": pattern,
                            "confidence": 0.7
                        })
        
        return hallucinations
    
    def _detect_unsupported_claims(
        self,
        output: str,
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect claims not supported by context"""
        unsupported = []
        
        if not context:
            return unsupported
        
        # Extract claims from output
        claim_patterns = [
            r"(?:according to|studies show|research indicates)\s+([^.]+)",
            r"(?:it is known that|it has been proven that)\s+([^.]+)",
            r"(?:experts agree|scientists believe)\s+([^.]+)",
        ]
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                # Check if claim substance appears in context
                claim_words = set(match.lower().split())
                context_words = set(context.lower().split())
                
                overlap = len(claim_words & context_words) / len(claim_words)
                if overlap < 0.3:  # Less than 30% overlap
                    unsupported.append({
                        "type": "unsupported_claim",
                        "text": match,
                        "overlap": overlap,
                        "confidence": 1.0 - overlap
                    })
        
        return unsupported
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text"""
        entities = set()
        
        # Look for capitalized sequences
        words = text.split()
        i = 0
        while i < len(words):
            if words[i][0].isupper():
                entity = words[i]
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    entity += " " + words[j]
                    j += 1
                if len(entity.split()) <= 4:  # Reasonable entity length
                    entities.add(entity.strip('.,!?;:'))
                i = j
            else:
                i += 1
        
        return entities
    
    def _compute_hallucination_score(
        self,
        entity_count: int,
        fact_count: int,
        unsupported_count: int,
        total_words: int
    ) -> float:
        """Compute overall hallucination score"""
        # Normalize by text length
        normalized_count = (entity_count + fact_count + unsupported_count) / max(total_words / 100, 1)
        
        # Convert to 0-1 score (higher means more hallucination)
        score = min(normalized_count, 1.0)
        
        return score

class ReasoningQualityAssessor:
    """Assesses quality of reasoning in model outputs"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(
        self,
        reasoning_chains: List[List[str]],
        problems: Optional[List[str]] = None,
        solutions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate reasoning quality"""
        results = {
            "step_counts": [],
            "coherence_scores": [],
            "validity_scores": [],
            "efficiency_scores": [],
            "completeness_scores": []
        }
        
        for i, chain in enumerate(tqdm(reasoning_chains, desc="Assessing reasoning quality")):
            problem = problems[i] if problems else None
            solution = solutions[i] if solutions else None
            
            # Count steps
            step_count = len(chain)
            results["step_counts"].append(step_count)
            
            # Assess coherence
            coherence = self._assess_coherence(chain)
            results["coherence_scores"].append(coherence)
            
            # Assess validity
            validity = self._assess_validity(chain, problem, solution)
            results["validity_scores"].append(validity)
            
            # Assess efficiency
            efficiency = self._assess_efficiency(chain, step_count)
            results["efficiency_scores"].append(efficiency)
            
            # Assess completeness
            completeness = self._assess_completeness(chain, problem, solution)
            results["completeness_scores"].append(completeness)
        
        # Aggregate metrics
        results["mean_step_count"] = np.mean(results["step_counts"])
        results["mean_coherence"] = np.mean(results["coherence_scores"])
        results["mean_validity"] = np.mean(results["validity_scores"])
        results["mean_efficiency"] = np.mean(results["efficiency_scores"])
        results["mean_completeness"] = np.mean(results["completeness_scores"])
        
        # Overall reasoning quality
        results["overall_quality"] = np.mean([
            results["mean_coherence"],
            results["mean_validity"],
            results["mean_efficiency"],
            results["mean_completeness"]
        ])
        
        return results
    
    def _assess_coherence(self, chain: List[str]) -> float:
        """Assess coherence of reasoning chain"""
        if len(chain) < 2:
            return 0.0
        
        coherence_scores = []
        
        # Check step-to-step coherence
        for i in range(1, len(chain)):
            prev_step = chain[i-1]
            curr_step = chain[i]
            
            # Measure semantic similarity
            similarity = self._semantic_similarity(prev_step, curr_step)
            
            # Check for logical connectors
            has_connector = any(
                conn in curr_step.lower()
                for conn in ["therefore", "thus", "so", "because", "since", "then"]
            )
            
            coherence = similarity * 0.7 + (0.3 if has_connector else 0)
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)
    
    def _assess_validity(
        self,
        chain: List[str],
        problem: Optional[str] = None,
        solution: Optional[str] = None
    ) -> float:
        """Assess validity of reasoning"""
        validity_score = 0.5  # Base score
        
        # Check if chain addresses the problem
        if problem:
            problem_words = set(problem.lower().split())
            chain_words = set(" ".join(chain).lower().split())
            relevance = len(problem_words & chain_words) / len(problem_words)
            validity_score += relevance * 0.25
        
        # Check if chain leads to solution
        if solution and chain:
            last_step_words = set(chain[-1].lower().split())
            solution_words = set(solution.lower().split())
            conclusion_match = len(last_step_words & solution_words) / max(len(solution_words), 1)
            validity_score += conclusion_match * 0.25
        
        return min(validity_score, 1.0)
    
    def _assess_efficiency(self, chain: List[str], step_count: int) -> float:
        """Assess efficiency of reasoning"""
        # Check if steps are within reasonable bounds
        if step_count < self.config.min_reasoning_steps:
            return 0.5  # Too few steps
        elif step_count > self.config.max_reasoning_steps:
            return 0.5  # Too many steps
        
        # Check for redundancy
        unique_steps = len(set(chain))
        redundancy_ratio = unique_steps / step_count
        
        # Ideal is no redundancy but some steps might legitimately repeat
        efficiency = redundancy_ratio * 0.8 + 0.2
        
        return efficiency
    
    def _assess_completeness(
        self,
        chain: List[str],
        problem: Optional[str] = None,
        solution: Optional[str] = None
    ) -> float:
        """Assess completeness of reasoning"""
        completeness_score = 0.0
        
        # Check for problem understanding step
        if any("understand" in step.lower() or "given" in step.lower() for step in chain[:2]):
            completeness_score += 0.25
        
        # Check for strategy/approach step
        if any(
            keyword in step.lower()
            for step in chain
            for keyword in ["approach", "strategy", "method", "solve by"]
        ):
            completeness_score += 0.25
        
        # Check for verification step
        if any(
            keyword in step.lower()
            for step in chain[-2:]
            for keyword in ["check", "verify", "confirm", "therefore"]
        ):
            completeness_score += 0.25
        
        # Check for clear conclusion
        if solution and chain:
            if any(
                keyword in chain[-1].lower()
                for keyword in ["answer", "solution", "result", "conclude"]
            ):
                completeness_score += 0.25
        
        return completeness_score
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts"""
        # Simple word overlap measure
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union

class CrossDomainTransferEvaluator:
    """Evaluates model's ability to transfer reasoning across domains"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.domain_patterns = self._load_domain_patterns()
    
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load characteristic patterns for each domain"""
        return {
            "mathematics": ["equation", "solve", "calculate", "prove", "theorem"],
            "science": ["hypothesis", "experiment", "observe", "conclude", "data"],
            "history": ["event", "date", "period", "consequence", "significance"],
            "literature": ["theme", "character", "symbolism", "narrative", "meaning"],
            "code": ["function", "algorithm", "implement", "return", "variable"]
        }
    
    def evaluate(
        self,
        model_outputs: Dict[str, List[str]],
        source_domain: str,
        target_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate cross-domain transfer"""
        if target_domains is None:
            target_domains = [d for d in self.config.test_domains if d != source_domain]
        
        results = {
            "source_domain": source_domain,
            "transfer_scores": {},
            "pattern_preservation": {},
            "domain_adaptation": {}
        }
        
        # Get source domain performance as baseline
        source_outputs = model_outputs.get(source_domain, [])
        source_patterns = self._extract_reasoning_patterns(source_outputs, source_domain)
        
        # Evaluate transfer to each target domain
        for target_domain in target_domains:
            target_outputs = model_outputs.get(target_domain, [])
            
            if not target_outputs:
                continue
            
            # Measure transfer effectiveness
            transfer_score = self._measure_transfer(
                source_patterns,
                target_outputs,
                target_domain
            )
            results["transfer_scores"][target_domain] = transfer_score
            
            # Measure pattern preservation
            preservation = self._measure_pattern_preservation(
                source_patterns,
                target_outputs
            )
            results["pattern_preservation"][target_domain] = preservation
            
            # Measure domain adaptation
            adaptation = self._measure_domain_adaptation(
                target_outputs,
                target_domain
            )
            results["domain_adaptation"][target_domain] = adaptation
        
        # Overall transfer ability
        results["mean_transfer_score"] = np.mean(list(results["transfer_scores"].values()))
        results["transfer_variance"] = np.var(list(results["transfer_scores"].values()))
        
        return results
    
    def _extract_reasoning_patterns(
        self,
        outputs: List[str],
        domain: str
    ) -> Dict[str, float]:
        """Extract reasoning patterns from outputs"""
        patterns = {
            "step_structure": 0.0,
            "logical_flow": 0.0,
            "problem_decomposition": 0.0,
            "verification": 0.0
        }
        
        for output in outputs:
            # Check for step structure
            if any(marker in output for marker in ["Step 1", "First,", "1."]):
                patterns["step_structure"] += 1
            
            # Check for logical flow
            if any(conn in output.lower() for conn in ["therefore", "thus", "because"]):
                patterns["logical_flow"] += 1
            
            # Check for problem decomposition
            if any(phrase in output.lower() for phrase in ["break down", "parts", "components"]):
                patterns["problem_decomposition"] += 1
            
            # Check for verification
            if any(word in output.lower() for word in ["check", "verify", "confirm"]):
                patterns["verification"] += 1
        
        # Normalize
        for key in patterns:
            patterns[key] /= max(len(outputs), 1)
        
        return patterns
    
    def _measure_transfer(
        self,
        source_patterns: Dict[str, float],
        target_outputs: List[str],
        target_domain: str
    ) -> float:
        """Measure how well reasoning transfers to target domain"""
        # Extract patterns from target outputs
        target_patterns = self._extract_reasoning_patterns(target_outputs, target_domain)
        
        # Compare pattern similarity
        pattern_similarity = 0.0
        for key in source_patterns:
            if key in target_patterns:
                # Similarity based on how close the pattern frequencies are
                diff = abs(source_patterns[key] - target_patterns[key])
                similarity = 1.0 - diff
                pattern_similarity += similarity
        
        pattern_similarity /= len(source_patterns)
        
        # Check if target domain characteristics are preserved
        domain_score = self._check_domain_characteristics(target_outputs, target_domain)
        
        # Combined score
        transfer_score = pattern_similarity * 0.6 + domain_score * 0.4
        
        return transfer_score
    
    def _measure_pattern_preservation(
        self,
        source_patterns: Dict[str, float],
        target_outputs: List[str]
    ) -> float:
        """Measure how well reasoning patterns are preserved"""
        preserved_count = 0
        total_patterns = len(source_patterns)
        
        for pattern, source_freq in source_patterns.items():
            if source_freq > 0.5:  # Pattern is significant in source
                # Check if pattern appears in target
                target_freq = self._count_pattern_frequency(pattern, target_outputs)
                if target_freq > 0.3:  # Pattern is reasonably preserved
                    preserved_count += 1
        
        return preserved_count / total_patterns if total_patterns > 0 else 0.0
    
    def _measure_domain_adaptation(
        self,
        outputs: List[str],
        domain: str
    ) -> float:
        """Measure how well outputs adapt to target domain"""
        domain_keywords = self.domain_patterns.get(domain, [])
        
        keyword_scores = []
        for output in outputs:
            output_lower = output.lower()
            keyword_count = sum(1 for keyword in domain_keywords if keyword in output_lower)
            keyword_score = keyword_count / len(domain_keywords) if domain_keywords else 0
            keyword_scores.append(keyword_score)
        
        return np.mean(keyword_scores) if keyword_scores else 0.0
    
    def _check_domain_characteristics(
        self,
        outputs: List[str],
        domain: str
    ) -> float:
        """Check if outputs have appropriate domain characteristics"""
        characteristics = {
            "mathematics": lambda o: any(char in o for char in ["=", "+", "-", "x", "∫"]),
            "science": lambda o: any(word in o.lower() for word in ["data", "hypothesis", "experiment"]),
            "history": lambda o: bool(re.search(r'\b\d{4}\b', o)),  # Years
            "literature": lambda o: any(word in o.lower() for word in ["theme", "character", "symbolism"]),
            "code": lambda o: any(char in o for char in ["()", "{}", "[]", "def", "function"])
        }
        
        if domain not in characteristics:
            return 0.5
        
        check_func = characteristics[domain]
        scores = [1.0 if check_func(output) else 0.0 for output in outputs]
        
        return np.mean(scores) if scores else 0.0
    
    def _count_pattern_frequency(self, pattern: str, outputs: List[str]) -> float:
        """Count frequency of a reasoning pattern in outputs"""
        count = 0
        
        pattern_indicators = {
            "step_structure": ["step", "first", "second", "then", "finally"],
            "logical_flow": ["therefore", "thus", "hence", "because", "since"],
            "problem_decomposition": ["break", "decompose", "parts", "components", "divide"],
            "verification": ["check", "verify", "confirm", "validate", "test"]
        }
        
        indicators = pattern_indicators.get(pattern, [])
        
        for output in outputs:
            output_lower = output.lower()
            if any(indicator in output_lower for indicator in indicators):
                count += 1
        
        return count / len(outputs) if outputs else 0.0

class ComprehensiveEvaluator:
    """Main evaluator combining all metrics"""
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize individual evaluators
        self.factual_evaluator = FactualAccuracyMetric(config)
        self.consistency_evaluator = LogicalConsistencyMetric(config)
        self.calibration_evaluator = CalibrationMetric(config)
        self.hallucination_detector = HallucinationDetector(config)
        self.reasoning_evaluator = ReasoningQualityAssessor(config)
        self.transfer_evaluator = CrossDomainTransferEvaluator(config)
        
        # Results storage
        self.results = {}
    
    def evaluate_all(
        self,
        model_outputs: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        all_results = {}
        
        # Factual accuracy
        if self.config.fact_check_enabled:
            logger.info("Evaluating factual accuracy...")
            all_results["factual_accuracy"] = self.factual_evaluator.evaluate(
                model_outputs.get("predictions", []),
                model_outputs.get("references"),
                model_outputs.get("contexts")
            )
        
        # Logical consistency
        if self.config.consistency_check_enabled:
            logger.info("Evaluating logical consistency...")
            all_results["logical_consistency"] = self.consistency_evaluator.evaluate(
                model_outputs.get("predictions", []),
                model_outputs.get("reasoning_chains")
            )
        
        # Calibration
        if "confidence_scores" in model_outputs and ground_truth:
            logger.info("Evaluating calibration...")
            all_results["calibration"] = self.calibration_evaluator.evaluate(
                model_outputs["predictions"],
                model_outputs["confidence_scores"],
                ground_truth["labels"],
                plot=self.config.plot_calibration
            )
        
        # Hallucination detection
        logger.info("Detecting hallucinations...")
        all_results["hallucination"] = self.hallucination_detector.evaluate(
            model_outputs.get("predictions", []),
            model_outputs.get("contexts"),
            model_outputs.get("references")
        )
        
        # Reasoning quality
        if "reasoning_chains" in model_outputs:
            logger.info("Assessing reasoning quality...")
            all_results["reasoning_quality"] = self.reasoning_evaluator.evaluate(
                model_outputs["reasoning_chains"],
                model_outputs.get("problems"),
                model_outputs.get("solutions")
            )
        
        # Cross-domain transfer
        if "domain_outputs" in model_outputs:
            logger.info("Evaluating cross-domain transfer...")
            all_results["transfer"] = self.transfer_evaluator.evaluate(
                model_outputs["domain_outputs"],
                model_outputs.get("source_domain", "mathematics")
            )
        
        # Compute overall scores
        all_results["overall"] = self._compute_overall_scores(all_results)
        
        # Save results
        if save_results and self.config.save_results:
            self._save_results(all_results)
        
        # Generate report
        if self.config.generate_report:
            self._generate_report(all_results)
        
        self.results = all_results
        return all_results
    
    def _compute_overall_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall evaluation scores"""
        overall = {}
        
        # Collect all scores
        score_components = []
        
        if "factual_accuracy" in results:
            score_components.append(("factual", results["factual_accuracy"]["mean_accuracy"]))
        
        if "logical_consistency" in results:
            score_components.append(("consistency", results["logical_consistency"]["mean_consistency"]))
        
        if "calibration" in results:
            score_components.append(("calibration", 1.0 - results["calibration"]["ece"]))
        
        if "hallucination" in results:
            score_components.append(("hallucination", 1.0 - results["hallucination"]["mean_hallucination_score"]))
        
        if "reasoning_quality" in results:
            score_components.append(("reasoning", results["reasoning_quality"]["overall_quality"]))
        
        if "transfer" in results:
            score_components.append(("transfer", results["transfer"]["mean_transfer_score"]))
        
        # Compute weighted average
        if score_components:
            weights = {
                "factual": 2.0,
                "consistency": 1.5,
                "calibration": 1.0,
                "hallucination": 2.0,
                "reasoning": 1.5,
                "transfer": 1.0
            }
            
            weighted_sum = sum(score * weights.get(name, 1.0) for name, score in score_components)
            total_weight = sum(weights.get(name, 1.0) for name, _ in score_components)
            
            overall["weighted_score"] = weighted_sum / total_weight
            overall["components"] = dict(score_components)
        
        return overall
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        with open(output_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as text
        with open(output_dir / "evaluation_summary.txt", 'w') as f:
            f.write("Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            if "overall" in results:
                f.write(f"Overall Weighted Score: {results['overall']['weighted_score']:.3f}\n\n")
                f.write("Component Scores:\n")
                for name, score in results['overall']['components'].items():
                    f.write(f"  {name}: {score:.3f}\n")
        
        logger.info(f"Results saved to {output_dir}")
    
    def _generate_report(self, results: Dict[str, Any]):
        """Generate comprehensive evaluation report"""
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create report
        report_lines = [
            "# Comprehensive Evaluation Report",
            "",
            f"## Overall Score: {results.get('overall', {}).get('weighted_score', 0):.3f}",
            "",
            "## Detailed Results",
            ""
        ]
        
        # Add section for each metric
        if "factual_accuracy" in results:
            report_lines.extend([
                "### Factual Accuracy",
                f"- Mean Accuracy: {results['factual_accuracy']['mean_accuracy']:.3f}",
                f"- Fact Verification Rate: {results['factual_accuracy']['fact_verification_rate']:.3f}",
                ""
            ])
        
        if "logical_consistency" in results:
            report_lines.extend([
                "### Logical Consistency",
                f"- Mean Consistency: {results['logical_consistency']['mean_consistency']:.3f}",
                f"- Contradiction Rate: {results['logical_consistency']['contradiction_rate']:.3f}",
                ""
            ])
        
        if "calibration" in results:
            report_lines.extend([
                "### Calibration",
                f"- ECE: {results['calibration']['ece']:.3f}",
                f"- MCE: {results['calibration']['mce']:.3f}",
                f"- Overconfidence: {results['calibration']['overconfidence']:.3f}",
                ""
            ])
        
        if "hallucination" in results:
            report_lines.extend([
                "### Hallucination Detection",
                f"- Mean Hallucination Score: {results['hallucination']['mean_hallucination_score']:.3f}",
                f"- Hallucination Rate: {results['hallucination']['hallucination_rate']:.3f}",
                ""
            ])
        
        if "reasoning_quality" in results:
            report_lines.extend([
                "### Reasoning Quality",
                f"- Overall Quality: {results['reasoning_quality']['overall_quality']:.3f}",
                f"- Mean Coherence: {results['reasoning_quality']['mean_coherence']:.3f}",
                f"- Mean Validity: {results['reasoning_quality']['mean_validity']:.3f}",
                ""
            ])
        
        # Write report
        with open(output_dir / "evaluation_report.md", 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Report generated at {output_dir / 'evaluation_report.md'}")