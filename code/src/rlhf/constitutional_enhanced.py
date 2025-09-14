"""
Enhanced Constitutional AI Implementation
Extends base constitutional AI with iterative refinement and advanced augmentation
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import json
from pathlib import Path
import re
from collections import defaultdict
from LLM.src.utils.path_utils import get_data_dir

logger = logging.getLogger(__name__)

@dataclass
class ConstitutionalConfig:
    """Configuration for Enhanced Constitutional AI"""
    # Base principles
    principles_path: Optional[str] = None
    use_default_principles: bool = True
    
    # Generation settings
    max_critique_length: int = 300
    max_revision_length: int = 600
    critique_temperature: float = 0.6
    revision_temperature: float = 0.7
    
    # Iterative refinement
    max_refinement_iterations: int = 3
    refinement_threshold: float = 0.8
    use_self_critique: bool = True
    
    # Training settings
    num_critiques_per_response: int = 3
    num_revisions_per_critique: int = 2
    constitutional_batch_size: int = 8
    
    # Data augmentation
    use_adversarial_augmentation: bool = True
    adversarial_ratio: float = 0.2
    paraphrase_augmentation: bool = True
    difficulty_progression: bool = True
    
    # Scoring and filtering
    min_improvement_threshold: float = 0.1
    critique_quality_threshold: float = 0.6
    use_reward_model_scoring: bool = True
    
    # Output settings
    save_constitutional_examples: bool = True
    examples_save_path: str = None
    
    def __post_init__(self):
        if self.examples_save_path is None:
            self.examples_save_path = os.path.join(get_data_dir(), "constitutional_examples")

class IterativeRefinement(nn.Module):
    """Iterative refinement module for constitutional improvements"""
    def __init__(self, config: ConstitutionalConfig):
        super().__init__()
        self.config = config
        
        # Quality assessment network
        self.quality_assessor = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 1),
            nn.Sigmoid()
        )
        
        # Improvement predictor
        self.improvement_predictor = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Tanh()  # -1 to 1 improvement score
        )
    
    def assess_quality(self, text_embedding: torch.Tensor) -> float:
        """Assess quality of text"""
        quality_score = self.quality_assessor(text_embedding)
        return quality_score.item()
    
    def predict_improvement(
        self,
        original_embedding: torch.Tensor,
        revised_embedding: torch.Tensor
    ) -> float:
        """Predict improvement from original to revised"""
        combined = torch.cat([original_embedding, revised_embedding], dim=-1)
        improvement = self.improvement_predictor(combined)
        return improvement.item()
    
    def should_continue_refinement(
        self,
        current_quality: float,
        improvement_trend: List[float],
        iteration: int
    ) -> bool:
        """Decide whether to continue refinement"""
        # Check iteration limit
        if iteration >= self.config.max_refinement_iterations:
            return False
        
        # Check quality threshold
        if current_quality >= self.config.refinement_threshold:
            return False
        
        # Check improvement trend
        if len(improvement_trend) >= 2:
            recent_improvement = np.mean(improvement_trend[-2:])
            if recent_improvement < self.config.min_improvement_threshold:
                return False
        
        return True

class ConstitutionalDataAugmentor:
    """Augments constitutional training data with various strategies"""
    def __init__(self, config: ConstitutionalConfig):
        self.config = config
        
        # Adversarial templates
        self.adversarial_templates = [
            "While this seems helpful, {}",
            "Although generally correct, {}",
            "This response could be improved by {}",
            "A potential issue is {}",
            "To be more precise, {}",
        ]
        
        # Difficulty levels for progression
        self.difficulty_levels = {
            "easy": {
                "ambiguity": 0.1,
                "complexity": 0.3,
                "edge_cases": 0.1
            },
            "medium": {
                "ambiguity": 0.3,
                "complexity": 0.5,
                "edge_cases": 0.3
            },
            "hard": {
                "ambiguity": 0.5,
                "complexity": 0.8,
                "edge_cases": 0.5
            }
        }
    
    def augment_constitutional_example(
        self,
        example: Dict[str, Any],
        difficulty: str = "medium"
    ) -> List[Dict[str, Any]]:
        """Augment a single constitutional example"""
        augmented = [example]  # Original
        
        # Adversarial augmentation
        if self.config.use_adversarial_augmentation:
            adversarial = self._create_adversarial_example(example, difficulty)
            if adversarial:
                augmented.append(adversarial)
        
        # Paraphrase augmentation
        if self.config.paraphrase_augmentation:
            paraphrased = self._create_paraphrased_example(example)
            if paraphrased:
                augmented.append(paraphrased)
        
        # Edge case augmentation
        edge_case = self._create_edge_case_example(example, difficulty)
        if edge_case:
            augmented.append(edge_case)
        
        return augmented
    
    def _create_adversarial_example(
        self,
        example: Dict[str, Any],
        difficulty: str
    ) -> Optional[Dict[str, Any]]:
        """Create adversarial example that tests constitutional boundaries"""
        original_response = example["response"]
        
        # Add subtle problematic content based on difficulty
        difficulty_params = self.difficulty_levels[difficulty]
        
        if np.random.random() < difficulty_params["ambiguity"]:
            # Add ambiguous statement
            adversarial_addition = "It's worth noting that some might interpret this differently."
        elif np.random.random() < difficulty_params["edge_cases"]:
            # Add edge case
            adversarial_addition = "In certain extreme circumstances, the opposite might be true."
        else:
            # Add complexity
            adversarial_addition = "This becomes more complex when considering all possible scenarios."
        
        template = np.random.choice(self.adversarial_templates)
        adversarial_response = f"{original_response} {template.format(adversarial_addition)}"
        
        return {
            "prompt": example["prompt"],
            "response": adversarial_response,
            "original_response": original_response,
            "augmentation_type": "adversarial",
            "difficulty": difficulty,
            "requires_revision": True
        }
    
    def _create_paraphrased_example(
        self,
        example: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create paraphrased version maintaining meaning"""
        # Simple paraphrasing rules
        paraphrase_rules = {
            "I think": "In my opinion",
            "However": "Nevertheless",
            "Therefore": "As a result",
            "For example": "For instance",
            "should": "ought to",
            "must": "need to",
        }
        
        paraphrased = example["response"]
        for original, replacement in paraphrase_rules.items():
            if original in paraphrased:
                paraphrased = paraphrased.replace(original, replacement, 1)
        
        # Only return if actually changed
        if paraphrased != example["response"]:
            return {
                "prompt": example["prompt"],
                "response": paraphrased,
                "original_response": example["response"],
                "augmentation_type": "paraphrase",
                "requires_revision": example.get("requires_revision", False)
            }
        
        return None
    
    def _create_edge_case_example(
        self,
        example: Dict[str, Any],
        difficulty: str
    ) -> Optional[Dict[str, Any]]:
        """Create edge case example"""
        # Modify prompt to be more challenging
        edge_case_modifiers = {
            "easy": "What if we consider a special case where",
            "medium": "In an unusual scenario where",
            "hard": "Consider an extreme edge case where"
        }
        
        modifier = edge_case_modifiers.get(difficulty, edge_case_modifiers["medium"])
        edge_case_prompt = f"{modifier} {example['prompt']}"
        
        return {
            "prompt": edge_case_prompt,
            "response": example["response"],
            "original_prompt": example["prompt"],
            "augmentation_type": "edge_case",
            "difficulty": difficulty,
            "requires_revision": True
        }

class EnhancedConstitutionalAI:
    """Enhanced Constitutional AI with iterative refinement and augmentation"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ConstitutionalConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Load principles
        self.principles = self._load_principles()
        
        # Initialize components
        self.iterative_refiner = IterativeRefinement(config)
        self.data_augmentor = ConstitutionalDataAugmentor(config)
        
        # Move to device
        device = next(model.parameters()).device
        self.iterative_refiner.to(device)
        
        # Track statistics
        self.stats = defaultdict(list)
    
    def _load_principles(self) -> List[Dict[str, Any]]:
        """Load constitutional principles"""
        principles = []
        
        if self.config.use_default_principles:
            # Default principles
            principles.extend([
                {
                    "name": "helpfulness",
                    "description": "Be genuinely helpful and provide accurate, useful information",
                    "critique_prompt": "Is this response genuinely helpful and accurate?",
                    "revision_prompt": "Revise to be more helpful and accurate."
                },
                {
                    "name": "harmlessness",
                    "description": "Avoid content that could cause harm",
                    "critique_prompt": "Could this response potentially cause harm?",
                    "revision_prompt": "Revise to remove any potentially harmful content."
                },
                {
                    "name": "honesty",
                    "description": "Be honest and acknowledge limitations",
                    "critique_prompt": "Is this response honest about uncertainties and limitations?",
                    "revision_prompt": "Revise to be more honest about uncertainties."
                },
                {
                    "name": "clarity",
                    "description": "Be clear and easy to understand",
                    "critique_prompt": "Is this response clear and well-structured?",
                    "revision_prompt": "Revise to improve clarity and structure."
                }
            ])
        
        # Load custom principles if provided
        if self.config.principles_path:
            with open(self.config.principles_path, 'r') as f:
                custom_principles = json.load(f)
                principles.extend(custom_principles)
        
        return principles
    
    def generate_training_data(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate constitutional training data with enhancements"""
        training_data = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating constitutional data")):
            # Generate or use provided response
            if responses and i < len(responses):
                initial_response = responses[i]
            else:
                initial_response = self._generate_response(prompt)
            
            # Generate constitutional examples
            examples = self._generate_constitutional_examples(
                prompt,
                initial_response
            )
            
            # Augment examples if configured
            if self.config.difficulty_progression:
                # Determine difficulty based on progress
                progress = i / len(prompts)
                if progress < 0.33:
                    difficulty = "easy"
                elif progress < 0.67:
                    difficulty = "medium"
                else:
                    difficulty = "hard"
            else:
                difficulty = "medium"
            
            augmented_examples = []
            for example in examples:
                augmented = self.data_augmentor.augment_constitutional_example(
                    example,
                    difficulty
                )
                augmented_examples.extend(augmented)
            
            training_data.extend(augmented_examples)
        
        # Filter and score examples
        filtered_data = self._filter_and_score_examples(training_data)
        
        # Save examples if configured
        if self.config.save_constitutional_examples:
            self._save_examples(filtered_data)
        
        return filtered_data
    
    def _generate_response(self, prompt: str) -> str:
        """Generate initial response"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_revision_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response[len(prompt):].strip()
        
        return response
    
    def _generate_constitutional_examples(
        self,
        prompt: str,
        initial_response: str
    ) -> List[Dict[str, Any]]:
        """Generate constitutional training examples"""
        examples = []
        
        # Apply each principle
        for principle in self.principles[:self.config.num_critiques_per_response]:
            # Generate critique
            critique = self._generate_critique(prompt, initial_response, principle)
            
            # Iterative refinement
            if self.config.use_self_critique:
                refined_response, refinement_history = self._iterative_refinement(
                    prompt,
                    initial_response,
                    critique,
                    principle
                )
            else:
                # Single revision
                refined_response = self._generate_revision(
                    prompt,
                    initial_response,
                    critique,
                    principle
                )
                refinement_history = []
            
            # Assess quality improvement
            is_better = self._assess_improvement(
                initial_response,
                refined_response,
                principle
            )
            
            example = {
                "prompt": prompt,
                "original_response": initial_response,
                "critique": critique,
                "revision": refined_response,
                "principle": principle["name"],
                "is_revision_better": is_better,
                "refinement_iterations": len(refinement_history),
                "refinement_history": refinement_history
            }
            
            examples.append(example)
        
        return examples
    
    def _generate_critique(
        self,
        prompt: str,
        response: str,
        principle: Dict[str, Any]
    ) -> str:
        """Generate critique based on principle"""
        critique_prompt = f"""
Principle: {principle['name']}
Description: {principle['description']}

User prompt: {prompt}
Assistant response: {response}

Task: {principle['critique_prompt']}
Provide a specific, constructive critique:
"""
        
        inputs = self.tokenizer(
            critique_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_critique_length,
                temperature=self.config.critique_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        critique = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        critique = critique[len(critique_prompt):].strip()
        
        return critique
    
    def _generate_revision(
        self,
        prompt: str,
        response: str,
        critique: str,
        principle: Dict[str, Any]
    ) -> str:
        """Generate single revision based on critique"""
        revision_prompt = f"""
Original prompt: {prompt}
Original response: {response}

Critique based on {principle['name']}: {critique}

Task: {principle['revision_prompt']}
Provide an improved response:
"""
        
        inputs = self.tokenizer(
            revision_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_revision_length,
                temperature=self.config.revision_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        revision = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        revision = revision[len(revision_prompt):].strip()
        
        return revision
    
    def _iterative_refinement(
        self,
        prompt: str,
        initial_response: str,
        initial_critique: str,
        principle: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Iteratively refine response"""
        current_response = initial_response
        current_critique = initial_critique
        refinement_history = []
        improvement_trend = []
        
        for iteration in range(self.config.max_refinement_iterations):
            # Generate revision
            revised_response = self._generate_revision(
                prompt,
                current_response,
                current_critique,
                principle
            )
            
            # Assess quality
            current_quality = self._assess_response_quality(revised_response)
            
            # Predict improvement
            if hasattr(self, '_get_embedding'):
                original_emb = self._get_embedding(current_response)
                revised_emb = self._get_embedding(revised_response)
                improvement = self.iterative_refiner.predict_improvement(
                    original_emb,
                    revised_emb
                )
            else:
                # Simple heuristic
                improvement = 0.1 if len(revised_response) > len(current_response) else 0.05
            
            improvement_trend.append(improvement)
            
            # Record history
            refinement_history.append({
                "iteration": iteration,
                "response": revised_response,
                "critique": current_critique,
                "quality": current_quality,
                "improvement": improvement
            })
            
            # Check if should continue
            if not self.iterative_refiner.should_continue_refinement(
                current_quality,
                improvement_trend,
                iteration
            ):
                break
            
            # Generate new critique for next iteration
            current_response = revised_response
            current_critique = self._generate_critique(
                prompt,
                current_response,
                principle
            )
        
        return current_response, refinement_history
    
    def _assess_response_quality(self, response: str) -> float:
        """Assess quality of response"""
        # Simple heuristics
        quality_score = 0.5
        
        # Length appropriateness
        word_count = len(response.split())
        if 50 <= word_count <= 500:
            quality_score += 0.1
        
        # Structure indicators
        if any(marker in response for marker in ["First", "Second", "Finally"]):
            quality_score += 0.1
        
        # Clarity indicators
        if "." in response and "?" not in response[-10:]:
            quality_score += 0.1
        
        # Completeness
        if response.endswith(".") and len(response) > 100:
            quality_score += 0.1
        
        # No obvious errors
        if not any(error in response.lower() for error in ["i don't know", "error", "sorry"]):
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _assess_improvement(
        self,
        original: str,
        revised: str,
        principle: Dict[str, Any]
    ) -> bool:
        """Assess if revision is better than original"""
        # Principle-specific checks
        if principle["name"] == "helpfulness":
            # Check if more informative
            return len(revised.split()) > len(original.split()) * 0.8
        
        elif principle["name"] == "harmlessness":
            # Check if removed problematic content
            harmful_words = ["dangerous", "harmful", "risky", "unsafe"]
            original_harmful = sum(1 for word in harmful_words if word in original.lower())
            revised_harmful = sum(1 for word in harmful_words if word in revised.lower())
            return revised_harmful < original_harmful
        
        elif principle["name"] == "honesty":
            # Check for uncertainty acknowledgment
            uncertainty_phrases = ["might", "could", "possibly", "uncertain", "not sure"]
            return any(phrase in revised.lower() for phrase in uncertainty_phrases)
        
        elif principle["name"] == "clarity":
            # Check structure improvement
            return revised.count(".") >= original.count(".") and len(revised) < len(original) * 1.5
        
        # Default: consider improved if revised
        return revised != original
    
    def _filter_and_score_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and score constitutional examples"""
        filtered = []
        
        for example in examples:
            # Score critique quality
            critique_score = self._score_critique(example["critique"])
            
            # Check improvement threshold
            if example["is_revision_better"]:
                # Estimate improvement magnitude
                improvement_score = self._estimate_improvement(
                    example["original_response"],
                    example["revision"]
                )
                
                if improvement_score >= self.config.min_improvement_threshold:
                    example["critique_score"] = critique_score
                    example["improvement_score"] = improvement_score
                    filtered.append(example)
        
        # Sort by quality
        filtered.sort(key=lambda x: x["improvement_score"], reverse=True)
        
        return filtered
    
    def _score_critique(self, critique: str) -> float:
        """Score critique quality"""
        score = 0.5
        
        # Specificity
        if len(critique.split()) > 20:
            score += 0.2
        
        # Constructive language
        constructive_words = ["could", "should", "improve", "better", "consider"]
        if any(word in critique.lower() for word in constructive_words):
            score += 0.2
        
        # Actionable
        if any(phrase in critique.lower() for phrase in ["try to", "make sure", "ensure"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _estimate_improvement(self, original: str, revised: str) -> float:
        """Estimate improvement magnitude"""
        # Simple estimation based on changes
        if original == revised:
            return 0.0
        
        # Character-level difference ratio
        char_diff = abs(len(revised) - len(original)) / max(len(original), 1)
        
        # Word-level difference
        original_words = set(original.lower().split())
        revised_words = set(revised.lower().split())
        word_diff = len(revised_words - original_words) / max(len(original_words), 1)
        
        # Combine metrics
        improvement = min(char_diff * 0.3 + word_diff * 0.7, 1.0)
        
        return improvement
    
    def _save_examples(self, examples: List[Dict[str, Any]]):
        """Save constitutional examples"""
        output_dir = Path(self.config.examples_save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        output_file = output_dir / f"constitutional_examples_{len(examples)}.jsonl"
        
        with open(output_file, 'w') as f:
            for example in examples:
                # Remove non-serializable data
                clean_example = {
                    k: v for k, v in example.items()
                    if not isinstance(v, (torch.Tensor, nn.Module))
                }
                f.write(json.dumps(clean_example) + '\n')
        
        logger.info(f"Saved {len(examples)} constitutional examples to {output_file}")
    
    def evaluate_adherence(
        self,
        test_prompts: List[str]
    ) -> Dict[str, float]:
        """Evaluate model's adherence to constitutional principles"""
        adherence_scores = defaultdict(list)
        
        for prompt in test_prompts:
            # Generate response
            response = self._generate_response(prompt)
            
            # Evaluate against each principle
            for principle in self.principles:
                critique = self._generate_critique(prompt, response, principle)
                
                # Score based on critique severity
                severity = self._assess_critique_severity(critique)
                adherence = 1.0 - severity
                adherence_scores[principle["name"]].append(adherence)
        
        # Aggregate scores
        results = {}
        for principle_name, scores in adherence_scores.items():
            results[f"adherence/{principle_name}"] = np.mean(scores)
        
        results["overall_adherence"] = np.mean([
            np.mean(scores) for scores in adherence_scores.values()
        ])
        
        return results
    
    def _assess_critique_severity(self, critique: str) -> float:
        """Assess severity of critique (0-1, higher is more severe)"""
        severity = 0.3  # Base severity
        
        # Severe issue indicators
        severe_words = ["harmful", "dangerous", "misleading", "wrong", "false"]
        for word in severe_words:
            if word in critique.lower():
                severity += 0.1
        
        # Moderate issue indicators
        moderate_words = ["unclear", "confusing", "incomplete", "vague"]
        for word in moderate_words:
            if word in critique.lower():
                severity += 0.05
        
        return min(severity, 1.0)
    
    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for constitutional training"""
        # This is a placeholder - actual implementation would compute
        # appropriate loss based on constitutional objectives
        
        # For now, return a dummy loss
        return torch.tensor(0.0, requires_grad=True)