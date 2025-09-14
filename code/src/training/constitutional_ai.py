"""
Constitutional AI Training
Implements self-supervised harmlessness training with AI feedback
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
from LLM.src.utils.path_utils import get_data_dir

logger = logging.getLogger(__name__)

@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle"""
    name: str
    description: str
    critique_prompt: str
    revision_prompt: str
    categories: List[str] = field(default_factory=list)
    severity: float = 1.0  # Weight for this principle

# Default constitutional principles
DEFAULT_PRINCIPLES = [
    ConstitutionalPrinciple(
        name="harmlessness",
        description="The AI should not provide information that could be used to harm individuals or groups",
        critique_prompt="Identify any ways in which the response could enable harm to individuals or groups. Consider physical, emotional, financial, or reputational harm.",
        revision_prompt="Revise the response to remove any content that could enable harm while maintaining helpfulness.",
        categories=["safety", "harm-prevention"],
        severity=2.0,
    ),
    ConstitutionalPrinciple(
        name="honesty",
        description="The AI should be honest and transparent about its limitations",
        critique_prompt="Identify any misleading statements, overconfidence, or failure to acknowledge uncertainty in the response.",
        revision_prompt="Revise the response to be more honest and transparent about limitations and uncertainties.",
        categories=["truthfulness", "transparency"],
        severity=1.5,
    ),
    ConstitutionalPrinciple(
        name="helpfulness",
        description="The AI should be genuinely helpful and provide useful information",
        critique_prompt="Evaluate whether the response genuinely addresses the user's needs and provides actionable information.",
        revision_prompt="Revise the response to be more helpful and directly address the user's query.",
        categories=["utility", "relevance"],
        severity=1.0,
    ),
    ConstitutionalPrinciple(
        name="respect",
        description="The AI should be respectful and avoid discriminatory or offensive content",
        critique_prompt="Identify any disrespectful, discriminatory, or offensive content in the response.",
        revision_prompt="Revise the response to be respectful and inclusive of all individuals and groups.",
        categories=["fairness", "respect"],
        severity=1.5,
    ),
    ConstitutionalPrinciple(
        name="privacy",
        description="The AI should respect privacy and not request or reveal personal information",
        critique_prompt="Check if the response requests unnecessary personal information or could compromise privacy.",
        revision_prompt="Revise the response to respect privacy and avoid requesting or revealing personal information.",
        categories=["privacy", "data-protection"],
        severity=1.5,
    ),
]

@dataclass
class ConstitutionalAIConfig:
    """Configuration for Constitutional AI training"""
    # Principles
    principles: List[ConstitutionalPrinciple] = field(default_factory=lambda: DEFAULT_PRINCIPLES)
    custom_principles_path: Optional[str] = None
    
    # Generation settings
    max_critique_length: int = 256
    max_revision_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Training settings
    num_critiques_per_response: int = 3
    num_revisions_per_critique: int = 2
    critique_batch_size: int = 8
    revision_batch_size: int = 4
    
    # Scoring
    use_critique_scoring: bool = True
    critique_score_threshold: float = 0.5
    use_principle_weighting: bool = True
    
    # Self-supervision
    num_self_critique_iterations: int = 2
    self_critique_temperature: float = 0.5
    
    # Data augmentation
    augment_with_paraphrasing: bool = True
    augment_with_adversarial: bool = True
    num_paraphrases: int = 2
    
    # Output
    save_constitutional_data: bool = True
    constitutional_data_path: str = None
    
    def __post_init__(self):
        if self.constitutional_data_path is None:
            self.constitutional_data_path = os.path.join(get_data_dir(), "constitutional")

class ConstitutionalCritique:
    """Generates critiques based on constitutional principles"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ConstitutionalAIConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device
        
        # Load custom principles if provided
        if config.custom_principles_path:
            self.principles = self._load_custom_principles(config.custom_principles_path)
        else:
            self.principles = config.principles
    
    def _load_custom_principles(self, path: str) -> List[ConstitutionalPrinciple]:
        """Load custom constitutional principles from file"""
        principles = []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for p in data:
            principle = ConstitutionalPrinciple(**p)
            principles.append(principle)
        
        logger.info(f"Loaded {len(principles)} custom constitutional principles")
        return principles
    
    def generate_critique(
        self,
        response: str,
        principle: ConstitutionalPrinciple,
        context: Optional[str] = None,
    ) -> str:
        """Generate critique for a response based on a principle"""
        # Construct critique prompt
        prompt = f"""You are reviewing an AI assistant's response for adherence to important principles.

Principle: {principle.name}
Description: {principle.description}

{"Context: " + context if context else ""}
AI Response: {response}

Task: {principle.critique_prompt}

Provide a detailed critique:"""
        
        # Generate critique
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_critique_length,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_critique_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        critique = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return critique.strip()
    
    def score_critique(self, critique: str) -> float:
        """Score how critical a critique is (0 = no issues, 1 = major issues)"""
        # Simple heuristic scoring based on keywords
        # In practice, this could use a trained classifier
        
        negative_keywords = [
            "harmful", "dangerous", "misleading", "false", "inappropriate",
            "offensive", "discriminatory", "privacy", "personal information",
            "unethical", "illegal", "unsafe", "problematic", "concerning"
        ]
        
        positive_keywords = [
            "appropriate", "helpful", "accurate", "respectful", "safe",
            "ethical", "constructive", "informative", "balanced"
        ]
        
        critique_lower = critique.lower()
        
        negative_count = sum(1 for keyword in negative_keywords if keyword in critique_lower)
        positive_count = sum(1 for keyword in positive_keywords if keyword in critique_lower)
        
        # Normalize score
        score = negative_count / (negative_count + positive_count + 1)
        
        return min(1.0, score)
    
    def generate_revision(
        self,
        response: str,
        critique: str,
        principle: ConstitutionalPrinciple,
        context: Optional[str] = None,
    ) -> str:
        """Generate revised response based on critique"""
        prompt = f"""You are revising an AI assistant's response based on constitutional critique.

Principle: {principle.name}
{"Context: " + context if context else ""}

Original Response: {response}

Critique: {critique}

Task: {principle.revision_prompt}

Provide the revised response:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_revision_length,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_revision_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        revision = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return revision.strip()

class ConstitutionalDataGenerator:
    """Generates constitutional training data through self-critique"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ConstitutionalAIConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.critiquer = ConstitutionalCritique(model, tokenizer, config)
        
        # Track generated data
        self.constitutional_data = []
    
    def generate_constitutional_data(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate constitutional training data from prompts
        
        Args:
            prompts: List of prompts to generate responses for
            responses: Optional pre-generated responses
            
        Returns:
            List of constitutional training examples
        """
        all_data = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating constitutional data")):
            # Generate initial response if not provided
            if responses and i < len(responses):
                initial_response = responses[i]
            else:
                initial_response = self._generate_response(prompt)
            
            # Generate critiques and revisions
            critique_revision_pairs = self._generate_critique_revision_pairs(
                prompt,
                initial_response
            )
            
            # Create training examples
            for critique, revision, principle_name, critique_score in critique_revision_pairs:
                example = {
                    "prompt": prompt,
                    "initial_response": initial_response,
                    "critique": critique,
                    "revision": revision,
                    "principle": principle_name,
                    "critique_score": critique_score,
                    "is_constitutional": critique_score < self.config.critique_score_threshold,
                }
                
                all_data.append(example)
        
        # Augment data if configured
        if self.config.augment_with_paraphrasing:
            all_data.extend(self._augment_with_paraphrases(all_data))
        
        if self.config.augment_with_adversarial:
            all_data.extend(self._augment_with_adversarial(all_data))
        
        # Save data if configured
        if self.config.save_constitutional_data:
            self._save_constitutional_data(all_data)
        
        self.constitutional_data.extend(all_data)
        
        return all_data
    
    def _generate_response(self, prompt: str) -> str:
        """Generate initial response to prompt"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_revision_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _generate_critique_revision_pairs(
        self,
        prompt: str,
        response: str,
    ) -> List[Tuple[str, str, str, float]]:
        """Generate critique and revision pairs for a response"""
        pairs = []
        
        # Sample principles to critique
        num_principles = min(
            self.config.num_critiques_per_response,
            len(self.config.principles)
        )
        sampled_principles = np.random.choice(
            self.config.principles,
            size=num_principles,
            replace=False,
            p=self._get_principle_weights() if self.config.use_principle_weighting else None
        )
        
        for principle in sampled_principles:
            # Generate critique
            critique = self.critiquer.generate_critique(response, principle, context=prompt)
            
            # Score critique
            if self.config.use_critique_scoring:
                critique_score = self.critiquer.score_critique(critique)
            else:
                critique_score = 0.5  # Neutral score
            
            # Generate revisions
            for _ in range(self.config.num_revisions_per_critique):
                revision = self.critiquer.generate_revision(
                    response, critique, principle, context=prompt
                )
                
                # Self-critique iterations
                current_revision = revision
                for iteration in range(self.config.num_self_critique_iterations):
                    self_critique = self.critiquer.generate_critique(
                        current_revision, principle, context=prompt
                    )
                    self_critique_score = self.critiquer.score_critique(self_critique)
                    
                    # Only revise if critique score is high
                    if self_critique_score > self.config.critique_score_threshold:
                        current_revision = self.critiquer.generate_revision(
                            current_revision, self_critique, principle, context=prompt
                        )
                    else:
                        break
                
                pairs.append((critique, current_revision, principle.name, critique_score))
        
        return pairs
    
    def _get_principle_weights(self) -> np.ndarray:
        """Get sampling weights for principles based on severity"""
        severities = [p.severity for p in self.config.principles]
        weights = np.array(severities)
        return weights / weights.sum()
    
    def _augment_with_paraphrases(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Augment data with paraphrased versions"""
        augmented = []
        
        for example in data[:len(data) // 4]:  # Augment 25% of data
            for _ in range(self.config.num_paraphrases):
                # Paraphrase the response
                paraphrase_prompt = f"Paraphrase the following text while maintaining its meaning:\n{example['revision']}"
                paraphrased = self._generate_response(paraphrase_prompt)
                
                augmented_example = example.copy()
                augmented_example["revision"] = paraphrased
                augmented_example["is_augmented"] = True
                augmented_example["augmentation_type"] = "paraphrase"
                
                augmented.append(augmented_example)
        
        return augmented
    
    def _augment_with_adversarial(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Augment data with adversarial examples"""
        augmented = []
        
        # Create adversarial examples by making responses slightly harmful
        harmful_modifications = [
            "Also, you could try [potentially harmful action]",
            "Another approach would be to [unethical suggestion]",
            "Some people might [inappropriate behavior]",
            "It's worth noting that [misleading information]",
        ]
        
        for example in data[:len(data) // 10]:  # Augment 10% of data
            if example["is_constitutional"]:
                # Make constitutional response unconstitutional
                adversarial_response = example["revision"] + " " + np.random.choice(harmful_modifications)
                
                # Generate critique for adversarial response
                principle = next(p for p in self.config.principles if p.name == example["principle"])
                critique = self.critiquer.generate_critique(
                    adversarial_response, principle, context=example["prompt"]
                )
                
                augmented_example = {
                    "prompt": example["prompt"],
                    "initial_response": adversarial_response,
                    "critique": critique,
                    "revision": example["revision"],  # Use original as good revision
                    "principle": example["principle"],
                    "critique_score": 0.8,  # High score for adversarial
                    "is_constitutional": False,
                    "is_augmented": True,
                    "augmentation_type": "adversarial",
                }
                
                augmented.append(augmented_example)
        
        return augmented
    
    def _save_constitutional_data(self, data: List[Dict[str, Any]]):
        """Save constitutional training data"""
        output_dir = Path(self.config.constitutional_data_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        output_file = output_dir / f"constitutional_data_{len(self.constitutional_data)}.jsonl"
        
        with open(output_file, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(data)} constitutional examples to {output_file}")

class ConstitutionalAITrainer:
    """Trainer for constitutional AI using self-supervision"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ConstitutionalAIConfig,
        base_trainer: Any,  # Base trainer (e.g., DPOTrainer)
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.base_trainer = base_trainer
        
        # Data generator
        self.data_generator = ConstitutionalDataGenerator(model, tokenizer, config)
    
    def train(
        self,
        train_prompts: List[str],
        num_rounds: int = 3,
        prompts_per_round: int = 1000,
    ):
        """
        Train model with constitutional AI
        
        Args:
            train_prompts: List of training prompts
            num_rounds: Number of constitutional training rounds
            prompts_per_round: Number of prompts to use per round
        """
        logger.info(f"Starting Constitutional AI training with {num_rounds} rounds")
        
        for round_idx in range(num_rounds):
            logger.info(f"Constitutional training round {round_idx + 1}/{num_rounds}")
            
            # Sample prompts for this round
            round_prompts = np.random.choice(
                train_prompts,
                size=min(prompts_per_round, len(train_prompts)),
                replace=False
            ).tolist()
            
            # Generate constitutional data
            constitutional_data = self.data_generator.generate_constitutional_data(round_prompts)
            
            # Convert to preference pairs for training
            preference_pairs = self._convert_to_preference_pairs(constitutional_data)
            
            # Train with base trainer
            logger.info(f"Training on {len(preference_pairs)} preference pairs")
            self.base_trainer.train_on_data(preference_pairs)
            
            # Evaluate constitutional adherence
            if round_idx < num_rounds - 1:
                metrics = self.evaluate_constitutional_adherence(
                    train_prompts[:100]  # Evaluate on subset
                )
                logger.info(f"Round {round_idx + 1} constitutional metrics: {metrics}")
    
    def _convert_to_preference_pairs(
        self,
        constitutional_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert constitutional data to preference pairs"""
        preference_pairs = []
        
        for example in constitutional_data:
            # Create preference pair
            if example["critique_score"] > self.config.critique_score_threshold:
                # Initial response is rejected, revision is preferred
                pair = {
                    "prompt": example["prompt"],
                    "chosen": example["revision"],
                    "rejected": example["initial_response"],
                    "metadata": {
                        "principle": example["principle"],
                        "critique_score": example["critique_score"],
                    }
                }
            else:
                # Initial response is already good
                continue
            
            preference_pairs.append(pair)
        
        return preference_pairs
    
    def evaluate_constitutional_adherence(
        self,
        test_prompts: List[str]
    ) -> Dict[str, float]:
        """Evaluate model's adherence to constitutional principles"""
        metrics = {
            "avg_critique_score": [],
            "constitutional_rate": [],
        }
        
        for principle in self.config.principles:
            principle_scores = []
            
            for prompt in test_prompts:
                # Generate response
                response = self.data_generator._generate_response(prompt)
                
                # Generate critique
                critique = self.data_generator.critiquer.generate_critique(
                    response, principle, context=prompt
                )
                
                # Score critique
                score = self.data_generator.critiquer.score_critique(critique)
                principle_scores.append(score)
            
            avg_score = np.mean(principle_scores)
            constitutional_rate = np.mean([s < self.config.critique_score_threshold for s in principle_scores])
            
            metrics[f"critique_score/{principle.name}"] = avg_score
            metrics[f"constitutional_rate/{principle.name}"] = constitutional_rate
            
            metrics["avg_critique_score"].append(avg_score)
            metrics["constitutional_rate"].append(constitutional_rate)
        
        # Overall metrics
        metrics["avg_critique_score"] = np.mean(metrics["avg_critique_score"])
        metrics["constitutional_rate"] = np.mean(metrics["constitutional_rate"])
        
        return metrics