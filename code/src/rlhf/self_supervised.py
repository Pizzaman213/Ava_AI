"""
Self-Supervised Learning Framework
Implements consistency training, confidence calibration, and contrastive learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import math
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SelfSupervisedConfig:
    """Configuration for self-supervised learning components"""
    # Consistency training
    consistency_weight: float = 0.5
    consistency_temperature: float = 0.7
    consistency_augmentation_types: List[str] = None
    
    # Confidence calibration
    confidence_bins: int = 10
    confidence_smoothing: float = 0.1
    calibration_weight: float = 0.3
    
    # Reasoning chain
    max_reasoning_steps: int = 10
    reasoning_temperature: float = 0.5
    reasoning_dropout: float = 0.1
    
    # Contrastive learning
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.5
    negative_samples: int = 8
    hard_negative_ratio: float = 0.5
    
    def __post_init__(self):
        if self.consistency_augmentation_types is None:
            self.consistency_augmentation_types = [
                "paraphrase", "reorder", "synonym", "backtranslation"
            ]

class ConsistencyTrainer(nn.Module):
    """Trains model to produce consistent outputs for semantically similar inputs"""
    def __init__(self, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config
        self.augmentation_functions = self._setup_augmentations()
        
    def _setup_augmentations(self) -> Dict[str, callable]:
        """Setup augmentation functions for consistency training"""
        augmentations = {}
        
        if "paraphrase" in self.config.consistency_augmentation_types:
            augmentations["paraphrase"] = self._paraphrase_augmentation
        if "reorder" in self.config.consistency_augmentation_types:
            augmentations["reorder"] = self._reorder_augmentation
        if "synonym" in self.config.consistency_augmentation_types:
            augmentations["synonym"] = self._synonym_augmentation
        if "backtranslation" in self.config.consistency_augmentation_types:
            augmentations["backtranslation"] = self._backtranslation_augmentation
            
        return augmentations
    
    def _paraphrase_augmentation(self, text: str, model: nn.Module, tokenizer: Any) -> str:
        """Generate paraphrase of input text"""
        prompt = f"Paraphrase the following text while maintaining its meaning:\n{text}\n\nParaphrase:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(text.split()) * 2,
                temperature=self.config.consistency_temperature,
                do_sample=True,
            )
        
        paraphrase = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return paraphrase.strip()
    
    def _reorder_augmentation(self, text: str, model: nn.Module, tokenizer: Any) -> str:
        """Reorder sentences while maintaining meaning"""
        sentences = text.split(". ")
        if len(sentences) <= 1:
            return text
        
        # Shuffle middle sentences, keep first and last
        if len(sentences) > 2:
            middle = sentences[1:-1]
            np.random.shuffle(middle)
            sentences = [sentences[0]] + middle + [sentences[-1]]
        
        return ". ".join(sentences)
    
    def _synonym_augmentation(self, text: str, model: nn.Module, tokenizer: Any) -> str:
        """Replace words with synonyms"""
        # Simplified version - in practice, use proper synonym database
        words = text.split()
        num_replacements = max(1, len(words) // 10)
        positions = np.random.choice(len(words), num_replacements, replace=False)
        
        for pos in positions:
            # Generate synonym using model
            prompt = f"Provide a synonym for '{words[pos]}' in the context: {text}"
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.7)
            
            synonym = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            words[pos] = synonym.strip().split()[0] if synonym.strip() else words[pos]
        
        return " ".join(words)
    
    def _backtranslation_augmentation(self, text: str, model: nn.Module, tokenizer: Any) -> str:
        """Simulate backtranslation augmentation"""
        # Simplified version - add noise and reconstruct
        prompt = f"Rewrite the following text in a slightly different way:\n{text}\n\nRewritten:"
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(text.split()) * 2,
                temperature=self.config.consistency_temperature,
                do_sample=True,
            )
        
        rewritten = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return rewritten.strip()
    
    def compute_consistency_loss(
        self,
        model: nn.Module,
        tokenizer: Any,
        original_text: str,
        original_output: torch.Tensor,
        augmented_texts: List[str] = None,
    ) -> torch.Tensor:
        """Compute consistency loss between original and augmented outputs"""
        if augmented_texts is None:
            # Generate augmented versions
            augmented_texts = []
            for aug_type, aug_fn in self.augmentation_functions.items():
                try:
                    augmented = aug_fn(original_text, model, tokenizer)
                    augmented_texts.append(augmented)
                except Exception as e:
                    logger.warning(f"Augmentation {aug_type} failed: {e}")
        
        consistency_losses = []
        
        for aug_text in augmented_texts:
            # Get model output for augmented text
            inputs = tokenizer(aug_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                aug_outputs = model(**inputs, return_dict=True)
                aug_hidden = aug_outputs.last_hidden_state.mean(dim=1)  # Pool hidden states
            
            # Compute consistency loss (cosine similarity)
            original_pooled = original_output.mean(dim=1) if original_output.dim() > 2 else original_output
            similarity = F.cosine_similarity(original_pooled, aug_hidden, dim=-1)
            loss = 1 - similarity  # Convert to loss
            
            consistency_losses.append(loss)
        
        return torch.stack(consistency_losses).mean() * self.config.consistency_weight

class ConfidenceCalibration(nn.Module):
    """Calibrates model confidence to match actual accuracy"""
    def __init__(self, hidden_size: int, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Calibration statistics
        self.calibration_stats = {
            "confidence_bins": torch.linspace(0, 1, config.confidence_bins + 1),
            "accuracy_per_bin": torch.zeros(config.confidence_bins),
            "count_per_bin": torch.zeros(config.confidence_bins),
        }
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict confidence score for hidden states"""
        # Pool hidden states if needed
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        
        confidence = self.confidence_head(pooled)
        return confidence.squeeze(-1)
    
    def compute_calibration_loss(
        self,
        confidence_scores: torch.Tensor,
        correctness_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute calibration loss and metrics"""
        # Expected Calibration Error (ECE)
        ece = 0.0
        metrics = {}
        
        for i in range(self.config.confidence_bins):
            bin_lower = self.calibration_stats["confidence_bins"][i]
            bin_upper = self.calibration_stats["confidence_bins"][i + 1]
            
            # Find predictions in this bin
            in_bin = (confidence_scores >= bin_lower) & (confidence_scores < bin_upper)
            
            if in_bin.sum() > 0:
                # Average confidence in bin
                avg_confidence = confidence_scores[in_bin].mean()
                
                # Average accuracy in bin
                avg_accuracy = correctness_labels[in_bin].float().mean()
                
                # Bin weight
                bin_weight = in_bin.sum().float() / len(confidence_scores)
                
                # ECE contribution
                ece += bin_weight * torch.abs(avg_confidence - avg_accuracy)
                
                # Update statistics
                self.calibration_stats["accuracy_per_bin"][i] = avg_accuracy
                self.calibration_stats["count_per_bin"][i] = in_bin.sum()
        
        # Calibration loss (MSE between confidence and correctness)
        calibration_loss = F.mse_loss(confidence_scores, correctness_labels.float())
        
        # Add smoothing regularization
        confidence_variance = confidence_scores.var()
        smoothing_loss = self.config.confidence_smoothing * (1 / (confidence_variance + 1e-8))
        
        total_loss = calibration_loss + smoothing_loss
        
        metrics["ece"] = ece.item()
        metrics["avg_confidence"] = confidence_scores.mean().item()
        metrics["avg_accuracy"] = correctness_labels.float().mean().item()
        
        return total_loss * self.config.calibration_weight, metrics

class ReasoningChainModule(nn.Module):
    """Models explicit multi-step reasoning with self-critique"""
    def __init__(self, hidden_size: int, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Reasoning step generator
        self.step_generator = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=config.reasoning_dropout,
            batch_first=True
        )
        
        # Step validity predictor
        self.validity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.reasoning_dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Step critique generator
        self.critique_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=config.reasoning_dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Step refinement module
        self.refinement_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.reasoning_dropout),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        initial_state: torch.Tensor,
        max_steps: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]:
        """Generate reasoning chain with self-critique"""
        max_steps = max_steps or self.config.max_reasoning_steps
        
        reasoning_steps = []
        validity_scores = []
        critiques = []
        
        # Initialize LSTM state
        batch_size = initial_state.size(0)
        h_0 = torch.zeros(2, batch_size, self.hidden_size).to(initial_state.device)
        c_0 = torch.zeros(2, batch_size, self.hidden_size).to(initial_state.device)
        
        current_state = initial_state.unsqueeze(1)
        hidden = (h_0, c_0)
        
        for step in range(max_steps):
            # Generate next reasoning step
            step_output, hidden = self.step_generator(current_state, hidden)
            step_output = step_output.squeeze(1)
            
            # Predict step validity
            validity = self.validity_head(step_output)
            
            # Generate critique for this step
            if len(reasoning_steps) > 0:
                # Critique based on previous steps
                prev_steps = torch.stack(reasoning_steps, dim=1)
                all_steps = torch.cat([prev_steps, step_output.unsqueeze(1)], dim=1)
                critique = self.critique_head(all_steps)[:, -1, :]  # Last position
            else:
                critique = self.critique_head(step_output.unsqueeze(1)).squeeze(1)
            
            # Refine step based on critique
            refined_step = self.refinement_head(torch.cat([step_output, critique], dim=-1))
            
            reasoning_steps.append(refined_step)
            validity_scores.append(validity.squeeze(-1))
            critiques.append(critique)
            
            # Update current state for next step
            current_state = refined_step.unsqueeze(1)
            
            # Early stopping if validity is low
            if validity.mean() < 0.3:
                break
        
        return reasoning_steps, validity_scores, critiques
    
    def compute_reasoning_loss(
        self,
        reasoning_steps: List[torch.Tensor],
        validity_scores: List[torch.Tensor],
        target_reasoning: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute loss for reasoning chain"""
        losses = []
        
        # Validity consistency loss
        if len(validity_scores) > 1:
            validity_tensor = torch.stack(validity_scores)
            validity_var = validity_tensor.var(dim=0).mean()
            losses.append(validity_var)  # Encourage consistent validity
        
        # Step coherence loss
        if len(reasoning_steps) > 1:
            for i in range(1, len(reasoning_steps)):
                # Adjacent steps should be related but not identical
                similarity = F.cosine_similarity(
                    reasoning_steps[i-1], reasoning_steps[i], dim=-1
                )
                coherence_loss = torch.abs(similarity - 0.7)  # Target moderate similarity
                losses.append(coherence_loss.mean())
        
        # Supervised loss if target reasoning provided
        if target_reasoning is not None:
            for i, (pred, target) in enumerate(zip(reasoning_steps, target_reasoning)):
                step_loss = F.mse_loss(pred, target)
                losses.append(step_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

class ContrastiveLearner(nn.Module):
    """Contrastive learning to distinguish high-quality from low-quality outputs"""
    def __init__(self, hidden_size: int, config: SelfSupervisedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 128)  # Project to smaller space
        )
        
        # Quality scorer
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states and compute quality scores"""
        # Pool if needed
        if hidden_states.dim() == 3:
            pooled = hidden_states.mean(dim=1)
        else:
            pooled = hidden_states
        
        # Contrastive projection
        projected = self.projection_head(pooled)
        projected = F.normalize(projected, p=2, dim=-1)
        
        # Quality score
        quality = self.quality_head(pooled)
        
        return projected, quality
    
    def compute_contrastive_loss(
        self,
        anchor_proj: torch.Tensor,
        positive_proj: torch.Tensor,
        negative_projs: List[torch.Tensor],
        hard_negative_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss"""
        batch_size = anchor_proj.size(0)
        
        # Positive similarity
        pos_sim = F.cosine_similarity(anchor_proj, positive_proj, dim=-1)
        pos_sim = pos_sim / self.config.contrastive_temperature
        
        # Negative similarities
        neg_sims = []
        for neg_proj in negative_projs:
            neg_sim = F.cosine_similarity(
                anchor_proj.unsqueeze(1), 
                neg_proj.unsqueeze(0), 
                dim=-1
            )
            neg_sims.append(neg_sim / self.config.contrastive_temperature)
        
        neg_sims = torch.stack(neg_sims, dim=-1)
        
        # Apply hard negative weighting if provided
        if hard_negative_mask is not None:
            # Give more weight to hard negatives
            weights = torch.ones_like(neg_sims)
            weights[hard_negative_mask] *= 2.0
            neg_sims = neg_sims * weights
        
        # InfoNCE loss
        all_sims = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(anchor_proj.device)
        
        loss = F.cross_entropy(all_sims, labels)
        
        return loss
    
    def generate_hard_negatives(
        self,
        model: nn.Module,
        tokenizer: Any,
        positive_text: str,
        num_negatives: int = 4
    ) -> List[str]:
        """Generate hard negative examples"""
        hard_negatives = []
        
        # Type 1: Factually incorrect version
        prompt = f"Generate a factually incorrect version of: {positive_text}"
        hard_neg = self._generate_with_prompt(model, tokenizer, prompt)
        hard_negatives.append(hard_neg)
        
        # Type 2: Logically inconsistent version
        prompt = f"Generate a logically inconsistent version of: {positive_text}"
        hard_neg = self._generate_with_prompt(model, tokenizer, prompt)
        hard_negatives.append(hard_neg)
        
        # Type 3: Off-topic version
        prompt = f"Generate text that seems related but is actually off-topic from: {positive_text}"
        hard_neg = self._generate_with_prompt(model, tokenizer, prompt)
        hard_negatives.append(hard_neg)
        
        # Type 4: Partially correct version
        prompt = f"Generate a partially correct but misleading version of: {positive_text}"
        hard_neg = self._generate_with_prompt(model, tokenizer, prompt)
        hard_negatives.append(hard_neg)
        
        return hard_negatives[:num_negatives]
    
    def _generate_with_prompt(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompt: str
    ) -> str:
        """Helper to generate text with a prompt"""
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
            )
        
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated.strip()

class SelfSupervisedFramework:
    """Unified self-supervised learning framework"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: SelfSupervisedConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize components
        hidden_size = model.config.hidden_size
        self.consistency_trainer = ConsistencyTrainer(config)
        self.confidence_calibrator = ConfidenceCalibration(hidden_size, config)
        self.reasoning_chain = ReasoningChainModule(hidden_size, config)
        self.contrastive_learner = ContrastiveLearner(hidden_size, config)
        
        # Move to device
        device = next(model.parameters()).device
        self.confidence_calibrator.to(device)
        self.reasoning_chain.to(device)
        self.contrastive_learner.to(device)
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
    
    def train_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Single training step with all self-supervised objectives"""
        losses = {}
        metrics = {}
        
        # Get model outputs
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            return_dict=True,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # 1. Consistency training
        if "text" in batch:
            consistency_loss = self.consistency_trainer.compute_consistency_loss(
                self.model,
                self.tokenizer,
                batch["text"][0],  # First example in batch
                hidden_states[0]
            )
            losses["consistency"] = consistency_loss
        
        # 2. Confidence calibration
        confidence_scores = self.confidence_calibrator(hidden_states)
        if "correctness_labels" in batch:
            calibration_loss, calib_metrics = self.confidence_calibrator.compute_calibration_loss(
                confidence_scores,
                batch["correctness_labels"]
            )
            losses["calibration"] = calibration_loss
            metrics.update({f"calibration/{k}": v for k, v in calib_metrics.items()})
        
        # 3. Reasoning chain
        reasoning_steps, validity_scores, critiques = self.reasoning_chain(hidden_states)
        reasoning_loss = self.reasoning_chain.compute_reasoning_loss(
            reasoning_steps,
            validity_scores,
            batch.get("target_reasoning")
        )
        losses["reasoning"] = reasoning_loss
        metrics["reasoning/num_steps"] = len(reasoning_steps)
        metrics["reasoning/avg_validity"] = torch.stack(validity_scores).mean().item()
        
        # 4. Contrastive learning
        anchor_proj, anchor_quality = self.contrastive_learner(hidden_states)
        
        if "positive_ids" in batch and "negative_ids" in batch:
            # Get positive projection
            pos_outputs = self.model(
                input_ids=batch["positive_ids"],
                attention_mask=batch.get("positive_mask"),
                return_dict=True,
                output_hidden_states=True
            )
            pos_hidden = pos_outputs.hidden_states[-1]
            pos_proj, pos_quality = self.contrastive_learner(pos_hidden)
            
            # Get negative projections
            neg_projs = []
            for neg_ids in batch["negative_ids"]:
                neg_outputs = self.model(
                    input_ids=neg_ids,
                    return_dict=True,
                    output_hidden_states=True
                )
                neg_hidden = neg_outputs.hidden_states[-1]
                neg_proj, _ = self.contrastive_learner(neg_hidden)
                neg_projs.append(neg_proj)
            
            contrastive_loss = self.contrastive_learner.compute_contrastive_loss(
                anchor_proj,
                pos_proj,
                neg_projs,
                batch.get("hard_negative_mask")
            )
            losses["contrastive"] = contrastive_loss
            
            # Quality prediction loss
            quality_labels = torch.cat([
                torch.ones_like(anchor_quality),
                torch.zeros(len(neg_projs), 1).to(anchor_quality.device)
            ])
            quality_preds = torch.cat([anchor_quality] + [self.contrastive_learner.quality_head(
                neg_hidden.mean(dim=1) if neg_hidden.dim() == 3 else neg_hidden
            ) for neg_hidden in [neg_outputs.hidden_states[-1]]])
            
            quality_loss = F.binary_cross_entropy_with_logits(quality_preds, quality_labels)
            losses["quality"] = quality_loss
        
        # Combine losses
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        
        # Update metrics history
        self.metrics_history.append({**losses, **metrics})
        
        return losses
    
    def generate_self_training_data(
        self,
        prompts: List[str],
        num_generations_per_prompt: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate training data through self-supervision"""
        training_data = []
        
        for prompt in tqdm(prompts, desc="Generating self-training data"):
            prompt_data = {
                "prompt": prompt,
                "generations": [],
                "quality_scores": [],
                "reasoning_chains": [],
            }
            
            # Generate multiple outputs
            for _ in range(num_generations_per_prompt):
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Generate response
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.8,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
                    
                    # Get hidden states for analysis
                    hidden_states = outputs.hidden_states[-1][-1]  # Last token, last layer
                    
                    # Get confidence score
                    confidence = self.confidence_calibrator(hidden_states.unsqueeze(0))
                    
                    # Get quality score
                    _, quality = self.contrastive_learner(hidden_states.unsqueeze(0))
                    
                    # Generate reasoning chain
                    reasoning_steps, validity_scores, _ = self.reasoning_chain(
                        hidden_states.unsqueeze(0)
                    )
                
                generation = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                
                prompt_data["generations"].append(generation)
                prompt_data["quality_scores"].append(quality.item())
                prompt_data["reasoning_chains"].append({
                    "steps": [step.cpu().numpy() for step in reasoning_steps],
                    "validity": [v.item() for v in validity_scores]
                })
            
            # Create contrastive pairs
            quality_scores = torch.tensor(prompt_data["quality_scores"])
            best_idx = quality_scores.argmax()
            worst_idx = quality_scores.argmin()
            
            if best_idx != worst_idx:
                training_example = {
                    "prompt": prompt,
                    "positive": prompt_data["generations"][best_idx],
                    "negative": prompt_data["generations"][worst_idx],
                    "quality_gap": (quality_scores[best_idx] - quality_scores[worst_idx]).item(),
                    "reasoning_positive": prompt_data["reasoning_chains"][best_idx],
                    "reasoning_negative": prompt_data["reasoning_chains"][worst_idx],
                }
                training_data.append(training_example)
        
        return training_data