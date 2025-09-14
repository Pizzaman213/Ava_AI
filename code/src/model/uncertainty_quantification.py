"""
Uncertainty Quantification for MoE++ Models

Implements various uncertainty estimation methods to help models
know when they are uncertain about their predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification"""
    # Methods to use
    use_monte_carlo_dropout: bool = True
    use_deep_ensembles: bool = False
    use_variational_inference: bool = False
    use_epistemic_uncertainty: bool = True
    use_aleatoric_uncertainty: bool = True
    
    # Monte Carlo settings
    mc_dropout_samples: int = 10
    mc_dropout_rate: float = 0.1
    
    # Ensemble settings
    ensemble_size: int = 5
    ensemble_diversity_loss_weight: float = 0.01
    
    # Variational settings
    kl_loss_weight: float = 1e-6
    prior_variance: float = 1.0
    
    # Uncertainty thresholds
    high_uncertainty_threshold: float = 0.5
    abstention_threshold: float = 0.8
    
    # Temperature scaling
    use_temperature_scaling: bool = True
    initial_temperature: float = 1.0


class MCDropoutMoE(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation in MoE models
    """
    
    def __init__(self, base_model: nn.Module, config: UncertaintyConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Enable dropout during inference
        self._enable_eval_dropout()
        
        # Temperature parameter for calibration
        if config.use_temperature_scaling:
            self.temperature = nn.Parameter(torch.tensor(config.initial_temperature))
        else:
            self.temperature = config.initial_temperature
            
    def _enable_eval_dropout(self):
        """Enable dropout modules during evaluation"""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional uncertainty estimation
        """
        if not return_uncertainty:
            # Standard forward pass
            return self.base_model(input_ids, attention_mask=attention_mask, **kwargs)
            
        # Monte Carlo forward passes
        outputs_list = []
        
        for _ in range(self.config.mc_dropout_samples):
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                outputs_list.append(outputs.logits)
                
        # Stack predictions
        logits_samples = torch.stack(outputs_list, dim=0)  # [samples, batch, seq, vocab]
        
        # Compute uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty(logits_samples)
        
        # Mean prediction
        mean_logits = logits_samples.mean(dim=0)
        
        return {
            "logits": mean_logits / self.temperature,
            "uncertainty": uncertainty_metrics,
            "logits_samples": logits_samples
        }
        
    def _compute_uncertainty(
        self,
        logits_samples: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute various uncertainty metrics"""
        # Convert to probabilities
        probs_samples = F.softmax(logits_samples / self.temperature, dim=-1)
        
        # Mean probabilities
        mean_probs = probs_samples.mean(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        # Measured as the entropy of the mean distribution
        epistemic_uncertainty = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-8),
            dim=-1
        )
        
        # Aleatoric uncertainty (data uncertainty)
        # Measured as the expected entropy
        sample_entropies = -torch.sum(
            probs_samples * torch.log(probs_samples + 1e-8),
            dim=-1
        )
        aleatoric_uncertainty = sample_entropies.mean(dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Variance-based uncertainty
        variance_uncertainty = probs_samples.var(dim=0).mean(dim=-1)
        
        # Mutual information (another epistemic measure)
        mutual_info = epistemic_uncertainty - aleatoric_uncertainty
        
        return {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "total": total_uncertainty,
            "variance": variance_uncertainty,
            "mutual_info": mutual_info.clamp(min=0)  # MI should be non-negative
        }
        
    def calibrate_temperature(
        self,
        val_loader: Any,
        criterion: nn.Module = nn.CrossEntropyLoss()
    ):
        """Calibrate temperature parameter on validation set"""
        if not self.config.use_temperature_scaling:
            return
            
        logger.info("Calibrating temperature...")
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.base_model(**batch)
                logits_list.append(outputs.logits)
                labels_list.append(batch["labels"])
                
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_temperature():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_temperature)
        
        logger.info(f"Calibrated temperature: {self.temperature.item():.3f}")


class DeepEnsembleMoE(nn.Module):
    """
    Deep Ensemble of MoE models for uncertainty estimation
    """
    
    def __init__(
        self,
        model_class: type,
        model_config: Any,
        uncertainty_config: UncertaintyConfig
    ):
        super().__init__()
        self.config = uncertainty_config
        
        # Create ensemble members
        self.ensemble = nn.ModuleList([
            model_class(model_config)
            for _ in range(uncertainty_config.ensemble_size)
        ])
        
        # Diversity encouraging loss
        self.diversity_loss_weight = uncertainty_config.ensemble_diversity_loss_weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        
        # Get predictions from all ensemble members
        outputs_list = []
        for model in self.ensemble:
            outputs = model(input_ids, attention_mask=attention_mask, **kwargs)
            outputs_list.append(outputs.logits)
            
        # Stack predictions
        logits_ensemble = torch.stack(outputs_list, dim=0)
        
        if return_uncertainty:
            # Compute uncertainty
            uncertainty_metrics = self._compute_uncertainty(logits_ensemble)
            
            return {
                "logits": logits_ensemble.mean(dim=0),
                "uncertainty": uncertainty_metrics,
                "logits_ensemble": logits_ensemble
            }
        else:
            return {"logits": logits_ensemble.mean(dim=0)}
            
    def _compute_uncertainty(
        self,
        logits_ensemble: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute ensemble-based uncertainty"""
        # Similar to MC Dropout but using ensemble predictions
        probs_ensemble = F.softmax(logits_ensemble, dim=-1)
        mean_probs = probs_ensemble.mean(dim=0)
        
        # Predictive uncertainty
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + 1e-8),
            dim=-1
        )
        
        # Ensemble variance
        ensemble_variance = probs_ensemble.var(dim=0).mean(dim=-1)
        
        return {
            "predictive_entropy": predictive_entropy,
            "ensemble_variance": ensemble_variance
        }
        
    def compute_diversity_loss(self) -> torch.Tensor:
        """Compute loss that encourages diverse ensemble predictions"""
        if len(self.ensemble) < 2:
            return torch.tensor(0.0)
            
        # Compute pairwise similarities between ensemble members
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(self.ensemble)):
            for j in range(i + 1, len(self.ensemble)):
                # Get parameters
                params_i = torch.cat([p.flatten() for p in self.ensemble[i].parameters()])
                params_j = torch.cat([p.flatten() for p in self.ensemble[j].parameters()])
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    params_i.unsqueeze(0),
                    params_j.unsqueeze(0)
                )
                
                # We want to minimize similarity (maximize diversity)
                diversity_loss += similarity
                num_pairs += 1
                
        return diversity_loss / num_pairs * self.diversity_loss_weight


class BayesianMoELayer(nn.Module):
    """
    Bayesian MoE layer with variational inference
    """
    
    def __init__(self, base_layer: nn.Module, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        
        # Convert deterministic weights to variational
        self.weight_mu = nn.Parameter(base_layer.weight.data.clone())
        self.weight_log_sigma = nn.Parameter(
            torch.full_like(base_layer.weight, np.log(0.1))
        )
        
        if base_layer.bias is not None:
            self.bias_mu = nn.Parameter(base_layer.bias.data.clone())
            self.bias_log_sigma = nn.Parameter(
                torch.full_like(base_layer.bias, np.log(0.1))
            )
        else:
            self.bias_mu = None
            self.bias_log_sigma = None
            
        # Prior parameters
        self.prior_mu = 0.0
        self.prior_sigma = config.prior_variance
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with weight sampling"""
        # Sample weights
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        # Sample bias
        if self.bias_mu is not None:
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            bias = None
            
        # Linear transformation
        output = F.linear(x, weight, bias)
        
        # Compute KL divergence for regularization
        kl_loss = self._compute_kl_divergence()
        
        return output, kl_loss
        
    def _compute_kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # Weight KL
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_kl = 0.5 * torch.sum(
            (weight_sigma ** 2) / (self.prior_sigma ** 2) +
            ((self.weight_mu - self.prior_mu) ** 2) / (self.prior_sigma ** 2) -
            1 + 2 * np.log(self.prior_sigma) - 2 * self.weight_log_sigma
        )
        
        # Bias KL
        bias_kl = 0
        if self.bias_mu is not None:
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias_kl = 0.5 * torch.sum(
                (bias_sigma ** 2) / (self.prior_sigma ** 2) +
                ((self.bias_mu - self.prior_mu) ** 2) / (self.prior_sigma ** 2) -
                1 + 2 * np.log(self.prior_sigma) - 2 * self.bias_log_sigma
            )
            
        return (weight_kl + bias_kl) * self.config.kl_loss_weight


class UncertaintyAwareMoE(nn.Module):
    """
    MoE model with integrated uncertainty quantification
    """
    
    def __init__(self, base_model: nn.Module, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        
        # Choose uncertainty method
        if config.use_monte_carlo_dropout:
            self.uncertainty_model = MCDropoutMoE(base_model, config)
        elif config.use_deep_ensembles:
            # Note: This requires multiple model instances
            raise NotImplementedError("Deep ensembles require special initialization")
        else:
            self.uncertainty_model = base_model
            
        # Uncertainty-aware routing
        self.uncertainty_router = UncertaintyRouter(
            base_model.config.hidden_size,
            base_model.config.num_experts
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation"""
        
        # Get predictions with uncertainty
        outputs = self.uncertainty_model(
            input_ids,
            attention_mask=attention_mask,
            return_uncertainty=return_uncertainty,
            **kwargs
        )
        
        if return_uncertainty:
            # Route based on uncertainty
            uncertainty = outputs["uncertainty"]["total"]
            routing_adjustments = self.uncertainty_router(uncertainty)
            outputs["routing_adjustments"] = routing_adjustments
            
            # Add abstention decisions
            outputs["should_abstain"] = self._should_abstain(uncertainty)
            
        return outputs
        
    def _should_abstain(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Decide whether to abstain from prediction due to high uncertainty"""
        return uncertainty > self.config.abstention_threshold
        
    def generate_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Generate text with uncertainty tracking"""
        generated = input_ids
        uncertainty_history = {
            "epistemic": [],
            "aleatoric": [],
            "total": []
        }
        
        for _ in range(max_length):
            # Get next token predictions with uncertainty
            outputs = self.forward(
                generated,
                return_uncertainty=True
            )
            
            # Track uncertainty
            uncertainty = outputs["uncertainty"]
            for key in uncertainty_history:
                if key in uncertainty:
                    uncertainty_history[key].append(
                        uncertainty[key][:, -1].mean().item()
                    )
                    
            # Check if we should stop due to high uncertainty
            if outputs["should_abstain"][:, -1].any():
                logger.warning("Stopping generation due to high uncertainty")
                break
                
            # Sample next token
            logits = outputs["logits"][:, -1]
            next_token = torch.multinomial(
                F.softmax(logits, dim=-1),
                num_samples=1
            )
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if next_token.item() == self.uncertainty_model.base_model.config.eos_token_id:
                break
                
        return generated, uncertainty_history


class UncertaintyRouter(nn.Module):
    """
    Routes tokens to experts based on uncertainty levels
    """
    
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Uncertainty-based routing adjustment
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Adjust routing based on uncertainty
        
        High uncertainty -> Route to more experts
        Low uncertainty -> Route to fewer experts
        """
        # Reshape uncertainty to [batch_size * seq_len, 1]
        uncertainty_flat = uncertainty.view(-1, 1)
        
        # Project to expert space
        routing_adjustment = self.uncertainty_proj(uncertainty_flat)
        
        # Sigmoid to get adjustment factors
        adjustment_factors = torch.sigmoid(routing_adjustment)
        
        return adjustment_factors.view(uncertainty.shape[0], uncertainty.shape[1], -1)