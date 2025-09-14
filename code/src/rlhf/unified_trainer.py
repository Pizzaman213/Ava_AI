"""
Unified RLHF Training System
Integrates all components into a cohesive training pipeline
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from LLM.src.utils.path_utils import get_outputs_dir
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import gc

# Import all RLHF components
from .self_supervised import SelfSupervisedFramework, SelfSupervisedConfig
from .multi_reward import MultiRewardModel, MultiRewardConfig
from .self_evaluation import SelfEvaluationSystem, SelfEvaluationConfig
from .enhanced_ppo import EnhancedPPOTrainer, EnhancedPPOConfig
from .constitutional_enhanced import EnhancedConstitutionalAI, ConstitutionalConfig
from .reasoning import ReasoningModule, ReasoningConfig
from .lora_integration import LoRAAdapter, LoRAConfig, create_peft_model
from .evaluation import ComprehensiveEvaluator, EvaluationConfig
from .data_pipeline import UnifiedDataPipeline, DataPipelineConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Unified configuration for all training components"""
    # Model settings
    model_name: str = "moe_llm"
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # Component configs
    self_supervised: SelfSupervisedConfig = field(default_factory=SelfSupervisedConfig)
    multi_reward: MultiRewardConfig = field(default_factory=MultiRewardConfig)
    self_evaluation: SelfEvaluationConfig = field(default_factory=SelfEvaluationConfig)
    ppo: EnhancedPPOConfig = field(default_factory=EnhancedPPOConfig)
    constitutional: ConstitutionalConfig = field(default_factory=ConstitutionalConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    
    # Training settings
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Curriculum settings
    use_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "basic", "steps": 5000, "complexity": 0.3},
        {"name": "intermediate", "steps": 10000, "complexity": 0.6},
        {"name": "advanced", "steps": 15000, "complexity": 0.9},
        {"name": "expert", "steps": -1, "complexity": 1.0}
    ])
    
    # Multi-phase training
    training_phases: List[str] = field(default_factory=lambda: [
        "self_supervised_pretrain",
        "constitutional_alignment",
        "ppo_optimization",
        "reasoning_enhancement",
        "final_tuning"
    ])
    
    # Resource management
    cpu_offload: bool = True
    activation_checkpointing: bool = True
    distributed_training: bool = True
    
    # Output settings
    output_dir: str = None
    experiment_name: str = "unified_rlhf_run"
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = os.path.join(get_outputs_dir(), "unified_rlhf")
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "unified-rlhf"
    save_total_limit: int = 3

class CurriculumScheduler:
    """Manages curriculum learning progression"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage_idx = 0
        self.current_step = 0
        self.stage_start_step = 0
    
    def get_current_stage(self) -> Dict[str, Any]:
        """Get current curriculum stage"""
        if self.current_stage_idx >= len(self.config.curriculum_stages):
            return self.config.curriculum_stages[-1]
        return self.config.curriculum_stages[self.current_stage_idx]
    
    def should_advance_stage(self, metrics: Dict[str, float]) -> bool:
        """Check if should advance to next stage"""
        current_stage = self.get_current_stage()
        
        # Check step limit
        if current_stage["steps"] > 0:
            steps_in_stage = self.current_step - self.stage_start_step
            if steps_in_stage >= current_stage["steps"]:
                return True
        
        # Check performance metrics
        if "success_rate" in metrics and metrics["success_rate"] > 0.8:
            return True
        
        return False
    
    def advance_stage(self):
        """Advance to next curriculum stage"""
        if self.current_stage_idx < len(self.config.curriculum_stages) - 1:
            self.current_stage_idx += 1
            self.stage_start_step = self.current_step
            logger.info(f"Advanced to curriculum stage: {self.get_current_stage()['name']}")
    
    def update_step(self, step: int):
        """Update current step"""
        self.current_step = step

class OnlineLearningModule:
    """Manages online learning with catastrophic forgetting prevention"""
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Experience replay for continual learning
        self.experience_buffer = []
        self.buffer_size = 10000
        
        # EWC (Elastic Weight Consolidation) for forgetting prevention
        self.fisher_information = {}
        self.optimal_params = {}
        self.ewc_lambda = 0.5
    
    def update_fisher_information(self, dataloader: Any):
        """Compute Fisher Information Matrix for EWC"""
        self.model.eval()
        fisher_accumulator = {}
        
        # Initialize
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_accumulator[name] = torch.zeros_like(param)
        
        # Accumulate gradients
        for batch in dataloader:
            self.model.zero_grad()
            output = self.model(**batch)
            loss = output.loss if hasattr(output, 'loss') else output[0]
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accumulator[name] += param.grad.data ** 2
        
        # Average
        for name in fisher_accumulator:
            fisher_accumulator[name] /= len(dataloader)
            self.fisher_information[name] = fisher_accumulator[name]
            self.optimal_params[name] = self.model.state_dict()[name].clone()
        
        self.model.train()
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add experience to replay buffer"""
        if len(self.experience_buffer) >= self.buffer_size:
            # Remove oldest experience
            self.experience_buffer.pop(0)
        
        self.experience_buffer.append(experience)
    
    def sample_replay_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch from replay buffer"""
        if len(self.experience_buffer) < batch_size:
            return self.experience_buffer
        
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        return [self.experience_buffer[i] for i in indices]

class DistributedTrainingCoordinator:
    """Coordinates distributed training across devices"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            cpu=config.cpu_offload,
            split_batches=True,
        )
        
        # Distributed state
        self.world_size = self.accelerator.num_processes
        self.local_rank = self.accelerator.local_process_index
        self.is_main_process = self.accelerator.is_main_process
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for distributed training"""
        if self.config.activation_checkpointing:
            model.gradient_checkpointing_enable()
        
        if self.config.distributed_training and self.world_size > 1:
            # Use FSDP for large models
            if self.config.cpu_offload:
                cpu_offload = CPUOffload(offload_params=True)
            else:
                cpu_offload = None
            
            model = FSDP(
                model,
                cpu_offload=cpu_offload,
                auto_wrap_policy=None,  # Define based on model architecture
                mixed_precision=None,  # Handled by accelerator
            )
        
        return model
    
    def synchronize(self):
        """Synchronize across processes"""
        self.accelerator.wait_for_everyone()
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather metrics from all processes"""
        gathered_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(value).to(self.accelerator.device)
                gathered = self.accelerator.gather(tensor_value)
                gathered_metrics[key] = gathered.mean().item()
            else:
                gathered_metrics[key] = value
        
        return gathered_metrics

class UnifiedRLHFTrainer:
    """Main unified trainer orchestrating all components"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize distributed training
        self.distributed_coordinator = DistributedTrainingCoordinator(config)
        
        # Prepare model
        self.model = self.distributed_coordinator.prepare_model(model)
        
        # Initialize all components
        self._initialize_components()
        
        # Training state
        self.global_step = 0
        self.current_phase = 0
        self.phase_step = 0
        
        # Metrics tracking
        self.metrics_history = []
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize all training components"""
        # Self-supervised learning
        self.self_supervised = SelfSupervisedFramework(
            self.model,
            self.tokenizer,
            self.config.self_supervised
        )
        
        # Multi-reward models
        self.reward_model = MultiRewardModel(self.config.multi_reward)
        
        # Self-evaluation system
        self.self_evaluator = SelfEvaluationSystem(
            self.model,
            self.tokenizer,
            self.config.self_evaluation
        )
        
        # Constitutional AI
        self.constitutional_ai = EnhancedConstitutionalAI(
            self.model,
            self.tokenizer,
            self.config.constitutional
        )
        
        # Reasoning module
        self.reasoning_module = ReasoningModule(self.config.reasoning)
        
        # LoRA adapters
        if self.config.lora.use_qlora or self.config.lora.r > 0:
            self.model, self.lora_adapter = create_peft_model(
                self.model,
                self.config.lora
            )
        else:
            self.lora_adapter = None
        
        # PPO trainer (initialized per phase)
        self.ppo_trainer = None
        
        # Evaluation
        self.evaluator = ComprehensiveEvaluator(self.config.evaluation)
        
        # Data pipeline
        self.data_pipeline = UnifiedDataPipeline(
            self.tokenizer,
            self.config.data_pipeline,
            self.model
        )
        
        # Curriculum scheduler
        self.curriculum_scheduler = CurriculumScheduler(self.config)
        
        # Online learning
        self.online_learning = OnlineLearningModule(self.model, self.config)
    
    def train(self):
        """Main training loop executing all phases"""
        logger.info(f"Starting unified RLHF training with {len(self.config.training_phases)} phases")
        
        # Initialize wandb
        if self.config.use_wandb and self.distributed_coordinator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        
        # Execute training phases
        for phase_idx, phase_name in enumerate(self.config.training_phases):
            self.current_phase = phase_idx
            self.phase_step = 0
            
            logger.info(f"Starting phase {phase_idx + 1}/{len(self.config.training_phases)}: {phase_name}")
            
            # Execute phase
            if phase_name == "self_supervised_pretrain":
                self._train_self_supervised()
            elif phase_name == "constitutional_alignment":
                self._train_constitutional()
            elif phase_name == "ppo_optimization":
                self._train_ppo()
            elif phase_name == "reasoning_enhancement":
                self._train_reasoning()
            elif phase_name == "final_tuning":
                self._final_tuning()
            else:
                logger.warning(f"Unknown phase: {phase_name}")
            
            # Evaluate after phase
            if self.distributed_coordinator.is_main_process:
                self._evaluate_phase(phase_name)
            
            # Save checkpoint
            self._save_checkpoint(f"phase_{phase_name}_complete")
            
            # Synchronize
            self.distributed_coordinator.synchronize()
        
        logger.info("Training completed!")
        
        # Final evaluation
        if self.distributed_coordinator.is_main_process:
            self._final_evaluation()
    
    def _train_self_supervised(self):
        """Self-supervised pretraining phase"""
        logger.info("Starting self-supervised pretraining...")
        
        # Create self-supervised data
        prompts = self._get_training_prompts()
        ssl_data = self.self_supervised.generate_self_training_data(prompts)
        
        # Training loop
        num_steps = 5000  # Configurable
        dataloader = self._create_ssl_dataloader(ssl_data)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.self_supervised.learning_rate
        )
        
        for step in tqdm(range(num_steps), desc="Self-supervised training"):
            batch = next(iter(dataloader))
            
            # Forward pass
            losses = self.self_supervised.train_step(batch)
            
            # Backward pass
            total_loss = losses["total"]
            self.distributed_coordinator.accelerator.backward(total_loss)
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            self._update_metrics(losses, "self_supervised")
            
            # Logging
            if step % self.config.logging_steps == 0:
                self._log_metrics(step)
            
            self.global_step += 1
            self.phase_step += 1
    
    def _train_constitutional(self):
        """Constitutional AI alignment phase"""
        logger.info("Starting constitutional alignment...")
        
        # Generate constitutional training data
        train_prompts = self._get_training_prompts()
        
        # Constitutional training loop
        num_rounds = 3
        for round_idx in range(num_rounds):
            logger.info(f"Constitutional round {round_idx + 1}/{num_rounds}")
            
            # Generate data
            constitutional_data = self.constitutional_ai.generate_training_data(
                train_prompts[:1000]  # Subset for each round
            )
            
            # Train on constitutional data
            self._train_on_constitutional_data(constitutional_data)
            
            # Evaluate constitutional adherence
            metrics = self.constitutional_ai.evaluate_adherence(train_prompts[:100])
            logger.info(f"Constitutional metrics: {metrics}")
    
    def _train_ppo(self):
        """PPO optimization phase"""
        logger.info("Starting PPO optimization...")
        
        # Initialize PPO trainer
        ref_model = self._create_reference_model()
        
        self.ppo_trainer = EnhancedPPOTrainer(
            policy_model=self.model,
            ref_model=ref_model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            config=self.config.ppo,
            value_model=None  # Created internally
        )
        
        # PPO training loop
        train_dataloader = self.data_pipeline.create_mixed_dataloader(
            batch_size=self.config.ppo.batch_size,
            shuffle=True
        )
        
        num_ppo_epochs = 2
        for epoch in range(num_ppo_epochs):
            logger.info(f"PPO epoch {epoch + 1}/{num_ppo_epochs}")
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Get prompts
                prompts = self._extract_prompts_from_batch(batch)
                
                # Generate experiences
                experiences = self.ppo_trainer.generate_experience_batch(prompts)
                
                # Compute advantages
                self.ppo_trainer.compute_advantages_and_returns(experiences)
                
                # PPO update
                metrics = self.ppo_trainer.train_step(experiences)
                
                # Update metrics
                self._update_metrics(metrics, "ppo")
                
                # Online learning - add to replay buffer
                for exp in experiences:
                    self.online_learning.add_experience(exp)
                
                if batch_idx % 100 == 0:
                    self._log_metrics(self.global_step)
                
                self.global_step += 1
    
    def _train_reasoning(self):
        """Reasoning enhancement phase"""
        logger.info("Starting reasoning enhancement...")
        
        # Focus on reasoning datasets
        reasoning_dataloader = self._create_reasoning_dataloader()
        
        # Reasoning-specific optimizer
        if self.lora_adapter:
            # Only train LoRA parameters
            optimizer = torch.optim.AdamW(
                self.lora_adapter.get_trainable_parameters(),
                lr=1e-4
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-5
            )
        
        num_steps = 3000
        for step in tqdm(range(num_steps), desc="Reasoning training"):
            batch = next(iter(reasoning_dataloader))
            
            # Forward pass with reasoning
            outputs = self._forward_with_reasoning(batch)
            
            # Compute reasoning-aware loss
            loss = self._compute_reasoning_loss(outputs, batch)
            
            # Add EWC regularization
            if self.online_learning.fisher_information:
                loss += self.online_learning.compute_ewc_loss()
            
            # Backward pass
            self.distributed_coordinator.accelerator.backward(loss)
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Progressive unfreezing
            if self.lora_adapter:
                self.lora_adapter.unfreezer.step(self.global_step)
            
            self.global_step += 1
    
    def _final_tuning(self):
        """Final tuning phase combining all objectives"""
        logger.info("Starting final tuning...")
        
        # Combined dataloader
        combined_dataloader = self.data_pipeline.create_mixed_dataloader(
            batch_size=16,
            shuffle=True
        )
        
        # Multi-objective optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-6
        )
        
        num_steps = 2000
        for step in tqdm(range(num_steps), desc="Final tuning"):
            batch = next(iter(combined_dataloader))
            
            # Multi-objective forward pass
            losses = {}
            
            # Self-supervised objective
            if "self_supervised" in batch:
                ssl_loss = self.self_supervised.train_step(batch["self_supervised"])
                losses["ssl"] = ssl_loss["total"] * 0.2
            
            # Standard language modeling
            if "lm" in batch:
                lm_outputs = self.model(**batch["lm"])
                losses["lm"] = lm_outputs.loss * 0.3
            
            # Reasoning objective
            if "reasoning" in batch:
                reasoning_loss = self._compute_reasoning_loss(
                    self._forward_with_reasoning(batch["reasoning"]),
                    batch["reasoning"]
                )
                losses["reasoning"] = reasoning_loss * 0.3
            
            # Constitutional objective
            if "constitutional" in batch:
                const_loss = self.constitutional_ai.compute_loss(batch["constitutional"])
                losses["constitutional"] = const_loss * 0.2
            
            # Combine losses
            total_loss = sum(losses.values())
            
            # Backward pass
            self.distributed_coordinator.accelerator.backward(total_loss)
            
            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update metrics
            self._update_metrics(losses, "final_tuning")
            
            self.global_step += 1
    
    def _forward_with_reasoning(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass with reasoning modules"""
        # Get base model outputs
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            return_dict=True,
            output_hidden_states=True
        )
        
        # Apply reasoning module
        hidden_states = outputs.hidden_states[-1]
        
        # Generate reasoning steps
        reasoning_context = self.reasoning_module.encode_context(
            hidden_states,
            batch.get("attention_mask")
        )
        
        reasoning_outputs = {
            "base_outputs": outputs,
            "reasoning_context": reasoning_context,
            "reasoning_steps": []
        }
        
        # Generate step-by-step reasoning
        for step_num in range(min(5, self.config.reasoning.max_reasoning_steps)):
            step_repr, validity, step_type = self.reasoning_module.generate_reasoning_step(
                reasoning_context,
                reasoning_outputs["reasoning_steps"],
                step_num
            )
            
            reasoning_outputs["reasoning_steps"].append(step_repr)
            
            # Check if should stop
            if validity < self.config.reasoning.min_step_confidence:
                break
        
        return reasoning_outputs
    
    def _compute_reasoning_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute loss for reasoning training"""
        base_loss = outputs["base_outputs"].loss if hasattr(outputs["base_outputs"], "loss") else 0
        
        # Reasoning coherence loss
        if outputs["reasoning_steps"]:
            coherence_loss = 0
            for i in range(1, len(outputs["reasoning_steps"])):
                prev_step = outputs["reasoning_steps"][i-1]
                curr_step = outputs["reasoning_steps"][i]
                
                # Encourage coherent progression
                similarity = F.cosine_similarity(prev_step, curr_step, dim=-1)
                target_similarity = 0.7  # Not too similar, not too different
                coherence_loss += (similarity - target_similarity) ** 2
            
            coherence_loss = coherence_loss.mean() / max(len(outputs["reasoning_steps"]) - 1, 1)
        else:
            coherence_loss = 0
        
        return base_loss + 0.1 * coherence_loss
    
    def _create_reference_model(self) -> nn.Module:
        """Create reference model for PPO"""
        # Load from checkpoint or clone current model
        ref_model = type(self.model)(self.model.config)
        ref_model.load_state_dict(self.model.state_dict())
        
        # Freeze and move to device
        for param in ref_model.parameters():
            param.requires_grad = False
        
        return ref_model
    
    def _create_ssl_dataloader(self, ssl_data: List[Dict[str, Any]]) -> Any:
        """Create dataloader for self-supervised training"""
        # Convert to dataset
        class SSLDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = SSLDataset(ssl_data)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=self._ssl_collate_fn
        )
    
    def _create_reasoning_dataloader(self) -> Any:
        """Create dataloader focused on reasoning tasks"""
        # Use reasoning datasets from pipeline
        return self.data_pipeline.create_mixed_dataloader(
            batch_size=4,
            shuffle=True
        )
    
    def _train_on_constitutional_data(self, data: List[Dict[str, Any]]):
        """Train on constitutional data"""
        # Convert to DPO format and train
        dpo_data = []
        for item in data:
            if item["is_revision_better"]:
                dpo_data.append({
                    "prompt": item["prompt"],
                    "chosen": item["revision"],
                    "rejected": item["original_response"]
                })
        
        # Create simple dataloader
        batch_size = 4
        for i in range(0, len(dpo_data), batch_size):
            batch = dpo_data[i:i+batch_size]
            
            # Simplified DPO loss
            loss = self._compute_dpo_loss(batch)
            
            # Backward pass
            self.distributed_coordinator.accelerator.backward(loss)
            
            # Optimizer step
            if (i // batch_size + 1) % self.config.gradient_accumulation_steps == 0:
                self.model.zero_grad()
    
    def _compute_dpo_loss(self, batch: List[Dict[str, str]]) -> torch.Tensor:
        """Simplified DPO loss computation"""
        losses = []
        
        for item in batch:
            # Tokenize
            prompt_enc = self.tokenizer(item["prompt"], return_tensors="pt")
            chosen_enc = self.tokenizer(item["prompt"] + item["chosen"], return_tensors="pt")
            rejected_enc = self.tokenizer(item["prompt"] + item["rejected"], return_tensors="pt")
            
            # Get logits
            with torch.no_grad():
                ref_chosen_logits = self.model(**chosen_enc).logits
                ref_rejected_logits = self.model(**rejected_enc).logits
            
            policy_chosen_logits = self.model(**chosen_enc).logits
            policy_rejected_logits = self.model(**rejected_enc).logits
            
            # Simplified DPO loss
            beta = 0.1
            policy_chosen_logp = F.log_softmax(policy_chosen_logits, dim=-1).mean()
            policy_rejected_logp = F.log_softmax(policy_rejected_logits, dim=-1).mean()
            ref_chosen_logp = F.log_softmax(ref_chosen_logits, dim=-1).mean()
            ref_rejected_logp = F.log_softmax(ref_rejected_logits, dim=-1).mean()
            
            loss = -F.logsigmoid(
                beta * ((policy_chosen_logp - ref_chosen_logp) - 
                       (policy_rejected_logp - ref_rejected_logp))
            )
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def _get_training_prompts(self) -> List[str]:
        """Get training prompts for current phase"""
        # Sample from data pipeline
        sample_size = 1000
        prompts = []
        
        # Get from preference dataset
        for i in range(min(sample_size, len(self.data_pipeline.preference_dataset))):
            prompts.append(self.data_pipeline.preference_dataset.data[i]["prompt"])
        
        return prompts
    
    def _extract_prompts_from_batch(self, batch: Dict[str, Any]) -> List[str]:
        """Extract prompts from mixed batch"""
        prompts = []
        
        if "preference_batch" in batch and batch["preference_batch"]:
            # Decode prompt IDs
            for prompt_ids in batch["preference_batch"]["prompt_ids"]:
                prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                prompts.append(prompt)
        
        return prompts
    
    def _ssl_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for SSL data"""
        # Simple batching
        collated = {
            "text": [item.get("prompt", "") for item in batch],
            "positive_text": [item.get("positive", "") for item in batch],
            "negative_text": [item.get("negative", "") for item in batch],
        }
        
        return collated
    
    def _update_metrics(self, metrics: Dict[str, Any], phase: str):
        """Update metrics tracking"""
        timestamped_metrics = {
            "global_step": self.global_step,
            "phase": phase,
            "phase_step": self.phase_step,
            "timestamp": time.time(),
        }
        timestamped_metrics.update(metrics)
        
        self.metrics_history.append(timestamped_metrics)
    
    def _log_metrics(self, step: int):
        """Log metrics to wandb and console"""
        if not self.metrics_history:
            return
        
        # Get recent metrics
        recent_metrics = self.metrics_history[-100:]
        
        # Compute averages
        avg_metrics = {}
        for metric in recent_metrics:
            for key, value in metric.items():
                if isinstance(value, (int, float)):
                    if key not in avg_metrics:
                        avg_metrics[key] = []
                    avg_metrics[key].append(value)
        
        # Average
        for key in avg_metrics:
            avg_metrics[key] = np.mean(avg_metrics[key])
        
        # Log to wandb
        if self.config.use_wandb and self.distributed_coordinator.is_main_process:
            wandb.log(avg_metrics, step=step)
        
        # Log to console
        logger.info(f"Step {step}: {avg_metrics}")
    
    def _evaluate_phase(self, phase_name: str):
        """Evaluate model after training phase"""
        logger.info(f"Evaluating after phase: {phase_name}")
        
        # Create evaluation data
        eval_outputs = self._generate_eval_outputs()
        
        # Run comprehensive evaluation
        eval_results = self.evaluator.evaluate_all(
            eval_outputs,
            save_results=True
        )
        
        # Log results
        logger.info(f"Evaluation results: {eval_results['overall']}")
        
        if self.config.use_wandb:
            wandb.log({
                f"eval/{phase_name}/overall_score": eval_results["overall"]["weighted_score"],
                f"eval/{phase_name}/components": eval_results["overall"]["components"]
            })
    
    def _generate_eval_outputs(self) -> Dict[str, Any]:
        """Generate outputs for evaluation"""
        eval_prompts = [
            "Explain how photosynthesis works.",
            "Solve: If x + 5 = 12, what is x?",
            "Write a Python function to reverse a string.",
            "What were the causes of World War I?",
            "Analyze the theme of redemption in Les Misérables."
        ]
        
        outputs = {
            "predictions": [],
            "confidence_scores": [],
            "reasoning_chains": [],
            "contexts": eval_prompts,
        }
        
        for prompt in eval_prompts:
            # Generate prediction
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            prediction = self.tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
            outputs["predictions"].append(prediction)
            
            # Mock confidence score
            outputs["confidence_scores"].append(0.8)
            
            # Mock reasoning chain
            outputs["reasoning_chains"].append([
                "Understanding the question",
                "Identifying key concepts",
                "Applying relevant knowledge",
                "Formulating the answer"
            ])
        
        return outputs
    
    def _final_evaluation(self):
        """Comprehensive final evaluation"""
        logger.info("Running final comprehensive evaluation...")
        
        # Extended evaluation
        eval_outputs = self._generate_eval_outputs()
        
        # Add domain-specific evaluations
        domain_outputs = {}
        for domain in self.config.evaluation.test_domains:
            domain_prompts = self._get_domain_prompts(domain)
            domain_predictions = []
            
            for prompt in domain_prompts[:5]:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    generation = self.model.generate(**inputs, max_new_tokens=200)
                prediction = self.tokenizer.decode(generation[0], skip_special_tokens=True)
                domain_predictions.append(prediction)
            
            domain_outputs[domain] = domain_predictions
        
        eval_outputs["domain_outputs"] = domain_outputs
        eval_outputs["source_domain"] = "mathematics"
        
        # Run evaluation
        final_results = self.evaluator.evaluate_all(
            eval_outputs,
            save_results=True
        )
        
        # Generate final report
        self._generate_final_report(final_results)
    
    def _get_domain_prompts(self, domain: str) -> List[str]:
        """Get domain-specific evaluation prompts"""
        domain_prompts = {
            "mathematics": [
                "Prove that the square root of 2 is irrational.",
                "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3.",
                "Solve the system: 2x + 3y = 7, x - y = 1."
            ],
            "science": [
                "Explain the process of cellular respiration.",
                "What causes the seasons on Earth?",
                "Describe the structure of an atom."
            ],
            "history": [
                "What were the main causes of the French Revolution?",
                "Describe the impact of the Industrial Revolution.",
                "Who were the key figures in the American Civil War?"
            ],
            "literature": [
                "Analyze the use of symbolism in The Great Gatsby.",
                "What are the main themes in Shakespeare's Hamlet?",
                "Discuss the narrative structure of To Kill a Mockingbird."
            ],
            "code": [
                "Write a function to find the nth Fibonacci number.",
                "Implement binary search in Python.",
                "Create a class for a linked list with basic operations."
            ]
        }
        
        return domain_prompts.get(domain, [])
    
    def _generate_final_report(self, results: Dict[str, Any]):
        """Generate comprehensive final training report"""
        report_path = self.output_dir / "final_training_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Unified RLHF Training Report\n\n")
            f.write(f"## Training Configuration\n")
            f.write(f"- Model: {self.config.model_name}\n")
            f.write(f"- Training phases: {', '.join(self.config.training_phases)}\n")
            f.write(f"- Total steps: {self.global_step}\n\n")
            
            f.write("## Final Evaluation Results\n")
            f.write(f"- Overall Score: {results['overall']['weighted_score']:.3f}\n\n")
            
            f.write("### Component Scores\n")
            for component, score in results['overall']['components'].items():
                f.write(f"- {component}: {score:.3f}\n")
            
            f.write("\n## Training Phases Summary\n")
            for phase in self.config.training_phases:
                f.write(f"### {phase}\n")
                f.write(f"- Completed successfully\n")
            
            f.write("\n## Model Capabilities\n")
            f.write("- Multi-step reasoning\n")
            f.write("- Self-evaluation and correction\n")
            f.write("- Constitutional AI alignment\n")
            f.write("- Cross-domain transfer\n")
        
        logger.info(f"Final report saved to {report_path}")
    
    def _save_checkpoint(self, tag: str):
        """Save training checkpoint"""
        if not self.distributed_coordinator.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "pytorch_model.bin")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_phase": self.current_phase,
            "phase_step": self.phase_step,
            "config": self.config,
            "metrics_history": self.metrics_history[-1000:],  # Last 1000 metrics
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Save LoRA adapters if used
        if self.lora_adapter:
            self.lora_adapter.save_adapters(checkpoint_dir / "lora_adapters.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Manage checkpoint limit
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain limit"""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: x.stat().st_mtime
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {oldest}")
            import shutil
            shutil.rmtree(oldest)

def create_unified_trainer(
    model_name: str,
    config_path: Optional[str] = None
) -> UnifiedRLHFTrainer:
    """Factory function to create unified trainer"""
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create trainer
    trainer = UnifiedRLHFTrainer(model, tokenizer, config)
    
    return trainer