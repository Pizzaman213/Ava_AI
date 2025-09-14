"""
Advanced Optimizers and Learning Rate Schedulers
Implements AdamW, Lion, and other state-of-the-art optimizers
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

class AdamW(Optimizer):
    """
    AdamW optimizer with weight decay fix and optional features
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        use_8bit: bool = False,
        stable_embedding_updates: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            use_8bit=use_8bit,
            stable_embedding_updates=stable_embedding_updates,
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        # Maintains max of all exp_avg_sq values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    # Maintains max of all 2nd moment running avg
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use max for normalizing running avg of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                if group['stable_embedding_updates'] and hasattr(p, 'is_embedding'):
                    # Stable updates for embeddings
                    p.data.addcdiv_(exp_avg, denom, value=-step_size * 0.1)
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

class Lion(Optimizer):
    """
    Lion optimizer - a more memory-efficient optimizer
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Interpolation
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                
                # Bias correction and update
                p.data.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update exponential average
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss

class AdaFactor(Optimizer):
    """
    AdaFactor optimizer - memory-efficient adaptive learning rates
    """
    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        cliping_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        if lr is not None and lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(
            lr=lr,
            eps=eps,
            cliping_threshold=cliping_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)
    
    def _get_lr(self, param_state, param_scale):
        """Compute learning rate"""
        if param_state["step"] == 0:
            min_step = 1e-6 * param_state["step"]
        else:
            min_step = 1e-2
        
        rel_step_sz = min_step if self.defaults["warmup_init"] else 1.0
        rel_step_sz = min(rel_step_sz, 1.0 / math.sqrt(param_state["step"]))
        
        param_scale = 1.0 if param_scale is None else param_scale
        return param_scale * rel_step_sz
    
    def _get_options(self, param_group, param_shape):
        """Determine factorization options"""
        factored = len(param_shape) >= 2 and param_shape[0] * param_shape[1] >= 32
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaFactor does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    
                    factored, use_first_moment = self._get_options(group, p.shape)
                    
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(p.shape[0])
                        state["exp_avg_sq_col"] = torch.zeros(p.shape[1:].numel())
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    
                    state["RMS"] = 0
                
                state["step"] += 1
                lr = group["lr"]
                
                if lr is None:
                    lr = self._get_lr(state, group["scale_parameter"])
                
                # Exponential moving average of gradient values
                if "exp_avg" in state:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(grad, alpha=1 - group["beta1"])
                
                # RMS
                rms = self._rms(p, grad, group, state)
                state["RMS"] = rms
                
                # Adaptive learning rate
                if "exp_avg_sq" in state:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(self._decay_rate(state["step"])).addcmul_(grad, grad, value=1)
                    update = grad / (exp_avg_sq.sqrt().add_(group["eps"][1]))
                else:
                    row_var = state["exp_avg_sq_row"]
                    col_var = state["exp_avg_sq_col"]
                    
                    row_var.mul_(self._decay_rate(state["step"]))
                    col_var.mul_(self._decay_rate(state["step"]))
                    
                    # Compute row and column variances
                    grad_sq = grad * grad
                    row_mean = grad_sq.mean(dim=list(range(1, len(p.shape))))
                    col_mean = grad_sq.mean(dim=0)
                    
                    row_var.add_(row_mean, alpha=1 - self._decay_rate(state["step"]))
                    col_var.add_(col_mean, alpha=1 - self._decay_rate(state["step"]))
                    
                    # Compute update
                    update = grad / (row_var[:, None] * col_var[None, :]).sqrt().add_(group["eps"][1])
                
                # Clipping
                if group["cliping_threshold"] > 0:
                    update.div_(max(1.0, self._rms(p, update, group, state) / group["cliping_threshold"]))
                
                # Weight decay
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * lr)
                
                # Apply update
                p.data.add_(update, alpha=-lr)
        
        return loss
    
    def _rms(self, p, grad, group, state):
        """Root mean square"""
        return grad.norm() / math.sqrt(grad.numel())
    
    def _decay_rate(self, step):
        """Decay rate for second moment"""
        return 1 - (step + 1) ** self.defaults["decay_rate"]

def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    use_8bit: bool = False,
    **kwargs
) -> Optimizer:
    """
    Create optimizer with specified configuration
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer (adamw, lion, adafactor)
        lr: Learning rate
        weight_decay: Weight decay
        adam_beta1: Adam beta1
        adam_beta2: Adam beta2
        adam_epsilon: Adam epsilon
        use_8bit: Whether to use 8-bit optimization
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Get parameters with weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for biases and layer norm
        if "bias" in name or "layernorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    # Create optimizer
    if optimizer_type.lower() == "adamw":
        if use_8bit:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    lr=lr,
                    betas=(adam_beta1, adam_beta2),
                    eps=adam_epsilon,
                    **kwargs
                )
            except ImportError:
                logger.warning("bitsandbytes not available, using standard AdamW")
                optimizer = AdamW(
                    param_groups,
                    lr=lr,
                    betas=(adam_beta1, adam_beta2),
                    eps=adam_epsilon,
                    **kwargs
                )
        else:
            optimizer = AdamW(
                param_groups,
                lr=lr,
                betas=(adam_beta1, adam_beta2),
                eps=adam_epsilon,
                **kwargs
            )
    
    elif optimizer_type.lower() == "lion":
        optimizer = Lion(
            param_groups,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            **kwargs
        )
    
    elif optimizer_type.lower() == "adafactor":
        optimizer = AdaFactor(
            param_groups,
            lr=lr,
            relative_step=False,
            scale_parameter=False,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with lr={lr}, weight_decay={weight_decay}")
    
    return optimizer

class WarmupCosineScheduler(LambdaLR):
    """Cosine learning rate scheduler with linear warmup"""
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        
        super().__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch
        )
    
    def lr_lambda(self, current_step: int) -> float:
        """Compute learning rate multiplier"""
        if current_step < self.num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress))
        )

class InverseSqrtScheduler(LambdaLR):
    """Inverse square root learning rate scheduler"""
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        
        super().__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch
        )
    
    def lr_lambda(self, current_step: int) -> float:
        """Compute learning rate multiplier"""
        if current_step < self.num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.num_warmup_steps))
        
        # Inverse square root decay
        return float(self.num_warmup_steps ** 0.5) / float(current_step ** 0.5)

def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 0,
    num_training_steps: int = None,
    **kwargs
) -> LambdaLR:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler (cosine, inverse_sqrt, linear)
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == "cosine":
        if num_training_steps is None:
            raise ValueError("num_training_steps required for cosine scheduler")
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    
    elif scheduler_type == "inverse_sqrt":
        scheduler = InverseSqrtScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            **kwargs
        )
    
    elif scheduler_type == "linear":
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_warmup_steps)
                )
            )
        
        scheduler = LambdaLR(optimizer, lr_lambda, **kwargs)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler with warmup_steps={num_warmup_steps}")
    
    return scheduler