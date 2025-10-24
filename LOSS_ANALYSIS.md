# Loss Components Analysis

## Currently Used in Training Pipeline

### Primary Loss Systems (enhanced_trainer.py)

1. **DeepSeekLoss** (deepseek_loss.py) - Used when `use_multi_token_prediction=True`
   - Contains:
     - `TemperatureScaledCrossEntropy` - Main loss with adaptive temperature & label smoothing
     - `MultiTokenPredictionLoss` - Predicts multiple future tokens
     - `AuxiliaryFreeMoEBalancer` - Gradient-based MoE load balancing

2. **AdaptiveMTPLoss** (adaptive_mtp_loss.py) - Used with AdaptiveMTPModel
   - Separate implementation from DeepSeekLoss
   - Confidence-weighted additional token losses
   - Used in integration_example.py

3. **Repetition Penalties** (repetition_penalty_loss.py)
   - `NGramRepetitionPenalty` - Penalizes repeated n-grams (default: enabled, weight=2.0)
   - `SequenceRepetitionDetector` - Detects immediate token repetition (default: enabled, weight=3.0)

4. **CompositeLoss** (advanced_losses.py) - Used when DeepSeekLoss is NOT active
   - Wraps multiple specialized losses:
     - `FocalLoss` - Class imbalance handling
     - `ContrastiveLoss` - Representation learning
     - `DiversityLoss` - Expert diversity in MoE
     - `AuxiliaryLoss` - MoE load balancing (traditional method)

5. **AdaptiveLossScaling** (advanced_losses.py)
   - Dynamically weights multiple loss components
   - Learnable weights in log space

## NOT Used in Training

- **AntiRepetitionLoss** (anti_repetition_loss.py) - ❌ NOT imported anywhere
  - Redundant with NGramRepetitionPenalty and SequenceRepetitionDetector
  - Different implementation of same concept

## Redundancy Analysis

### 1. Repetition Handling (3 implementations!)
- `anti_repetition_loss.py` - **UNUSED**
- `repetition_penalty_loss.py` - **USED** (NGramRepetitionPenalty, SequenceRepetitionDetector)
- `deepseek_loss.py` - Contains EOS penalty in TemperatureScaledCrossEntropy

### 2. MTP (2 implementations)
- `adaptive_mtp_loss.py` - Used with AdaptiveMTPModel
- `deepseek_loss.py` - Contains MultiTokenPredictionLoss

### 3. MoE Balancing (2 implementations)
- `advanced_losses.py::AuxiliaryLoss` - Traditional auxiliary loss
- `deepseek_loss.py::AuxiliaryFreeMoEBalancer` - Gradient-based (preferred)

## Consolidation Plan

### File 1: `core_losses.py`
**Core training objectives**
- TemperatureScaledCrossEntropy (from deepseek_loss.py)
- AdaptiveLossScaling (from advanced_losses.py)
- CompositeLoss (from advanced_losses.py)
- FocalLoss (from advanced_losses.py)
- LabelSmoothingLoss (from advanced_losses.py)
- PerplexityLoss (from advanced_losses.py)

### File 2: `regularization_losses.py`
**Regularization and anti-collapse mechanisms**
- NGramRepetitionPenalty (from repetition_penalty_loss.py)
- SequenceRepetitionDetector (from repetition_penalty_loss.py)
- DiversityLoss (from advanced_losses.py)
- ConsistencyLoss (from advanced_losses.py)
- ContrastiveLoss (from advanced_losses.py)

### File 3: `mtp_moe_losses.py`
**Multi-token prediction and MoE-specific losses**
- MultiTokenPredictionLoss (from deepseek_loss.py)
- AdaptiveMTPLoss (from adaptive_mtp_loss.py)
- AuxiliaryFreeMoEBalancer (from deepseek_loss.py)
- AuxiliaryLoss (from advanced_losses.py - legacy MoE balancing)
- DeepSeekLoss (from deepseek_loss.py - combined loss)

## Files to Remove
- `anti_repetition_loss.py` - Unused, redundant with repetition_penalty_loss.py

## Backward Compatibility
All existing imports must continue to work:
```python
from Ava.losses import DeepSeekLoss, NGramRepetitionPenalty, SequenceRepetitionDetector
from Ava.losses import AdaptiveLossScaling, CompositeLoss
from Ava.losses.adaptive_mtp_loss import AdaptiveMTPLoss
```
