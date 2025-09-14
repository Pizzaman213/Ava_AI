"""
Advanced RLHF with Self-Supervised Learning Framework
Comprehensive reinforcement learning from human feedback with self-improvement capabilities
"""

from .self_supervised import (
    SelfSupervisedFramework,
    SelfSupervisedConfig,
    ConsistencyTrainer,
    ConfidenceCalibration,
    ReasoningChainModule,
    ContrastiveLearner,
)

from .multi_reward import (
    MultiRewardModel,
    MultiRewardConfig,
    FactualAccuracyReward,
    CoherenceReward,
    HelpfulnessReward,
    SafetyReward,
    RewardAggregator,
)

from .self_evaluation import (
    SelfEvaluationSystem,
    SelfEvaluationConfig,
    FactChecker,
    LogicalConsistencyChecker,
    ConfidenceScorer,
    SelfCorrectionPipeline,
)

from .enhanced_ppo import (
    EnhancedPPOTrainer,
    EnhancedPPOConfig,
    AdaptiveKLController,
    ReplayBufferManager,
    MetaLearningOptimizer,
)

from .constitutional_enhanced import (
    EnhancedConstitutionalAI,
    ConstitutionalConfig,
    IterativeRefinement,
    ConstitutionalDataAugmentor,
)

from .reasoning import (
    ReasoningModule,
    ReasoningConfig,
    ChainOfThoughtGenerator,
    StepVerifier,
    LogicalReasoningHead,
    MathematicalReasoningHead,
    ScientificReasoningHead,
    CodeReasoningHead,
)

from .lora_integration import (
    LoRAAdapter,
    LoRAConfig,
    QLoRAAdapter,
    ReasoningLoRA,
    TaskSpecificLoRA,
    ProgressiveUnfreezing,
)

from .evaluation import (
    ComprehensiveEvaluator,
    EvaluationConfig,
    FactualAccuracyMetric,
    LogicalConsistencyMetric,
    CalibrationMetric,
    HallucinationDetector,
    ReasoningQualityAssessor,
    CrossDomainTransferEvaluator,
)

from .unified_trainer import (
    UnifiedRLHFTrainer,
    TrainingConfig,
    CurriculumScheduler,
    OnlineLearningModule,
    DistributedTrainingCoordinator,
)

from .data_pipeline import (
    UnifiedDataPipeline,
    DataPipelineConfig,
)

__all__ = [
    # Self-supervised components
    "SelfSupervisedFramework",
    "SelfSupervisedConfig",
    "ConsistencyTrainer",
    "ConfidenceCalibration",
    "ReasoningChainModule",
    "ContrastiveLearner",
    
    # Multi-reward models
    "MultiRewardModel",
    "MultiRewardConfig",
    "FactualAccuracyReward",
    "CoherenceReward",
    "HelpfulnessReward",
    "SafetyReward",
    "RewardAggregator",
    
    # Self-evaluation
    "SelfEvaluationSystem",
    "SelfEvaluationConfig",
    "FactChecker",
    "LogicalConsistencyChecker",
    "ConfidenceScorer",
    "SelfCorrectionPipeline",
    
    # Enhanced PPO
    "EnhancedPPOTrainer",
    "EnhancedPPOConfig",
    "AdaptiveKLController",
    "ReplayBufferManager",
    "MetaLearningOptimizer",
    
    # Constitutional AI
    "EnhancedConstitutionalAI",
    "ConstitutionalConfig",
    "IterativeRefinement",
    "ConstitutionalDataAugmentor",
    
    # Reasoning modules
    "ReasoningModule",
    "ReasoningConfig",
    "ChainOfThoughtGenerator",
    "StepVerifier",
    "LogicalReasoningHead",
    "MathematicalReasoningHead",
    "ScientificReasoningHead",
    "CodeReasoningHead",
    
    # LoRA integration
    "LoRAAdapter",
    "LoRAConfig",
    "QLoRAAdapter",
    "ReasoningLoRA",
    "TaskSpecificLoRA",
    "ProgressiveUnfreezing",
    
    # Evaluation
    "ComprehensiveEvaluator",
    "EvaluationConfig",
    "FactualAccuracyMetric",
    "LogicalConsistencyMetric",
    "CalibrationMetric",
    "HallucinationDetector",
    "ReasoningQualityAssessor",
    "CrossDomainTransferEvaluator",
    
    # Unified trainer
    "UnifiedRLHFTrainer",
    "TrainingConfig",
    "CurriculumScheduler",
    "OnlineLearningModule",
    "DistributedTrainingCoordinator",
    
    # Data pipeline
    "UnifiedDataPipeline",
    "DataPipelineConfig",
]