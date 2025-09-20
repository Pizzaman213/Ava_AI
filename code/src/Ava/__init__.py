"""
Ava MoE++ Architecture Package

A comprehensive implementation of advanced LLM architectures with:
- Enhanced Mixture of Experts (MoE++) with hierarchical routing
- Mixture of Depths (MoD) for dynamic computation
- Mixture of Heads (MoH) for adaptive attention
- Mixture of Activations (MoA) for dynamic activation selection
- Retrieval-Augmented Generation (RAG) capabilities
- Multi-modal cross-attention layers
- Advanced loss functions and training techniques
- Gradient surgery for multi-task learning
- Comprehensive evaluation suite
- Model quantization and optimization
- Production-ready serving infrastructure
"""

# Core models and configurations
from .models.moe_model import EnhancedMoEModel, EnhancedMoEConfig

# Layer components
from .layers.experts import ExpertBalancer, SparseExpert
from .layers.routing import (
    ExpertSelector, MoEPlusPlusLayer,
    SwitchTransformerRouting, GSERouting,
    HashingExpertRouting, StochasticExpertRouting
)
from .layers.attention import (
    EnhancedMultiheadAttention, RotaryPositionEmbedding,
    ALiBiPositionEmbedding, FlashAttention
)
from .layers.mixture_of_heads import (
    MixtureOfHeads, AdaptiveHeadAttention
)
from .layers.cross_attention import (
    MultiModalCrossAttention, PerceiversCrossAttention,
    AdaptiveCrossAttention, HierarchicalCrossAttention
)
from .layers.mixture_of_activations import (
    MixtureOfActivations, AdaptiveActivation,
    ContextualActivation, HierarchicalActivation
)

# Retrieval and RAG
from .retrieval import (
    RAGSystem, AdaptiveRAG, DenseRetriever,
    KnowledgeBase, RAGFusion
)

# Training utilities
from .training import (
    GradientSurgeon, AdaptiveGradientSurgeon,
    GradientConflictAnalyzer,
    CosineAnnealingWarmRestarts, OneCycleLR, PolynomialDecayLR, AdaptiveLRScheduler,
    NoisyStudentScheduler, SchedulerFactory,
    ProgressiveTrainingConfig, CurriculumLearning, GrowLengthScheduler,
    DynamicBatchSizer, ProgressiveModelScaler, ProgressiveTrainer
)

# Loss functions
from .losses import (
    ContrastiveLoss, FocalLoss, LabelSmoothingLoss,
    DiversityLoss, AuxiliaryLoss, ConsistencyLoss,
    PerplexityLoss, AdaptiveLossScaling, CompositeLoss
)

# Evaluation
from .evaluation.comprehensive_eval import (
    ComprehensiveEvaluator, PerplexityEvaluator,
    BLEUEvaluator, ROUGEEvaluator, ToxicityEvaluator,
    BiasEvaluator, CoherenceEvaluator
)

# Optimization
from .optimization import (
    ModelQuantizer, LinearQuantized, DynamicQuantization,
    INT4Quantization, QuantizationConfig, quantize_model_pipeline,
    LionOptimizer, SophiaOptimizer, AdaFactorOptimizer,
    OptimizerFactory,
    FP8Format, FP8Config, FP8Handler, FP8Linear, FP8MultiHeadAttention,
    FP8LayerNorm, FP8TransformerLayer, FP8ModelWrapper,
    create_fp8_model, benchmark_fp8_training
)

# Memory and continual learning
from .memory import (
    EpisodicMemoryBank, MemoryEntry, MemoryRetriever,
    AdaptiveMemoryManager, ExperienceReplay
)

# Serving
from .serving.fastapi_server import LLMServer

__version__ = "2.0.0"

__all__ = [
    # Core
    "EnhancedMoEModel",
    "EnhancedMoEConfig",

    # Expert layers
    "ExpertBalancer",
    "SparseExpert",
    "ExpertSelector",
    "MoEPlusPlusLayer",

    # Routing variants
    "SwitchTransformerRouting",
    "GSERouting",
    "HashingExpertRouting",
    "StochasticExpertRouting",

    # Attention mechanisms
    "EnhancedMultiheadAttention",
    "RotaryPositionEmbedding",
    "ALiBiPositionEmbedding",
    "FlashAttention",
    "MixtureOfHeads",
    "AdaptiveHeadAttention",

    # Cross-attention
    "MultiModalCrossAttention",
    "PerceiversCrossAttention",
    "AdaptiveCrossAttention",
    "HierarchicalCrossAttention",

    # Activations
    "MixtureOfActivations",
    "AdaptiveActivation",
    "ContextualActivation",
    "HierarchicalActivation",

    # RAG system
    "RAGSystem",
    "AdaptiveRAG",
    "DenseRetriever",
    "KnowledgeBase",
    "RAGFusion",

    # Training
    "GradientSurgeon",
    "AdaptiveGradientSurgeon",
    "GradientConflictAnalyzer",

    # Advanced Schedulers
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialDecayLR",
    "AdaptiveLRScheduler",
    "NoisyStudentScheduler",
    "SchedulerFactory",

    # Progressive Training
    "ProgressiveTrainingConfig",
    "CurriculumLearning",
    "GrowLengthScheduler",
    "DynamicBatchSizer",
    "ProgressiveModelScaler",
    "ProgressiveTrainer",

    # Losses
    "ContrastiveLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "DiversityLoss",
    "AuxiliaryLoss",
    "ConsistencyLoss",
    "PerplexityLoss",
    "AdaptiveLossScaling",
    "CompositeLoss",

    # Evaluation
    "ComprehensiveEvaluator",
    "PerplexityEvaluator",
    "BLEUEvaluator",
    "ROUGEEvaluator",
    "ToxicityEvaluator",
    "BiasEvaluator",
    "CoherenceEvaluator",

    # Optimization
    "ModelQuantizer",
    "LinearQuantized",
    "DynamicQuantization",
    "INT4Quantization",
    "QuantizationConfig",
    "quantize_model_pipeline",

    # Advanced Optimizers
    "LionOptimizer",
    "SophiaOptimizer",
    "AdaFactorOptimizer",
    "OptimizerFactory",

    # FP8 Training
    "FP8Format",
    "FP8Config",
    "FP8Handler",
    "FP8Linear",
    "FP8MultiHeadAttention",
    "FP8LayerNorm",
    "FP8TransformerLayer",
    "FP8ModelWrapper",
    "create_fp8_model",
    "benchmark_fp8_training",

    # Memory
    "EpisodicMemoryBank",
    "MemoryEntry",
    "MemoryRetriever",
    "AdaptiveMemoryManager",
    "ExperienceReplay",

    # Serving
    "LLMServer",
]