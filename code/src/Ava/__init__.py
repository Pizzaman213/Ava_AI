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
try:
    from .models.moe_model import EnhancedMoEModel, EnhancedMoEConfig  # type: ignore[import]
except ImportError:
    EnhancedMoEModel = None
    EnhancedMoEConfig = None

# Adaptive Multi-Token Prediction
try:
    from .models.adaptive_mtp_model import AdaptiveMTPModel, AdaptiveMTPConfig
    from .models.confidence_gate import ConfidenceGate
    from .models.prediction_heads import MultiTokenPredictionHeads
except ImportError:
    AdaptiveMTPModel = AdaptiveMTPConfig = None
    ConfidenceGate = MultiTokenPredictionHeads = None

# Layer components
try:
    from .layers.experts import ExpertBalancer, SparseExpert
except ImportError:
    ExpertBalancer = SparseExpert = None

try:
    from .layers.routing import (
        ExpertSelector, MoEPlusPlusLayer,
        SwitchTransformerRouting, GSERouting,
        HashingExpertRouting, StochasticExpertRouting
    )
except ImportError:
    ExpertSelector = MoEPlusPlusLayer = None
    SwitchTransformerRouting = GSERouting = None
    HashingExpertRouting = StochasticExpertRouting = None

try:
    from .layers.attention import (  # type: ignore[import-not-found]
        EnhancedMultiheadAttention, RotaryPositionEmbedding,
        ALiBiPositionEmbedding, FlashAttention
    )
except ImportError:
    EnhancedMultiheadAttention = RotaryPositionEmbedding = None
    ALiBiPositionEmbedding = FlashAttention = None

try:
    from .layers.mixture_of_heads import (  # type: ignore[import-not-found]
        MixtureOfHeads, AdaptiveHeadAttention
    )
except ImportError:
    MixtureOfHeads = AdaptiveHeadAttention = None

try:
    from .layers.cross_attention import (  # type: ignore[import-not-found]
        MultiModalCrossAttention, PerceiversCrossAttention,
        AdaptiveCrossAttention, HierarchicalCrossAttention
    )
except ImportError:
    MultiModalCrossAttention = PerceiversCrossAttention = None
    AdaptiveCrossAttention = HierarchicalCrossAttention = None

try:
    from .layers.mixture_of_activations import (  # type: ignore[import-not-found]
        MixtureOfActivations, AdaptiveActivation,
        ContextualActivation, HierarchicalActivation
    )
except ImportError:
    MixtureOfActivations = AdaptiveActivation = None
    ContextualActivation = HierarchicalActivation = None

# Retrieval and RAG
try:
    from .retrieval import (  # type: ignore[import-not-found]
        RAGSystem, AdaptiveRAG, DenseRetriever,
        KnowledgeBase, RAGFusion
    )
except ImportError:
    RAGSystem = AdaptiveRAG = DenseRetriever = None
    KnowledgeBase = RAGFusion = None

# Training utilities - Gradient Management
try:
    from .optimization.gradients.surgery import (
        GradientSurgeon, AdaptiveGradientSurgeon,
        GradientConflictAnalyzer
    )
except ImportError:
    GradientSurgeon = AdaptiveGradientSurgeon = None
    GradientConflictAnalyzer = None

# Training utilities - Learning Rate Schedulers
try:
    from .optimization.learning_rate import (
        CosineAnnealingWarmRestarts, OneCycleLR, PolynomialDecayLR,
        AdaptiveLRScheduler, NoisyStudentScheduler, SchedulerFactory
    )
except ImportError:
    CosineAnnealingWarmRestarts = OneCycleLR = PolynomialDecayLR = AdaptiveLRScheduler = None
    NoisyStudentScheduler = SchedulerFactory = None

# Training utilities - Progressive Training
try:
    from .training.strategies.progressive_training import (
        ProgressiveTrainingConfig, CurriculumLearning, GrowLengthScheduler,
        DynamicBatchSizer, ProgressiveModelScaler, ProgressiveTrainer
    )
except ImportError:
    ProgressiveTrainingConfig = CurriculumLearning = GrowLengthScheduler = None
    DynamicBatchSizer = ProgressiveModelScaler = ProgressiveTrainer = None

# Loss functions
try:
    from .losses import (
        ContrastiveLoss, FocalLoss, LabelSmoothingLoss,
        DiversityLoss, AuxiliaryLoss, ConsistencyLoss,
        PerplexityLoss, AdaptiveLossScaling, CompositeLoss
    )
    from .losses.losses import AdaptiveMTPLoss
except ImportError:
    ContrastiveLoss = FocalLoss = LabelSmoothingLoss = None
    DiversityLoss = AuxiliaryLoss = ConsistencyLoss = None
    PerplexityLoss = AdaptiveLossScaling = CompositeLoss = None
    AdaptiveMTPLoss = None

# Evaluation
try:
    from .evaluation.comprehensive_eval import (
        ComprehensiveEvaluator, PerplexityEvaluator,
        BLEUEvaluator, ROUGEEvaluator, ToxicityEvaluator,
        BiasEvaluator, CoherenceEvaluator
    )
except ImportError:
    ComprehensiveEvaluator = PerplexityEvaluator = None
    BLEUEvaluator = ROUGEEvaluator = ToxicityEvaluator = None
    BiasEvaluator = CoherenceEvaluator = None

# Optimization
try:
    from .optimization import (
        ModelQuantizer, LinearQuantized, DynamicQuantization,
        INT4Quantization, QuantizationObserver,
        LionOptimizer, SophiaOptimizer, AdaFactorOptimizer,
        OptimizerFactory,
        FP8Handler, FP8Linear, FP8MultiHeadAttention,
        FP8LayerNorm, FP8TransformerLayer, FP8ModelWrapper,
        GradientHealthMonitor, LossHealthMonitor,
        GradientSurgeon, AdaptiveGradientSurgeon,
        LRFinder, LRFinderConfig,
        AdaptiveLearningRateManager, IntelligentLRManager,
    )
except ImportError:
    ModelQuantizer = LinearQuantized = DynamicQuantization = None
    INT4Quantization = QuantizationObserver = None
    LionOptimizer = SophiaOptimizer = AdaFactorOptimizer = None
    OptimizerFactory = None
    FP8Handler = FP8Linear = FP8MultiHeadAttention = None
    FP8LayerNorm = FP8TransformerLayer = FP8ModelWrapper = None
    GradientHealthMonitor = LossHealthMonitor = None
    GradientSurgeon = AdaptiveGradientSurgeon = None
    LRFinder = LRFinderConfig = None
    AdaptiveLearningRateManager = IntelligentLRManager = None

# Memory and continual learning
try:
    from .memory import (
        EpisodicMemoryBank, MemoryEntry, MemoryRetriever,
        AdaptiveMemoryManager, ExperienceReplay
    )
except ImportError:
    EpisodicMemoryBank = MemoryEntry = MemoryRetriever = None
    AdaptiveMemoryManager = ExperienceReplay = None

# Configuration Management
try:
    from .config import (
        EnhancedTrainingConfig, TrainingConfigManager,
        ArchitectureConfig, RAGConfig, LossConfig, GradientConfig,
        EvaluationConfig, QuantizationConfig, EpisodicMemoryConfig,
        DataConfig, MultiColumnDataConfig, TrainingConfig,
        OutputConfig, RunManagementConfig, WandBConfig, PerformanceConfig
    )
    from .config.training_config import AdaptiveMTPConfig
except ImportError:
    EnhancedTrainingConfig = TrainingConfigManager = None
    ArchitectureConfig = RAGConfig = LossConfig = GradientConfig = None
    EvaluationConfig = QuantizationConfig = EpisodicMemoryConfig = None
    DataConfig = MultiColumnDataConfig = TrainingConfig = None
    OutputConfig = RunManagementConfig = WandBConfig = PerformanceConfig = None
    AdaptiveMTPConfig = None

# Serving
try:
    from .serving.fastapi_server import LLMServer  # type: ignore[import-not-found]
except ImportError:
    LLMServer = None

__version__ = "2.0.0"

__all__ = [
    # Core
    "EnhancedMoEModel",
    "EnhancedMoEConfig",

    # Adaptive Multi-Token Prediction
    "AdaptiveMTPModel",
    "AdaptiveMTPConfig",
    "ConfidenceGate",
    "MultiTokenPredictionHeads",
    "AdaptiveMTPLoss",

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

    # Configuration Management
    "EnhancedTrainingConfig",
    "TrainingConfigManager",
    "ArchitectureConfig",
    "RAGConfig",
    "LossConfig",
    "GradientConfig",
    "EvaluationConfig",
    "QuantizationConfig",
    "EpisodicMemoryConfig",
    "DataConfig",
    "MultiColumnDataConfig",
    "TrainingConfig",
    "OutputConfig",
    "RunManagementConfig",
    "WandBConfig",
    "PerformanceConfig",

    # Serving
    "LLMServer",
]