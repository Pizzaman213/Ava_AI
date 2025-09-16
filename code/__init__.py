"""MoE++ Language Learning Model Package"""

try:
    from LLM.src.model.moe_transformer import MoEModel, MoEForCausalLM
    from LLM.src.generation.text_generator import TextGenerator
    # Use MoEForCausalLM as the default model for text generation
    MoEModel = MoEForCausalLM
    __all__ = ["MoEModel", "MoEForCausalLM", "TextGenerator"]
except ImportError:
    # Handle cases where LLM module is not available
    __all__ = []