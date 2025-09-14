"""Quantization modules for model compression"""

from .gptq import GPTQQuantizer, GPTQConfig
from .awq import AWQQuantizer, AWQConfig, quantize_model_awq
from .quantized_linear import QuantizedLinear, DynamicQuantizedLinear