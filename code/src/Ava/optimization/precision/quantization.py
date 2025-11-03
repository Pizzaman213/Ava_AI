"""
Model quantization for efficient inference.

This module implements various quantization techniques including INT8, INT4,
and other optimization methods for reducing model size and inference time.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import math

# TorchAO integration for hardware-accelerated NVFP4
TORCHAO_AVAILABLE = False
try:
    from torchao.quantization import quantize_  # type: ignore[import]
    from torchao.prototype.mx_formats import NVFP4InferenceConfig  # type: ignore[import]
    from torchao.quantization.qat import QATConfig  # type: ignore[import]
    TORCHAO_AVAILABLE = True
    print(" TorchAO available - Hardware-accelerated NVFP4 enabled")
except (ImportError, AttributeError) as e:
    # AttributeError can occur with version mismatches (e.g., torch.int1 not available)
    quantize_ = None  # type: ignore[assignment]
    NVFP4InferenceConfig = None  # type: ignore[assignment,misc]
    QATConfig = None  # type: ignore[assignment,misc]
    print(f" TorchAO not available ({e.__class__.__name__}) - using custom NVFP4 implementation")


@dataclass
class QuantizationConfig:
    """Configuration for quantization settings."""
    bit_width: int = 8
    symmetric: bool = True
    per_channel: bool = True
    reduce_range: bool = False
    observer_type: str = "histogram"
    calibration_steps: int = 100
    # NVFP4 specific settings
    use_nvfp4: bool = False
    nvfp4_block_size: int = 16
    stochastic_rounding: bool = False
    use_hadamard_transform: bool = False


class LinearQuantized(nn.Module):
    """
    Quantized linear layer supporting INT8 and INT4 quantization.

    This layer performs quantized matrix multiplication with support for
    different bit widths and quantization schemes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bit_width: int = 8,
        symmetric: bool = True,
        per_channel: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.per_channel = per_channel

        # Quantization parameters
        self.register_buffer('weight_scale', torch.ones(out_features if per_channel else 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features if per_channel else 1, dtype=torch.int32))
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0, dtype=torch.int32))

        # Quantized weights
        if bit_width == 8:
            self.register_buffer('quantized_weight', torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))
        elif bit_width == 4:
            self.register_buffer('quantized_weight', torch.randint(-8, 7, (out_features, in_features), dtype=torch.int8))
        else:
            raise ValueError(f"Unsupported bit width: {bit_width}")

        # Bias
        if bias:
            self.register_buffer('quantized_bias', torch.zeros(out_features, dtype=torch.int32))
        else:
            self.quantized_bias = None

        # Calibration flag
        self.calibrated = False

    def quantize_weight(self, weight: torch.Tensor):
        """Quantize the weight tensor."""
        if self.per_channel:
            # Per-channel quantization (along output channels)
            weight_reshaped = weight.view(self.out_features, -1)

            if self.symmetric:
                # Symmetric quantization
                weight_max = torch.max(torch.abs(weight_reshaped), dim=1)[0]
                self.weight_scale = weight_max / (2**(self.bit_width-1) - 1)
                self.weight_zero_point.zero_()
            else:
                # Asymmetric quantization
                weight_min = torch.min(weight_reshaped, dim=1)[0]
                weight_max = torch.max(weight_reshaped, dim=1)[0]

                qmin = -2**(self.bit_width-1)
                qmax = 2**(self.bit_width-1) - 1

                self.weight_scale = (weight_max - weight_min) / (qmax - qmin)
                self.weight_zero_point = qmin - torch.round(weight_min / self.weight_scale).int()

            # Quantize weights
            scale_expanded = self.weight_scale.unsqueeze(1)
            zero_point_expanded = self.weight_zero_point.unsqueeze(1)

            quantized = torch.round(weight / scale_expanded + zero_point_expanded)

            if self.bit_width == 8:
                quantized = torch.clamp(quantized, -128, 127).to(torch.int8)
            elif self.bit_width == 4:
                quantized = torch.clamp(quantized, -8, 7).to(torch.int8)

        else:
            # Per-tensor quantization
            if self.symmetric:
                weight_max = torch.max(torch.abs(weight))
                self.weight_scale.fill_(weight_max / (2**(self.bit_width-1) - 1))
                self.weight_zero_point.zero_()
            else:
                weight_min = torch.min(weight)
                weight_max = torch.max(weight)

                qmin = -2**(self.bit_width-1)
                qmax = 2**(self.bit_width-1) - 1

                self.weight_scale.fill_((weight_max - weight_min) / (qmax - qmin))
                self.weight_zero_point.fill_(qmin - round((weight_min / self.weight_scale.item()).item()))

            quantized = torch.round(weight / self.weight_scale + self.weight_zero_point)

            if self.bit_width == 8:
                quantized = torch.clamp(quantized, -128, 127).to(torch.int8)
            elif self.bit_width == 4:
                quantized = torch.clamp(quantized, -8, 7).to(torch.int8)

        self.quantized_weight.copy_(quantized)  # type: ignore[misc]

    def dequantize_weight(self) -> torch.Tensor:
        """Dequantize the weight tensor."""
        if self.per_channel:
            scale_expanded = self.weight_scale.unsqueeze(1)
            zero_point_expanded = self.weight_zero_point.unsqueeze(1).float()
        else:
            scale_expanded = self.weight_scale
            zero_point_expanded = self.weight_zero_point.float()

        return (self.quantized_weight.float() - zero_point_expanded) * scale_expanded

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input activations."""
        if self.symmetric:
            x_max = torch.max(torch.abs(x))
            self.input_scale = x_max / (2**(self.bit_width-1) - 1)
            self.input_zero_point.zero_()
        else:
            x_min = torch.min(x)
            x_max = torch.max(x)

            qmin = -2**(self.bit_width-1)
            qmax = 2**(self.bit_width-1) - 1

            self.input_scale = (x_max - x_min) / (qmax - qmin)
            self.input_zero_point = qmin - torch.round(x_min / self.input_scale).int()

        quantized = torch.round(x / self.input_scale + self.input_zero_point)

        if self.bit_width == 8:
            return torch.clamp(quantized, -128, 127).to(torch.int8)
        elif self.bit_width == 4:
            return torch.clamp(quantized, -8, 7).to(torch.int8)
        else:
            # Default to int8 for unsupported bit widths
            return torch.clamp(quantized, -128, 127).to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized computation."""
        if not self.calibrated:
            # If not calibrated, use fake quantization
            return F.linear(x, self.dequantize_weight(),
                          self.quantized_bias.float() if self.quantized_bias is not None else None)

        # Quantize input
        x_quantized = self.quantize_input(x)

        # Perform quantized matrix multiplication
        # This is a simplified version - real implementation would use optimized kernels
        weight_dequantized = self.dequantize_weight()
        x_dequantized = (x_quantized.float() - self.input_zero_point.float()) * self.input_scale

        output = F.linear(x_dequantized, weight_dequantized)

        # Add bias if present
        if self.quantized_bias is not None:
            bias_dequantized = self.quantized_bias.float() * self.weight_scale * self.input_scale
            if self.per_channel:
                output += bias_dequantized
            else:
                output += bias_dequantized.item()

        return output


class LinearNVFP4(nn.Module):
    """
    NVFP4 quantized linear layer for training and inference.

    Supports weight and activation quantization using NVFP4 format
    with micro-block scaling and stochastic rounding.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 16,
        stochastic_rounding: bool = False,
        use_hadamard_transform: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.stochastic_rounding = stochastic_rounding
        self.use_hadamard_transform = use_hadamard_transform

        # NVFP4 quantizer
        self.nvfp4_quantizer = NVFP4Quantization(
            block_size=block_size,
            stochastic_rounding=stochastic_rounding
        )

        # Weight parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Quantized weight storage
        self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=torch.uint8))
        self.register_buffer('weight_scales', torch.zeros(1, dtype=torch.uint8))

        # Training/inference mode
        self.training_mode = True
        self.calibrated = False

    def quantize_weights(self):
        """Quantize weights to NVFP4 format."""
        with torch.no_grad():
            quantized, scales = self.nvfp4_quantizer.quantize_tensor_nvfp4(self.weight)
            self.quantized_weight.copy_(quantized.to(torch.uint8))  # type: ignore[misc]
            self.weight_scales = scales
            self.calibrated = True

    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to float."""
        if not self.calibrated:
            return self.weight

        return self.nvfp4_quantizer.dequantize_tensor_nvfp4(
            self.quantized_weight,  # type: ignore[arg-type]
            self.weight_scales,
            self.weight.shape
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NVFP4 quantization."""
        # Apply Hadamard transform if enabled
        if self.use_hadamard_transform:
            x = self.nvfp4_quantizer.hadamard_transform(x)

        if self.training and self.training_mode:
            # Training mode: use fake quantization
            # Quantize and immediately dequantize for gradient flow
            quantized_weight, _ = self.nvfp4_quantizer.quantize_tensor_nvfp4(self.weight)
            dequantized_weight = self.nvfp4_quantizer.dequantize_tensor_nvfp4(
                quantized_weight,
                torch.zeros(1, dtype=torch.uint8),  # Dummy scales for fake quantization
                self.weight.shape
            )

            # Also quantize activations during training
            if x.requires_grad:
                quantized_x, _ = self.nvfp4_quantizer.quantize_tensor_nvfp4(x)
                x = self.nvfp4_quantizer.dequantize_tensor_nvfp4(
                    quantized_x,
                    torch.zeros(1, dtype=torch.uint8),
                    x.shape
                )

            output = F.linear(x, dequantized_weight, self.bias)
        else:
            # Inference mode: use stored quantized weights
            if self.calibrated:
                dequantized_weight = self.dequantize_weights()
            else:
                dequantized_weight = self.weight

            output = F.linear(x, dequantized_weight, self.bias)

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, block_size={self.block_size}, ' \
               f'stochastic_rounding={self.stochastic_rounding}'


class QuantizationObserver:
    """
    Observer for collecting statistics during calibration.

    This class tracks activation statistics to determine optimal
    quantization parameters.
    """

    def __init__(
        self,
        bit_width: int = 8,
        symmetric: bool = True,
        observer_type: str = "histogram"
    ):
        self.bit_width = bit_width
        self.symmetric = symmetric
        self.observer_type = observer_type

        # Statistics
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.histogram = None
        self.bin_edges = None
        self.total_samples = 0

        if observer_type == "histogram":
            self.num_bins = 2048
            self.histogram = torch.zeros(self.num_bins)

    def update(self, tensor: torch.Tensor):
        """Update statistics with new tensor."""
        self.min_val = min(self.min_val, tensor.min().item())
        self.max_val = max(self.max_val, tensor.max().item())
        self.total_samples += tensor.numel()

        if self.observer_type == "histogram":
            range_val = max(abs(self.min_val), abs(self.max_val))
            if self.bin_edges is None:
                # Initialize histogram bins
                self.bin_edges = torch.linspace(-range_val, range_val, self.num_bins + 1)

            # Update histogram
            hist = torch.histc(tensor, bins=self.num_bins, min=-range_val, max=range_val)
            if self.histogram is None:
                self.histogram = hist
            else:
                self.histogram += hist

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate quantization parameters based on collected statistics."""
        if self.observer_type == "minmax":
            return self._calculate_minmax_qparams()
        elif self.observer_type == "histogram":
            return self._calculate_histogram_qparams()
        else:
            raise ValueError(f"Unknown observer type: {self.observer_type}")

    def _calculate_minmax_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate qparams using min-max method."""
        if self.symmetric:
            max_val = max(abs(self.min_val), abs(self.max_val))
            scale = max_val / (2**(self.bit_width-1) - 1)
            zero_point = torch.tensor(0, dtype=torch.int32)
        else:
            qmin = -2**(self.bit_width-1)
            qmax = 2**(self.bit_width-1) - 1
            scale = (self.max_val - self.min_val) / (qmax - qmin)
            zero_point = qmin - round(self.min_val / scale)
            zero_point = torch.tensor(zero_point, dtype=torch.int32)

        return torch.tensor(scale), zero_point

    def _calculate_histogram_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate qparams using histogram method with outlier removal."""
        if self.histogram is None or self.bin_edges is None:
            return self._calculate_minmax_qparams()

        # Remove outliers (e.g., top and bottom 0.01%)
        total_count = self.histogram.sum()
        outlier_ratio = 0.0001
        outlier_count = total_count * outlier_ratio

        cumsum = torch.cumsum(self.histogram, dim=0)

        # Find lower bound
        lower_idx = torch.searchsorted(cumsum, outlier_count)
        lower_bound = self.bin_edges[lower_idx] if lower_idx < len(self.bin_edges) else self.bin_edges[0]

        # Find upper bound
        upper_idx = torch.searchsorted(cumsum, total_count - outlier_count)
        upper_bound = self.bin_edges[upper_idx] if upper_idx < len(self.bin_edges) else self.bin_edges[-1]

        if self.symmetric:
            max_val = max(abs(lower_bound), abs(upper_bound))
            scale = max_val / (2**(self.bit_width-1) - 1)
            zero_point = torch.tensor(0, dtype=torch.int32)
        else:
            qmin = -2**(self.bit_width-1)
            qmax = 2**(self.bit_width-1) - 1
            scale = (upper_bound - lower_bound) / (qmax - qmin)
            zero_point = qmin - round(lower_bound / scale)
            zero_point = torch.tensor(zero_point, dtype=torch.int32)

        return torch.tensor(scale.item()), zero_point


class ModelQuantizer:
    """
    Model quantizer for converting PyTorch models to quantized versions.

    This class handles the quantization of entire models including
    calibration and conversion processes.
    """

    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
        skip_layers: Optional[List[str]] = None
    ):
        self.config = config or QuantizationConfig()
        self.skip_layers = skip_layers or ['lm_head', 'embed_tokens']
        self.observers = {}
        self.calibrated = False

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for quantization by replacing layers with quantizable versions.

        Args:
            model: Model to prepare for quantization

        Returns:
            Model with quantizable layers
        """
        # Clone the model to avoid modifying the original
        quantized_model = self._clone_model_structure(model)

        # Replace linear layers with quantized versions
        self._replace_layers_recursive(quantized_model, "", model)

        return quantized_model

    def _clone_model_structure(self, model: nn.Module) -> nn.Module:
        """Clone model structure without copying weights."""
        # This is a simplified approach - in practice, you'd need more sophisticated cloning
        return type(model)(**model.__dict__.get('config', {}).__dict__ if hasattr(model, 'config') else {})

    def _replace_layers_recursive(self, quantized_model: nn.Module, prefix: str, original_model: nn.Module):
        """Recursively replace layers with quantized versions."""
        for name, module in original_model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, nn.Linear) and not any(skip in full_name for skip in self.skip_layers):
                # Choose quantization type based on config
                if self.config.use_nvfp4:
                    # Replace with NVFP4 quantized linear layer
                    quantized_layer = LinearNVFP4(
                        module.in_features,
                        module.out_features,
                        bias=(module.bias is not None),
                        block_size=self.config.nvfp4_block_size,
                        stochastic_rounding=self.config.stochastic_rounding,
                        use_hadamard_transform=self.config.use_hadamard_transform
                    )

                    # Copy original weights
                    quantized_layer.weight.data.copy_(module.weight.data)
                    if module.bias is not None:
                        quantized_layer.bias.data.copy_(module.bias.data)

                else:
                    # Replace with traditional quantized linear layer
                    quantized_layer = LinearQuantized(
                        module.in_features,
                        module.out_features,
                        bias=(module.bias is not None),
                        bit_width=self.config.bit_width,
                        symmetric=self.config.symmetric,
                        per_channel=self.config.per_channel
                    )

                    # Copy original weights for calibration
                    quantized_layer.quantize_weight(module.weight.data)
                    if module.bias is not None and quantized_layer.quantized_bias is not None:
                        quantized_layer.quantized_bias.copy_(module.bias.data.round().int())

                setattr(quantized_model, name, quantized_layer)

            elif len(list(module.children())) > 0:
                # Recursively process child modules
                child_module = getattr(quantized_model, name, None)
                if child_module is None:
                    # Create child module if it doesn't exist
                    # We need to properly initialize the module, so we use type() with no args
                    # This may fail for modules that require constructor arguments
                    try:
                        child_module = type(module)()  # type: ignore[call-arg]
                    except TypeError:
                        # Skip modules that can't be instantiated without args
                        continue
                    setattr(quantized_model, name, child_module)

                self._replace_layers_recursive(child_module, full_name, module)
            else:
                # Copy non-linear layers as-is
                setattr(quantized_model, name, module)

    def calibrate(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ):
        """
        Calibrate quantization parameters using calibration data.

        Args:
            model: Model to calibrate
            calibration_loader: DataLoader with calibration data
            device: Device to run calibration on
        """
        model.eval()
        model.to(device)

        # Install observers
        self._install_observers(model)

        print(f"Starting calibration with {self.config.calibration_steps} steps...")

        with torch.no_grad():
            for step, batch in enumerate(calibration_loader):
                if step >= self.config.calibration_steps:
                    break

                if isinstance(batch, dict):
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    model(**inputs)
                else:
                    inputs = batch.to(device) if torch.is_tensor(batch) else batch
                    model(inputs)

                if (step + 1) % 10 == 0:
                    print(f"Calibration step {step + 1}/{self.config.calibration_steps}")

        # Calculate quantization parameters
        self._calculate_qparams(model)
        self.calibrated = True

        print("Calibration completed!")

    def _install_observers(self, model: nn.Module, prefix: str = ""):
        """Install observers for activation tracking."""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, LinearQuantized):
                # Install forward hook to observe activations
                observer = QuantizationObserver(
                    bit_width=self.config.bit_width,
                    symmetric=self.config.symmetric,
                    observer_type=self.config.observer_type
                )
                self.observers[full_name] = observer

                def make_hook(obs):
                    def hook(module, input, output):
                        if len(input) > 0 and torch.is_tensor(input[0]):
                            obs.update(input[0])
                    return hook

                module.register_forward_hook(make_hook(observer))

            elif len(list(module.children())) > 0:
                self._install_observers(module, full_name)

    def _calculate_qparams(self, model: nn.Module, prefix: str = ""):
        """Calculate and apply quantization parameters."""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(module, LinearQuantized):
                if full_name in self.observers:
                    observer = self.observers[full_name]
                    scale, zero_point = observer.calculate_qparams()

                    # Update quantization parameters
                    module.input_scale.copy_(scale)
                    module.input_zero_point.copy_(zero_point)
                    module.calibrated = True

            elif len(list(module.children())) > 0:
                self._calculate_qparams(module, full_name)

    def convert(self, model: nn.Module) -> nn.Module:
        """
        Convert calibrated model to final quantized format.

        Args:
            model: Calibrated model

        Returns:
            Fully quantized model
        """
        if not self.calibrated:
            raise RuntimeError("Model must be calibrated before conversion")

        # Mark all quantized layers as calibrated
        self._mark_calibrated(model)

        return model

    def _mark_calibrated(self, model: nn.Module):
        """Mark all quantized layers as calibrated."""
        for module in model.modules():
            if isinstance(module, LinearQuantized):
                module.calibrated = True

    def save_quantized_model(self, model: nn.Module, path: str):
        """Save quantized model to disk."""
        torch.save({
            'model_state_dict': model.state_dict(),
            'quantization_config': self.config,
            'calibrated': self.calibrated
        }, path)

    def load_quantized_model(self, model: nn.Module, path: str) -> nn.Module:
        """Load quantized model from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint.get('quantization_config', self.config)
        self.calibrated = checkpoint.get('calibrated', False)
        return model


class DynamicQuantization:
    """
    Dynamic quantization for runtime weight quantization.

    This approach quantizes weights but keeps activations in full precision,
    providing a good balance between performance and accuracy.
    """

    def __init__(self, bit_width: int = 8):
        self.bit_width = bit_width

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Args:
            model: Model to quantize

        Returns:
            Dynamically quantized model
        """
        # Use PyTorch's built-in dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8 if self.bit_width == 8 else torch.quint8
        )

        return quantized_model


class NVFP4Quantization:
    """
    NVIDIA NVFP4 4-bit floating-point quantization for training and inference.

    Implements NVFP4 format with micro-block scaling (16-element blocks),
    E4M3 scale factors, and stochastic rounding for training.
    """

    def __init__(self, block_size: int = 16, stochastic_rounding: bool = False):
        self.block_size = block_size
        self.stochastic_rounding = stochastic_rounding

        # NVFP4 format: E2M1 (2 exponent bits, 1 mantissa bit)
        self.exponent_bits = 2
        self.mantissa_bits = 1
        self.bias = 1  # 2^(exp_bits-1) - 1

        # E4M3 scale format for higher precision scaling
        self.scale_exp_bits = 4
        self.scale_mantissa_bits = 3
        self.scale_bias = 7  # 2^(4-1) - 1

    def hadamard_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard transform to reshape tensor distributions."""
        # Simplified 2x2 Hadamard transform for demonstration
        # In practice, use larger matrices for better distribution reshaping
        original_shape = x.shape
        if x.numel() % 2 != 0:
            # Pad for even number of elements
            x = F.pad(x.flatten(), (0, 1))

        x_reshaped = x.view(-1, 2)
        # 2x2 Hadamard matrix: [[1, 1], [1, -1]] / sqrt(2)
        h_matrix = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=x.device) / math.sqrt(2)
        transformed = torch.matmul(x_reshaped, h_matrix.T)

        # Reshape back and trim padding if needed
        result = transformed.flatten()[:torch.numel(torch.zeros(original_shape))]
        return result.view(original_shape)

    def encode_fp4_e2m1(self, value: float) -> int:
        """Encode a float value to NVFP4 E2M1 format (4 bits)."""
        if value == 0.0:
            return 0

        # Handle special cases
        if math.isnan(value) or math.isinf(value):
            return 0  # Map to zero for stability

        sign = 0 if value >= 0 else 1
        abs_value = abs(value)

        # Find the best exponent and mantissa representation
        best_encoded = 0
        best_error = float('inf')

        # Try all possible 4-bit encodings (16 values)
        for encoded in range(16):
            decoded = self.decode_fp4_e2m1(encoded)
            error = abs(abs_value - abs(decoded))
            if error < best_error:
                best_error = error
                best_encoded = encoded

        return best_encoded

    def decode_fp4_e2m1(self, encoded: int) -> float:
        """Decode NVFP4 E2M1 4-bit value to float."""
        if encoded == 0:
            return 0.0

        # Extract bits: SEEE (sign + 3 value bits for E2M1)
        sign = (encoded >> 3) & 1
        exp_mant = encoded & 0x7  # 3 bits for exponent and mantissa

        if exp_mant == 0:
            return 0.0

        # Simple mapping for E2M1 format
        # Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        value_map = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        value = value_map[min(exp_mant, len(value_map) - 1)]

        return -value if sign else value

    def encode_e4m3_scale(self, scale: float) -> int:
        """Encode scale factor in E4M3 format (8 bits)."""
        if scale <= 0:
            return 0

        # Find closest E4M3 representation
        log_scale = math.log2(scale)
        exp = max(0, min(15, int(log_scale) + self.scale_bias))  # 4-bit exponent

        # Extract mantissa
        mantissa_val = scale / (2.0 ** (exp - self.scale_bias))
        mantissa = int((mantissa_val - 1.0) * 8) if mantissa_val >= 1.0 else 0
        mantissa = max(0, min(7, mantissa))  # 3-bit mantissa

        return (exp << 3) | mantissa

    def decode_e4m3_scale(self, encoded: int) -> float:
        """Decode E4M3 scale factor to float."""
        if encoded == 0:
            return 0.0

        exp = (encoded >> 3) & 0xF  # 4 bits
        mantissa = encoded & 0x7   # 3 bits

        # Decode E4M3: value = (1 + mantissa/8) * 2^(exp - bias)
        mantissa_val = 1.0 + mantissa / 8.0
        scale = mantissa_val * (2.0 ** (exp - self.scale_bias))

        return scale

    def stochastic_round(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic rounding to reduce bias."""
        if not self.stochastic_rounding:
            return torch.round(x)

        # Stochastic rounding: probability proportional to fractional part
        floor_x = torch.floor(x)
        frac_x = x - floor_x

        # Generate random values
        random_vals = torch.rand_like(frac_x)

        # Round up where random value < fractional part
        return floor_x + (random_vals < frac_x).float()

    def quantize_tensor_nvfp4(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NVFP4 format with micro-block scaling.

        Returns:
            Tuple of (quantized_values, scale_factors)
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Pad to multiple of block_size
        pad_size = (self.block_size - (tensor_flat.numel() % self.block_size)) % self.block_size
        if pad_size > 0:
            tensor_flat = F.pad(tensor_flat, (0, pad_size))

        # Reshape to blocks
        tensor_blocks = tensor_flat.view(-1, self.block_size)
        num_blocks = tensor_blocks.size(0)

        # Compute scale factors per block (E4M3 precision)
        block_max = torch.max(torch.abs(tensor_blocks), dim=1)[0]
        # Scale to utilize full NVFP4 range (max value ~6.0)
        scale_factors = block_max / 6.0
        scale_factors = torch.clamp(scale_factors, min=1e-7)  # Avoid zeros

        # Quantize each block
        quantized_blocks = torch.zeros_like(tensor_blocks, dtype=torch.uint8)

        for i in range(num_blocks):
            block = tensor_blocks[i]
            scale = scale_factors[i]

            # Normalize by scale factor
            normalized_block = block / scale

            # Apply stochastic rounding if enabled
            normalized_block = self.stochastic_round(normalized_block)

            # Encode to 4-bit NVFP4
            for j in range(self.block_size):
                val = normalized_block[j].item()
                encoded = self.encode_fp4_e2m1(val)
                quantized_blocks[i, j] = encoded

        # Encode scale factors to E4M3
        encoded_scales = torch.zeros(num_blocks, dtype=torch.uint8)
        for i in range(num_blocks):
            encoded_scales[i] = self.encode_e4m3_scale(scale_factors[i].item())

        # Trim back to original size
        quantized_flat = quantized_blocks.flatten()[:tensor.numel()]

        return quantized_flat.view(original_shape), encoded_scales

    def dequantize_tensor_nvfp4(
        self,
        quantized: torch.Tensor,
        encoded_scales: torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:
        """Dequantize NVFP4 tensor back to float."""
        quantized_flat = quantized.flatten()

        # Pad to multiple of block_size
        pad_size = (self.block_size - (quantized_flat.numel() % self.block_size)) % self.block_size
        if pad_size > 0:
            quantized_flat = F.pad(quantized_flat, (0, pad_size))

        quantized_blocks = quantized_flat.view(-1, self.block_size)
        num_blocks = quantized_blocks.size(0)

        # Decode scale factors from E4M3
        scale_factors = torch.zeros(num_blocks, device=quantized.device)
        for i in range(len(encoded_scales)):
            scale_factors[i] = self.decode_e4m3_scale(int(encoded_scales[i].item()))

        # Dequantize each block
        dequantized_blocks = torch.zeros_like(quantized_blocks, dtype=torch.float32)

        for i in range(num_blocks):
            scale = scale_factors[i]

            for j in range(self.block_size):
                encoded_val = int(quantized_blocks[i, j].item())
                decoded_val = self.decode_fp4_e2m1(encoded_val)
                dequantized_blocks[i, j] = decoded_val * scale

        # Reshape and trim back to original size
        dequantized_flat = dequantized_blocks.flatten()[:torch.numel(torch.zeros(original_shape))]
        return dequantized_flat.view(original_shape)


class TorchAONVFP4Wrapper:
    """
    Wrapper for TorchAO NVFP4 implementation.

    Provides hardware-accelerated NVFP4 training and inference
    when TorchAO is available.
    """

    def __init__(self, use_qat: bool = True):
        if not TORCHAO_AVAILABLE:
            raise ImportError("TorchAO is not available. Install with: pip install torchao")
        if NVFP4InferenceConfig is None or QATConfig is None:
            raise ImportError("TorchAO components not available")

        self.use_qat = use_qat
        self.base_config = NVFP4InferenceConfig()  # type: ignore[misc]

        if use_qat:
            self.qat_config_prepare = QATConfig(self.base_config, step="prepare")  # type: ignore[misc]
            self.qat_config_convert = QATConfig(self.base_config, step="convert")  # type: ignore[misc]

    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare model for NVFP4 quantization-aware training."""
        if not self.use_qat:
            raise ValueError("QAT not enabled for this wrapper")
        if quantize_ is None:
            raise ImportError("quantize_ not available")

        print(" Preparing model for NVFP4 training with TorchAO...")
        quantized_model = quantize_(model, self.qat_config_prepare)  # type: ignore[misc]
        print(" Model prepared for NVFP4 training")

        return quantized_model  # type: ignore[return-value]

    def convert_model_after_training(self, model: nn.Module) -> nn.Module:
        """Convert model to final NVFP4 format after training."""
        if not self.use_qat:
            raise ValueError("QAT not enabled for this wrapper")
        if quantize_ is None:
            raise ImportError("quantize_ not available")

        print(" Converting model to final NVFP4 format...")
        final_model = quantize_(model, self.qat_config_convert)  # type: ignore[misc]
        print(" Model converted to NVFP4 format")

        return final_model  # type: ignore[return-value]

    def quantize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Quantize model directly for NVFP4 inference."""
        if quantize_ is None:
            raise ImportError("quantize_ not available")

        print(" Quantizing model for NVFP4 inference...")
        quantized_model = quantize_(model, self.base_config)  # type: ignore[misc]
        print(" Model quantized for NVFP4 inference")

        return quantized_model  # type: ignore[return-value]


class INT4Quantization:
    """
    Specialized INT4 quantization for ultra-low precision inference.

    This implements aggressive 4-bit quantization for maximum compression.
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size

    def quantize_weights(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize weights to INT4 with group-wise scaling.

        Args:
            weights: Weight tensor to quantize

        Returns:
            Tuple of (quantized_weights, scales, zeros)
        """
        original_shape = weights.shape
        weights_flat = weights.flatten()

        # Group the weights
        num_groups = (weights_flat.numel() + self.group_size - 1) // self.group_size
        padded_size = num_groups * self.group_size

        if weights_flat.numel() < padded_size:
            weights_flat = torch.cat([weights_flat, torch.zeros(padded_size - weights_flat.numel())])

        weights_grouped = weights_flat.view(num_groups, self.group_size)

        # Compute scales and zero points per group
        weight_min = weights_grouped.min(dim=1)[0]
        weight_max = weights_grouped.max(dim=1)[0]

        scales = (weight_max - weight_min) / 15  # 4-bit range: 0-15
        zeros = weight_min

        # Quantize
        quantized = torch.round((weights_grouped - zeros.unsqueeze(1)) / scales.unsqueeze(1))
        quantized = torch.clamp(quantized, 0, 15).to(torch.uint8)

        # Pack 2 4-bit values into each byte
        quantized_packed = torch.zeros(num_groups, self.group_size // 2, dtype=torch.uint8)
        for i in range(0, self.group_size, 2):
            quantized_packed[:, i // 2] = (quantized[:, i] << 4) | quantized[:, i + 1]

        return quantized_packed, scales, zeros

    def dequantize_weights(
        self,
        quantized_weights: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        original_shape: torch.Size
    ) -> torch.Tensor:
        """Dequantize INT4 weights back to float."""
        num_groups, packed_size = quantized_weights.shape

        # Unpack 4-bit values
        dequantized = torch.zeros(num_groups, packed_size * 2)
        for i in range(packed_size):
            dequantized[:, i * 2] = (quantized_weights[:, i] >> 4) & 0xF
            dequantized[:, i * 2 + 1] = quantized_weights[:, i] & 0xF

        # Scale back to float
        dequantized = dequantized * scales.unsqueeze(1) + zeros.unsqueeze(1)

        # Reshape to original
        dequantized_flat = dequantized.flatten()[:np.prod(original_shape)]
        return dequantized_flat.view(original_shape)


def quantize_model_pipeline(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    config: Optional[QuantizationConfig] = None,
    output_path: Optional[str] = None
) -> nn.Module:
    """
    Complete quantization pipeline.

    Args:
        model: Model to quantize
        calibration_loader: Calibration data
        config: Quantization configuration
        output_path: Path to save quantized model

    Returns:
        Quantized model
    """
    config = config or QuantizationConfig()
    quantizer = ModelQuantizer(config)

    # Prepare model
    print("Preparing model for quantization...")
    quantized_model = quantizer.prepare_model(model)

    # Calibrate
    print("Calibrating quantization parameters...")
    quantizer.calibrate(quantized_model, calibration_loader)

    # Convert
    print("Converting to quantized model...")
    final_model = quantizer.convert(quantized_model)

    # Save if path provided
    if output_path:
        print(f"Saving quantized model to {output_path}")
        quantizer.save_quantized_model(final_model, output_path)

    return final_model