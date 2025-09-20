"""
Model quantization for efficient inference.

This module implements various quantization techniques including INT8, INT4,
and other optimization methods for reducing model size and inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class QuantizationConfig:
    """Configuration for quantization settings."""
    bit_width: int = 8
    symmetric: bool = True
    per_channel: bool = True
    reduce_range: bool = False
    observer_type: str = "histogram"
    calibration_steps: int = 100


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

        self.quantized_weight.copy_(quantized)

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
            if self.bin_edges is None:
                # Initialize histogram bins
                range_val = max(abs(self.min_val), abs(self.max_val))
                self.bin_edges = torch.linspace(-range_val, range_val, self.num_bins + 1)

            # Update histogram
            hist = torch.histc(tensor, bins=self.num_bins, min=-range_val, max=range_val)
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
        if self.histogram is None:
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
        config: QuantizationConfig = None,
        skip_layers: List[str] = None
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
                # Replace with quantized linear layer
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
                if module.bias is not None:
                    quantized_layer.quantized_bias.copy_(module.bias.data.round().int())

                setattr(quantized_model, name, quantized_layer)

            elif len(list(module.children())) > 0:
                # Recursively process child modules
                child_module = getattr(quantized_model, name, None)
                if child_module is None:
                    # Create child module if it doesn't exist
                    child_module = type(module)()
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
    config: QuantizationConfig = None,
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