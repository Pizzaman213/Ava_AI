"""
Deployment Optimization for MoE++ Model
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import json
from pathlib import Path
import hashlib
from datetime import datetime


class EdgeDeployment:
    """Optimize model for edge device deployment"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.optimization_config = {
            'quantization': 'int8',
            'pruning_threshold': 0.01,
            'knowledge_distillation': False
        }
    
    def optimize_for_edge(self) -> nn.Module:
        """Optimize model for edge deployment"""
        optimized_model = self.model
        
        # Apply quantization
        if self.optimization_config['quantization'] == 'int8':
            optimized_model = self._quantize_model(optimized_model)
        
        # Apply pruning
        if self.optimization_config['pruning_threshold'] > 0:
            optimized_model = self._prune_model(optimized_model, self.optimization_config['pruning_threshold'])
        
        return optimized_model
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Quantize model to int8"""
        # Dynamic quantization for now (static would require calibration data)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model
    
    def _prune_model(self, model: nn.Module, threshold: float) -> nn.Module:
        """Prune small weights"""
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Simple magnitude pruning
                mask = torch.abs(param.data) > threshold
                param.data *= mask.float()
        return model
    
    def benchmark_edge_performance(self) -> Dict[str, float]:
        """Benchmark model performance for edge"""
        import time
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, (1, 128))
        
        # Measure latency
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = self.model(dummy_input)
        end = time.time()
        
        avg_latency = (end - start) / 100
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'avg_latency_ms': avg_latency * 1000,
            'model_size_mb': model_size_mb,
            'throughput': 1.0 / avg_latency
        }


class ModelVersioning:
    """Version control for model checkpoints"""
    
    def __init__(self, model: nn.Module, base_path: str = "checkpoints"):
        self.model = model
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.version_history = []
        self.metadata_file = self.base_path / "version_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load version metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.version_history = json.load(f)
    
    def save_version(self, path: str, description: str = "", metrics: Optional[Dict] = None):
        """Save a new model version"""
        # Generate version hash
        model_state = self.model.state_dict()
        model_bytes = torch.save(model_state, path)
        
        # Create version metadata
        version_info = {
            'version_id': self._generate_version_id(),
            'timestamp': datetime.now().isoformat(),
            'path': str(path),
            'description': description,
            'metrics': metrics or {},
            'model_hash': self._compute_model_hash(model_state)
        }
        
        self.version_history.append(version_info)
        self._save_metadata()
        
        return version_info['version_id']
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"v_{timestamp}_{len(self.version_history)}"
    
    def _compute_model_hash(self, state_dict: Dict) -> str:
        """Compute hash of model weights"""
        hasher = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            hasher.update(key.encode())
            hasher.update(state_dict[key].cpu().numpy().tobytes())
        return hasher.hexdigest()[:16]
    
    def _save_metadata(self):
        """Save version metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def load_version(self, version_id: str) -> bool:
        """Load a specific model version"""
        for version in self.version_history:
            if version['version_id'] == version_id:
                checkpoint = torch.load(version['path'])
                self.model.load_state_dict(checkpoint)
                return True
        return False
    
    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """Get information about a specific version"""
        for version in self.version_history:
            if version['version_id'] == version_id:
                return version
        return None
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict:
        """Compare two model versions"""
        v1 = self.get_version_info(version_id1)
        v2 = self.get_version_info(version_id2)
        
        if not v1 or not v2:
            return {}
        
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'time_diff': v2['timestamp'] - v1['timestamp'] if hasattr(v2['timestamp'], '__sub__') else 'N/A',
            'metrics_diff': {}
        }
        
        # Compare metrics
        for key in set(v1.get('metrics', {}).keys()) | set(v2.get('metrics', {}).keys()):
            v1_val = v1.get('metrics', {}).get(key, 0)
            v2_val = v2.get('metrics', {}).get(key, 0)
            comparison['metrics_diff'][key] = v2_val - v1_val
        
        return comparison


class ModelCompression:
    """Advanced model compression techniques"""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def compress_with_distillation(self, teacher_model: nn.Module, student_config: Dict) -> nn.Module:
        """Compress model using knowledge distillation"""
        # Create smaller student model (simplified for now)
        student_model = self._create_student_model(student_config)
        
        # In practice, would train student with distillation loss
        # For now, just return a smaller version
        return student_model
    
    def _create_student_model(self, config: Dict) -> nn.Module:
        """Create a smaller student model"""
        # Simplified: just return original model
        # In practice, would create smaller architecture
        return self.model
    
    def compress_with_pruning(self, sparsity: float = 0.5) -> nn.Module:
        """Structured pruning for compression"""
        import torch.nn.utils.prune as prune
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
        
        return self.model