"""
Developer Tools for MoE++ Model
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import json
import time
import numpy as np
from pathlib import Path


class ModelPlayground:
    """Interactive playground for model experimentation"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.experiment_history = []
        self.saved_states = {}
    
    def interactive_inference(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Run inference with interactive parameter adjustment"""
        # Mock tokenization for demo
        input_ids = torch.randint(0, 1000, (1, len(input_text.split())))
        
        result = {
            'input': input_text,
            'parameters': kwargs,
            'timestamp': time.time()
        }
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            if isinstance(outputs, dict):
                result['output'] = {k: v.shape if torch.is_tensor(v) else v 
                                  for k, v in outputs.items()}
            else:
                result['output'] = outputs.shape if torch.is_tensor(outputs) else outputs
        
        self.experiment_history.append(result)
        return result
    
    def save_experiment_state(self, name: str):
        """Save current experiment state"""
        self.saved_states[name] = {
            'model_state': self.model.state_dict(),
            'history': self.experiment_history.copy(),
            'timestamp': time.time()
        }
    
    def load_experiment_state(self, name: str):
        """Load saved experiment state"""
        if name in self.saved_states:
            state = self.saved_states[name]
            self.model.load_state_dict(state['model_state'])
            self.experiment_history = state['history'].copy()
            return True
        return False
    
    def generate_playground(self, output_path: str):
        """Generate HTML playground interface"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MoE++ Model Playground</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .input-area { margin: 20px 0; }
                .output-area { background: #f0f0f0; padding: 10px; border-radius: 5px; }
                button { padding: 10px 20px; margin: 5px; }
                textarea { width: 100%; height: 100px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MoE++ Model Playground</h1>
                <div class="input-area">
                    <h3>Input Text</h3>
                    <textarea id="input-text" placeholder="Enter your text here..."></textarea>
                </div>
                <div class="parameters">
                    <h3>Parameters</h3>
                    <label>Temperature: <input type="range" id="temperature" min="0" max="2" step="0.1" value="1.0"></label><br>
                    <label>Top-K: <input type="number" id="top-k" value="50"></label><br>
                    <label>Max Length: <input type="number" id="max-length" value="100"></label>
                </div>
                <button onclick="runInference()">Run Inference</button>
                <button onclick="saveState()">Save State</button>
                <button onclick="loadState()">Load State</button>
                <div class="output-area">
                    <h3>Output</h3>
                    <pre id="output"></pre>
                </div>
                <div class="history">
                    <h3>Experiment History</h3>
                    <ul id="history-list"></ul>
                </div>
            </div>
            <script>
                function runInference() {
                    // In a real implementation, this would call the backend
                    document.getElementById('output').innerText = 'Running inference...';
                }
                function saveState() {
                    alert('State saved!');
                }
                function loadState() {
                    alert('State loaded!');
                }
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path


class AutomatedBenchmarking:
    """Automated benchmarking and performance analysis"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.benchmark_results = {}
    
    def run_benchmark(self, benchmark_suite: str = "default") -> Dict[str, float]:
        """Run comprehensive benchmark suite"""
        results = {}
        
        # Latency benchmark
        results.update(self._benchmark_latency())
        
        # Throughput benchmark
        results.update(self._benchmark_throughput())
        
        # Memory benchmark
        results.update(self._benchmark_memory())
        
        # Expert utilization benchmark
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_experts'):
            results.update(self._benchmark_expert_utilization())
        
        self.benchmark_results[benchmark_suite] = results
        return results
    
    def _benchmark_latency(self) -> Dict[str, float]:
        """Benchmark inference latency"""
        latencies = []
        
        for seq_len in [128, 256, 512]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(input_ids)
            
            # Measure
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.perf_counter()
                    _ = self.model(input_ids)
                    times.append(time.perf_counter() - start)
            
            latencies.append(np.mean(times) * 1000)  # Convert to ms
        
        return {
            'latency_128': latencies[0],
            'latency_256': latencies[1],
            'latency_512': latencies[2]
        }
    
    def _benchmark_throughput(self) -> Dict[str, float]:
        """Benchmark inference throughput"""
        batch_sizes = [1, 4, 8]
        throughputs = []
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, 1000, (batch_size, 256))
            
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(50):
                    _ = self.model(input_ids)
            elapsed = time.perf_counter() - start
            
            throughput = (50 * batch_size) / elapsed
            throughputs.append(throughput)
        
        return {
            'throughput_bs1': throughputs[0],
            'throughput_bs4': throughputs[1],
            'throughput_bs8': throughputs[2]
        }
    
    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory usage"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            # Run inference
            input_ids = torch.randint(0, 1000, (4, 512)).cuda()
            with torch.no_grad():
                _ = self.model(input_ids)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            
            return {
                'peak_memory_mb': peak_memory,
                'current_memory_mb': current_memory
            }
        else:
            return {
                'peak_memory_mb': 0,
                'current_memory_mb': 0
            }
    
    def _benchmark_expert_utilization(self) -> Dict[str, float]:
        """Benchmark expert utilization patterns"""
        num_experts = self.model.config.num_experts
        expert_counts = torch.zeros(num_experts)
        
        # Run multiple inferences
        for _ in range(100):
            input_ids = torch.randint(0, 1000, (1, 256))
            with torch.no_grad():
                outputs = self.model(input_ids)
                
                # Track expert usage if available
                if isinstance(outputs, dict) and 'expert_indices' in outputs:
                    indices = outputs['expert_indices'].flatten()
                    for idx in indices:
                        if 0 <= idx < num_experts:
                            expert_counts[idx] += 1
        
        # Calculate statistics
        total_usage = expert_counts.sum().item()
        if total_usage > 0:
            utilization = expert_counts / total_usage
            return {
                'expert_balance': 1.0 - utilization.std().item(),
                'most_used_expert': int(expert_counts.argmax().item()),
                'least_used_expert': int(expert_counts.argmin().item())
            }
        else:
            return {
                'expert_balance': 1.0,
                'most_used_expert': 0,
                'least_used_expert': 0
            }
    
    def compare_configurations(self, configs: List[Dict]) -> Dict[str, Dict]:
        """Compare different model configurations"""
        comparison_results = {}
        
        for i, config in enumerate(configs):
            # Apply configuration (simplified)
            config_name = f"config_{i}"
            
            # Run benchmarks
            results = self.run_benchmark(config_name)
            comparison_results[config_name] = results
        
        return comparison_results
    
    def generate_report(self, output_path: str):
        """Generate comprehensive benchmark report"""
        report = {
            'timestamp': time.time(),
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'layers': len(list(self.model.modules()))
            },
            'benchmarks': self.benchmark_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path


class VisualModelBuilder:
    """Visual interface for building model architectures"""
    
    def __init__(self):
        self.architecture_graph = {
            'nodes': [],
            'edges': []
        }
    
    def add_layer(self, layer_type: str, config: Dict) -> str:
        """Add a layer to the architecture"""
        layer_id = f"layer_{len(self.architecture_graph['nodes'])}"
        
        self.architecture_graph['nodes'].append({
            'id': layer_id,
            'type': layer_type,
            'config': config
        })
        
        # Auto-connect to previous layer
        if len(self.architecture_graph['nodes']) > 1:
            prev_id = self.architecture_graph['nodes'][-2]['id']
            self.add_connection(prev_id, layer_id)
        
        return layer_id
    
    def add_connection(self, from_id: str, to_id: str):
        """Add connection between layers"""
        self.architecture_graph['edges'].append({
            'from': from_id,
            'to': to_id
        })
    
    def export_architecture(self) -> Dict:
        """Export architecture definition"""
        return self.architecture_graph
    
    def build_model(self) -> nn.Module:
        """Build PyTorch model from visual definition"""
        # Simplified model building
        layers = []
        
        for node in self.architecture_graph['nodes']:
            if node['type'] == 'linear':
                layers.append(nn.Linear(
                    node['config'].get('in_features', 768),
                    node['config'].get('out_features', 768)
                ))
            elif node['type'] == 'attention':
                # Simplified attention layer
                layers.append(nn.MultiheadAttention(
                    node['config'].get('embed_dim', 768),
                    node['config'].get('num_heads', 8)
                ))
        
        return nn.Sequential(*layers) if layers else nn.Identity()


class AutomatedDocumentation:
    """Automated documentation generation for models"""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def generate_documentation(self) -> str:
        """Generate comprehensive model documentation"""
        doc = []
        doc.append("# MoE++ Model Documentation\n")
        doc.append("## Model Architecture\n")
        
        # Document layers
        for name, module in self.model.named_modules():
            if name:
                doc.append(f"### {name}\n")
                doc.append(f"- Type: {module.__class__.__name__}\n")
                
                # Document parameters
                params = list(module.parameters())
                if params:
                    total_params = sum(p.numel() for p in params)
                    doc.append(f"- Parameters: {total_params:,}\n")
        
        # Document configuration
        if hasattr(self.model, 'config'):
            doc.append("\n## Configuration\n")
            for key, value in self.model.config.__dict__.items():
                doc.append(f"- {key}: {value}\n")
        
        return ''.join(doc)


class InteractiveDebugging:
    """Interactive debugging tools for model development"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations[module] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients[module] = grad_output[0].detach()
        
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                fhook = module.register_forward_hook(forward_hook)
                bhook = module.register_backward_hook(backward_hook)
                self.hooks.extend([fhook, bhook])
    
    def analyze_activations(self) -> Dict[str, Any]:
        """Analyze activation patterns"""
        analysis = {}
        
        for module, activation in self.activations.items():
            analysis[str(module)] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'dead_neurons': (activation == 0).float().mean().item()
            }
        
        return analysis
    
    def analyze_gradients(self) -> Dict[str, Any]:
        """Analyze gradient flow"""
        analysis = {}
        
        for module, gradient in self.gradients.items():
            analysis[str(module)] = {
                'mean': gradient.mean().item(),
                'std': gradient.std().item(),
                'norm': gradient.norm().item()
            }
        
        return analysis
    
    def cleanup_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []