#!/usr/bin/env python3
"""Add assertion guards to benchmark methods in unified_benchmark.py"""

import re

# Read the file
with open('/project/code/scripts/benchmarks/unified_benchmark.py', 'r') as f:
    content = f.read()

# List of methods that need assertions (from error list)
methods_needing_assertions = [
    '_benchmark_attention_mechanisms',
    '_benchmark_mixed_precision_training',
    '_benchmark_cnn_workload',
    '_benchmark_language_model_workload',
    '_benchmark_embedding_workload',
    '_benchmark_dataloader_performance',
    '_benchmark_threading_performance',
    '_benchmark_memory_intensive_workloads',
    '_benchmark_matrix_operations',
    '_benchmark_memory_bandwidth_stress',
    '_benchmark_cache_performance',
    '_benchmark_cpu_ai_acceleration',
    '_benchmark_load_balancing',
    '_benchmark_data_pipeline_efficiency',
    '_benchmark_memory_transfer_efficiency',
]

# Pattern to find method definitions and their first line
for method_name in methods_needing_assertions:
    # Find the method definition
    pattern = rf'(    def {method_name}\([^)]+\)[^:]*:\s*\n\s*"""[^"]+"""\s*\n)'

    def add_assertion(match):
        original = match.group(1)
        # Check if assertion already exists
        if 'assert self.system_capabilities is not None' in original:
            return original
        # Add assertion after the docstring
        return original + '        assert self.system_capabilities is not None, "System capabilities must be initialized"\n'

    content = re.sub(pattern, add_assertion, content)

# Write back
with open('/project/code/scripts/benchmarks/unified_benchmark.py', 'w') as f:
    f.write(content)

print("Added assertions to benchmark methods")
