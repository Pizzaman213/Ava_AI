"""
Data Profiling and Statistics for Multi-Column Datasets

This module provides utilities for analyzing and profiling datasets,
computing statistics, and generating data quality reports.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .multi_column_data import (
    MultiColumnDataset,
    DatasetConfig,
    ColumnType,
    create_multi_column_dataloader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColumnStatistics:
    """Statistics for a single column"""
    name: str
    type: str
    count: int
    null_count: int
    unique_count: int

    # Text statistics
    avg_length: Optional[float] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    vocab_size: Optional[int] = None

    # Numeric statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None

    # Categorical statistics
    category_distribution: Optional[Dict[str, int]] = None

    # Image/tensor statistics
    shape_distribution: Optional[Dict[str, int]] = None
    dtype: Optional[str] = None


@dataclass
class DatasetProfile:
    """Complete profile of a dataset"""
    name: str
    total_samples: int
    column_stats: Dict[str, ColumnStatistics]
    missing_data_report: Dict[str, float]
    data_quality_score: float
    timestamp: str
    warnings: List[str]
    recommendations: List[str]


class DataProfiler:
    """
    Utility class for profiling multi-column datasets
    """

    def __init__(self, config: Union[DatasetConfig, str], tokenizer=None):
        """
        Initialize profiler

        Args:
            config: DatasetConfig or path to config file
            tokenizer: Optional tokenizer for text columns
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                import yaml
                config_dict = yaml.safe_load(f)
                self.config = self._parse_config(config_dict)
        else:
            self.config = config

        self.tokenizer = tokenizer
        self.stats = {}
        self.warnings = []
        self.recommendations = []

    def _parse_config(self, config_dict: Dict) -> DatasetConfig:
        """Parse configuration dictionary"""
        from .multi_column_data import ColumnConfig, ColumnType

        columns = []
        for col_dict in config_dict.get('columns', []):
            col_type = ColumnType(col_dict.get('type', 'text'))
            columns.append(ColumnConfig(
                name=col_dict['name'],
                type=col_type,
                role=col_dict.get('role', 'input'),
                preprocessing=col_dict.get('preprocessing', {}),
                max_length=col_dict.get('max_length'),
                normalize=col_dict.get('normalize', False),
                vocab=col_dict.get('vocab'),
                default_value=col_dict.get('default_value'),
                validation_rules=col_dict.get('validation_rules', {}),
                augmentation=col_dict.get('augmentation', {}),
                cache_processed=col_dict.get('cache_processed', False),
                required=col_dict.get('required', True)
            ))

        return DatasetConfig(
            columns=columns,
            combine_strategy=config_dict.get('combine_strategy', 'concatenate'),
            template=config_dict.get('template'),
            shuffle_buffer_size=config_dict.get('shuffle_buffer_size', 10000),
            max_samples=config_dict.get('max_samples'),
            hf_dataset_name=config_dict.get('hf_dataset_name'),
            hf_dataset_config=config_dict.get('hf_dataset_config'),
            hf_split=config_dict.get('hf_split'),
            cache_dir=config_dict.get('cache_dir'),
            validation_enabled=config_dict.get('validation_enabled', True),
            augmentation_probability=config_dict.get('augmentation_probability', 0.0)
        )

    def profile_dataset(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        sample_size: Optional[int] = None
    ) -> DatasetProfile:
        """
        Profile a dataset and generate statistics

        Args:
            data_dir: Directory containing data files
            split: Data split to profile
            sample_size: Number of samples to analyze (None for all)

        Returns:
            DatasetProfile with complete statistics
        """
        logger.info(f"Profiling dataset for {split} split...")

        # Create dataset
        dataset = MultiColumnDataset(
            config=self.config,
            tokenizer=self.tokenizer,
            data_dir=data_dir,
            split=split,
            streaming=False
        )

        # Limit sample size if specified
        n_samples = len(dataset)
        if sample_size:
            n_samples = min(sample_size, n_samples)

        # Collect statistics for each column
        column_data = defaultdict(list)
        missing_counts = defaultdict(int)

        for i in range(n_samples):
            if i % 1000 == 0:
                logger.info(f"Processing sample {i}/{n_samples}...")

            try:
                sample = dataset.data[i] if hasattr(dataset, 'data') else dataset[i]

                for col_config in self.config.columns:
                    col_name = col_config.name

                    if col_name in sample:
                        value = sample[col_name]
                        if value is not None:
                            column_data[col_name].append(value)
                        else:
                            missing_counts[col_name] += 1
                    else:
                        missing_counts[col_name] += 1

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                self.warnings.append(f"Failed to process sample {i}")

        # Compute statistics for each column
        column_stats = {}
        for col_config in self.config.columns:
            col_name = col_config.name
            col_type = col_config.type
            values = column_data.get(col_name, [])

            stats = self._compute_column_stats(
                col_name,
                col_type,
                values,
                missing_counts.get(col_name, 0),
                n_samples
            )

            column_stats[col_name] = stats

            # Generate warnings and recommendations
            self._generate_insights(col_config, stats)

        # Compute missing data percentages
        missing_data_report = {
            col: (count / n_samples * 100)
            for col, count in missing_counts.items()
        }

        # Calculate data quality score
        quality_score = self._calculate_quality_score(
            column_stats,
            missing_data_report
        )

        # Create profile
        profile = DatasetProfile(
            name=f"{split}_profile",
            total_samples=n_samples,
            column_stats=column_stats,
            missing_data_report=missing_data_report,
            data_quality_score=quality_score,
            timestamp=datetime.now().isoformat(),
            warnings=self.warnings,
            recommendations=self.recommendations
        )

        return profile

    def _compute_column_stats(
        self,
        name: str,
        col_type: ColumnType,
        values: List[Any],
        missing_count: int,
        total_count: int
    ) -> ColumnStatistics:
        """Compute statistics for a single column"""

        stats = ColumnStatistics(
            name=name,
            type=col_type.value,
            count=len(values),
            null_count=missing_count,
            unique_count=len(set(str(v) for v in values))
        )

        if not values:
            return stats

        if col_type == ColumnType.TEXT:
            lengths = [len(str(v)) for v in values]
            stats.avg_length = np.mean(lengths)
            stats.max_length = max(lengths)
            stats.min_length = min(lengths)

            # Vocabulary size (unique tokens)
            if self.tokenizer:
                all_tokens = []
                for text in values[:100]:  # Sample for efficiency
                    tokens = self.tokenizer.tokenize(str(text))
                    all_tokens.extend(tokens)
                stats.vocab_size = len(set(all_tokens))

        elif col_type == ColumnType.NUMERIC:
            numeric_values = [float(v) for v in values if v is not None]
            if numeric_values:
                stats.mean = np.mean(numeric_values)
                stats.std = np.std(numeric_values)
                stats.min_value = min(numeric_values)
                stats.max_value = max(numeric_values)
                stats.quantiles = {
                    "25%": np.percentile(numeric_values, 25),
                    "50%": np.percentile(numeric_values, 50),
                    "75%": np.percentile(numeric_values, 75)
                }

        elif col_type == ColumnType.CATEGORICAL:
            category_counts = Counter(values)
            stats.category_distribution = dict(category_counts.most_common(20))

        elif col_type in [ColumnType.IMAGE, ColumnType.TENSOR, ColumnType.EMBEDDING]:
            shapes = []
            for v in values[:100]:  # Sample for efficiency
                if hasattr(v, 'shape'):
                    shapes.append(str(v.shape))
                elif isinstance(v, (list, np.ndarray)):
                    shapes.append(str(np.array(v).shape))

            if shapes:
                shape_counts = Counter(shapes)
                stats.shape_distribution = dict(shape_counts.most_common(10))

            if values and hasattr(values[0], 'dtype'):
                stats.dtype = str(values[0].dtype)

        return stats

    def _generate_insights(self, col_config, stats: ColumnStatistics):
        """Generate warnings and recommendations based on statistics"""

        # Check for high missing data
        if stats.null_count > 0:
            missing_pct = (stats.null_count / (stats.count + stats.null_count)) * 100
            if missing_pct > 50 and col_config.required:
                self.warnings.append(
                    f"Column '{stats.name}' has {missing_pct:.1f}% missing data but is marked as required"
                )
            elif missing_pct > 20:
                self.recommendations.append(
                    f"Consider imputation strategy for column '{stats.name}' ({missing_pct:.1f}% missing)"
                )

        # Check text length consistency
        if col_config.type == ColumnType.TEXT and stats.max_length:
            if col_config.max_length and stats.max_length > col_config.max_length:
                self.warnings.append(
                    f"Column '{stats.name}' has text longer than configured max_length "
                    f"({stats.max_length} > {col_config.max_length})"
                )

        # Check numeric range
        if col_config.type == ColumnType.NUMERIC:
            rules = col_config.validation_rules
            if 'min_value' in rules and stats.min_value < rules['min_value']:
                self.warnings.append(
                    f"Column '{stats.name}' has values below validation minimum "
                    f"({stats.min_value} < {rules['min_value']})"
                )
            if 'max_value' in rules and stats.max_value > rules['max_value']:
                self.warnings.append(
                    f"Column '{stats.name}' has values above validation maximum "
                    f"({stats.max_value} > {rules['max_value']})"
                )

        # Check categorical distribution
        if col_config.type == ColumnType.CATEGORICAL and stats.category_distribution:
            if col_config.vocab:
                unknown_categories = set(stats.category_distribution.keys()) - set(col_config.vocab)
                if unknown_categories:
                    self.warnings.append(
                        f"Column '{stats.name}' has unknown categories: {unknown_categories}"
                    )

            # Check for imbalanced categories
            if len(stats.category_distribution) > 1:
                values = list(stats.category_distribution.values())
                max_count = max(values)
                min_count = min(values)
                if max_count / min_count > 10:
                    self.recommendations.append(
                        f"Column '{stats.name}' has imbalanced categories (ratio {max_count/min_count:.1f})"
                    )

    def _calculate_quality_score(
        self,
        column_stats: Dict[str, ColumnStatistics],
        missing_data_report: Dict[str, float]
    ) -> float:
        """Calculate overall data quality score (0-100)"""

        scores = []

        # Completeness score (based on missing data)
        avg_missing = np.mean(list(missing_data_report.values())) if missing_data_report else 0
        completeness_score = max(0, 100 - avg_missing)
        scores.append(completeness_score)

        # Consistency score (based on warnings)
        consistency_score = max(0, 100 - len(self.warnings) * 10)
        scores.append(consistency_score)

        # Validity score (based on unique values and distributions)
        validity_scores = []
        for stats in column_stats.values():
            if stats.count > 0:
                # High unique count relative to total is good for most columns
                uniqueness_ratio = stats.unique_count / stats.count
                validity_scores.append(min(100, uniqueness_ratio * 100))

        if validity_scores:
            scores.append(np.mean(validity_scores))

        return np.mean(scores) if scores else 0.0

    def save_profile(self, profile: DatasetProfile, output_path: str):
        """Save profile to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        profile_dict = asdict(profile)

        with open(output_path, 'w') as f:
            json.dump(profile_dict, f, indent=2, default=str)

        logger.info(f"Profile saved to {output_path}")

    def generate_report(
        self,
        profile: DatasetProfile,
        output_dir: str,
        include_plots: bool = True
    ):
        """Generate HTML report with visualizations"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots if requested
        if include_plots:
            self._generate_plots(profile, output_dir)

        # Generate HTML report
        html_content = self._generate_html_report(profile, include_plots)

        report_path = output_dir / "data_profile_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Report generated at {report_path}")

    def _generate_plots(self, profile: DatasetProfile, output_dir: Path):
        """Generate visualization plots"""

        # Missing data plot
        if profile.missing_data_report:
            plt.figure(figsize=(10, 6))
            columns = list(profile.missing_data_report.keys())
            missing_pct = list(profile.missing_data_report.values())

            plt.bar(columns, missing_pct)
            plt.xlabel('Column')
            plt.ylabel('Missing Data (%)')
            plt.title('Missing Data by Column')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'missing_data.png')
            plt.close()

        # Column type distribution
        type_counts = Counter(stats.type for stats in profile.column_stats.values())
        if type_counts:
            plt.figure(figsize=(8, 6))
            plt.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            plt.title('Column Type Distribution')
            plt.savefig(output_dir / 'column_types.png')
            plt.close()

    def _generate_html_report(self, profile: DatasetProfile, include_plots: bool) -> str:
        """Generate HTML report content"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .warning {{ color: orange; }}
                .recommendation {{ color: blue; }}
                .score {{ font-size: 24px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Dataset Profile Report</h1>
            <p><strong>Generated:</strong> {profile.timestamp}</p>
            <p><strong>Total Samples:</strong> {profile.total_samples}</p>
            <p class="score">Data Quality Score: {profile.data_quality_score:.1f}/100</p>

            <h2>Column Statistics</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Count</th>
                    <th>Missing %</th>
                    <th>Unique</th>
                    <th>Details</th>
                </tr>
        """

        for col_name, stats in profile.column_stats.items():
            missing_pct = profile.missing_data_report.get(col_name, 0)

            details = []
            if stats.avg_length:
                details.append(f"Avg length: {stats.avg_length:.1f}")
            if stats.mean is not None:
                details.append(f"Mean: {stats.mean:.2f}")
            if stats.category_distribution:
                top_cat = list(stats.category_distribution.keys())[0]
                details.append(f"Top: {top_cat}")

            html += f"""
                <tr>
                    <td>{col_name}</td>
                    <td>{stats.type}</td>
                    <td>{stats.count}</td>
                    <td>{missing_pct:.1f}%</td>
                    <td>{stats.unique_count}</td>
                    <td>{', '.join(details)}</td>
                </tr>
            """

        html += "</table>"

        if profile.warnings:
            html += "<h2>Warnings</h2><ul>"
            for warning in profile.warnings:
                html += f'<li class="warning">{warning}</li>'
            html += "</ul>"

        if profile.recommendations:
            html += "<h2>Recommendations</h2><ul>"
            for rec in profile.recommendations:
                html += f'<li class="recommendation">{rec}</li>'
            html += "</ul>"

        if include_plots:
            html += """
                <h2>Visualizations</h2>
                <img src="missing_data.png" alt="Missing Data" style="max-width:600px;">
                <img src="column_types.png" alt="Column Types" style="max-width:400px;">
            """

        html += """
        </body>
        </html>
        """

        return html


def validate_config(config_path: str) -> List[str]:
    """
    Validate a dataset configuration file

    Args:
        config_path: Path to configuration YAML file

    Returns:
        List of validation errors (empty if valid)
    """
    import yaml

    errors = []

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required fields
        if 'columns' not in config:
            errors.append("Missing required field 'columns'")
            return errors

        if not isinstance(config['columns'], list):
            errors.append("'columns' must be a list")
            return errors

        # Validate each column
        column_names = set()
        for i, col in enumerate(config['columns']):
            if 'name' not in col:
                errors.append(f"Column {i} missing 'name' field")
            else:
                if col['name'] in column_names:
                    errors.append(f"Duplicate column name: {col['name']}")
                column_names.add(col['name'])

            if 'type' not in col:
                errors.append(f"Column {col.get('name', i)} missing 'type' field")
            else:
                valid_types = ['text', 'image', 'audio', 'video', 'numeric',
                              'categorical', 'embedding', 'json', 'binary', 'tensor']
                if col['type'] not in valid_types:
                    errors.append(f"Invalid type '{col['type']}' for column {col.get('name', i)}")

            if 'role' in col:
                valid_roles = ['input', 'target', 'auxiliary', 'weight']
                if col['role'] not in valid_roles:
                    errors.append(f"Invalid role '{col['role']}' for column {col.get('name', i)}")

        # Validate combine strategy
        if 'combine_strategy' in config:
            valid_strategies = ['concatenate', 'separate', 'template', 'custom']
            if config['combine_strategy'] not in valid_strategies:
                errors.append(f"Invalid combine_strategy: {config['combine_strategy']}")

            if config['combine_strategy'] == 'template' and 'template' not in config:
                errors.append("'template' field required when combine_strategy is 'template'")

    except Exception as e:
        errors.append(f"Error parsing configuration: {str(e)}")

    return errors


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile multi-column datasets")
    parser.add_argument("config", help="Path to dataset configuration file")
    parser.add_argument("--data-dir", help="Directory containing data files")
    parser.add_argument("--split", default="train", help="Data split to profile")
    parser.add_argument("--sample-size", type=int, help="Number of samples to analyze")
    parser.add_argument("--output-dir", default="./profile_output", help="Output directory for report")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")

    args = parser.parse_args()

    if args.validate_only:
        errors = validate_config(args.config)
        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Configuration is valid!")
    else:
        profiler = DataProfiler(args.config)
        profile = profiler.profile_dataset(
            data_dir=args.data_dir,
            split=args.split,
            sample_size=args.sample_size
        )

        # Save profile and generate report
        profiler.save_profile(profile, f"{args.output_dir}/profile.json")
        profiler.generate_report(profile, args.output_dir)

        print(f"\n Data Quality Score: {profile.data_quality_score:.1f}/100")
        print(f" Report generated at {args.output_dir}/data_profile_report.html")