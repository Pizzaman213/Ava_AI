"""
Run Manager for Organizing Training Experiments

This module provides the RunManager class that handles:
- Creating unique run directories with proper organization
- Setting up structured logging for different aspects of training
- Saving run metadata and configurations
- Tracking metrics across training runs
- Managing checkpoints with run context
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid
import yaml
import torch  # type: ignore[import]


class RunManager:
    """
    Manages training runs with organized directory structure and logging.

    Directory structure created:
    runs/
    ├── run_YYYYMMDD_HHMMSS_ABC123/
    │   ├── logs/
    │   │   ├── training.log
    │   │   ├── evaluation.log
    │   │   ├── errors.log
    │   │   └── debug.log
    │   ├── checkpoints/
    │   │   ├── best_model.pt
    │   │   ├── latest_model.pt
    │   │   └── step_N/
    │   ├── configs/
    │   │   ├── model_config.yaml
    │   │   ├── training_config.yaml
    │   │   └── run_metadata.json
    │   ├── metrics/
    │   │   ├── training_metrics.json
    │   │   ├── evaluation_metrics.json
    │   │   └── loss_curves.json
    │   └── outputs/
    │       ├── generated_samples.txt
    │       └── visualizations/
    """

    def __init__(self,
                 base_output_dir: str = "/project/code/outputs",
                 run_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 description: Optional[str] = None):
        """
        Initialize a new training run.

        Args:
            base_output_dir: Base directory for all runs
            run_name: Optional custom name for the run
            tags: Optional tags for categorizing the run
            description: Optional description of the experiment
        """
        self.base_output_dir = Path(base_output_dir)
        self.tags = tags or []
        self.description = description

        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        if run_name:
            self.run_id = f"{run_name}_{timestamp}_{unique_id}"
        else:
            self.run_id = f"run_{timestamp}_{unique_id}"

        # Create run directory structure
        self.run_dir = self.base_output_dir / "runs" / self.run_id
        self._create_directory_structure()

        # Initialize logging
        self.loggers = {}
        self._setup_logging()

        # Initialize metrics tracking
        self.metrics = {
            'training': [],
            'evaluation': [],
            'loss_curves': {'train_loss': [], 'val_loss': [], 'steps': []}
        }

        # Save initial run metadata
        self._save_run_metadata()

        print(f"Created new training run: {self.run_id}")
        print(f"Run directory: {self.run_dir}")

    def _create_directory_structure(self):
        """Create the organized directory structure for the run."""
        subdirs = [
            'logs',
            'checkpoints',
            'configs',
            'metrics',
            'outputs',
            'outputs/visualizations'
        ]

        for subdir in subdirs:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Set up multiple loggers for different aspects of training."""
        log_configs = {
            'training': {
                'level': logging.INFO,
                'format': '%(asctime)s - %(levelname)s - [TRAIN] %(message)s',
                'file': 'logs/training.log'
            },
            'evaluation': {
                'level': logging.INFO,
                'format': '%(asctime)s - %(levelname)s - [EVAL] %(message)s',
                'file': 'logs/evaluation.log'
            },
            'errors': {
                'level': logging.ERROR,
                'format': '%(asctime)s - %(levelname)s - [ERROR] %(message)s',
                'file': 'logs/errors.log'
            },
            'debug': {
                'level': logging.DEBUG,
                'format': '%(asctime)s - %(levelname)s - [DEBUG] %(message)s',
                'file': 'logs/debug.log'
            }
        }

        for logger_name, config in log_configs.items():
            logger = logging.getLogger(f"{self.run_id}_{logger_name}")
            logger.setLevel(config['level'])

            # Remove existing handlers to avoid duplication
            logger.handlers.clear()

            # File handler
            file_handler = logging.FileHandler(self.run_dir / config['file'])
            file_handler.setLevel(config['level'])
            file_formatter = logging.Formatter(config['format'])
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Console handler for training and errors
            if logger_name in ['training', 'errors']:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(config['level'])
                console_formatter = logging.Formatter(config['format'])
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)

            self.loggers[logger_name] = logger

    def _save_run_metadata(self):
        """Save initial run metadata."""
        metadata = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'tags': self.tags,
            'description': self.description,
            'status': 'started',
            'total_epochs': None,
            'total_steps': 0,
            'best_val_loss': float('inf'),
            'system_info': {
                'python_version': f"{__import__('sys').version}",
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

        with open(self.run_dir / 'configs/run_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_config(self, config: Dict[str, Any], config_type: str = 'model'):
        """
        Save configuration files for the run.

        Args:
            config: Configuration dictionary to save
            config_type: Type of config ('model', 'training', 'data', etc.)
        """
        config_file = self.run_dir / f'configs/{config_type}_config.yaml'

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        self.log('training', f"Saved {config_type} config to {config_file}")

    def save_args(self, args):
        """Save command line arguments."""
        args_dict = vars(args) if hasattr(args, '__dict__') else args

        with open(self.run_dir / 'configs/args.json', 'w') as f:
            json.dump(args_dict, f, indent=2, default=str)

    def log(self, logger_name: str, message: str, level: str = 'info'):
        """
        Log a message using the specified logger.

        Args:
            logger_name: Name of the logger ('training', 'evaluation', 'errors', 'debug')
            message: Message to log
            level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        """
        if logger_name not in self.loggers:
            print(f"Warning: Logger '{logger_name}' not found. Using training logger.")
            logger_name = 'training'

        logger = self.loggers[logger_name]
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(message)

    def log_metrics(self, metrics: Dict[str, float], step: int, epoch: Optional[int] = None, metric_type: str = 'training'):
        """
        Log metrics for a training step.

        Args:
            metrics: Dictionary of metric names to values
            step: Training step number
            epoch: Training epoch number (optional)
            metric_type: Type of metrics ('training' or 'evaluation')
        """
        # Add to metrics tracking
        metric_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        self.metrics[metric_type].append(metric_entry)

        # Update loss curves if loss values are present
        if 'total_loss' in metrics:
            self.metrics['loss_curves']['steps'].append(step)
            if metric_type == 'training':
                self.metrics['loss_curves']['train_loss'].append(metrics['total_loss'])
            elif metric_type == 'evaluation':
                self.metrics['loss_curves']['val_loss'].append(metrics['total_loss'])

        # Save metrics to file
        self._save_metrics()

        # Log to appropriate logger
        metric_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        log_message = f"Step {step}"
        if epoch is not None:
            log_message += f" (Epoch {epoch})"
        log_message += f" - {metric_str}"

        logger_name = 'evaluation' if metric_type == 'evaluation' else 'training'
        self.log(logger_name, log_message)

    def _save_metrics(self):
        """Save current metrics to JSON files."""
        # Save individual metric types
        for metric_type, data in self.metrics.items():
            if metric_type == 'loss_curves':
                continue
            file_path = self.run_dir / f'metrics/{metric_type}_metrics.json'
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

        # Save loss curves
        loss_curves_path = self.run_dir / 'metrics/loss_curves.json'
        with open(loss_curves_path, 'w') as f:
            json.dump(self.metrics['loss_curves'], f, indent=2)

    def save_checkpoint(self,
                       model_state: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       epoch: int,
                       step: int,
                       loss: float,
                       is_best: bool = False,
                       additional_data: Optional[Dict[str, Any]] = None):
        """
        Save a model checkpoint with run context using atomic write pattern.

        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            epoch: Current epoch
            step: Current step
            loss: Current loss value
            is_best: Whether this is the best checkpoint so far
            additional_data: Additional data to save with checkpoint
        """
        import shutil
        import os

        checkpoint_data = {
            'run_id': self.run_id,
            'epoch': epoch,
            'step': step,  # Legacy: can be either optimizer or micro steps
            'micro_step_count': step,  # Forward compatibility: explicit micro step count
            'optimizer_step_count': step,  # Forward compatibility: explicit optimizer step count
            'loss': loss,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'timestamp': datetime.now().isoformat(),
            'run_directory': str(self.run_dir)
        }

        if additional_data:
            checkpoint_data.update(additional_data)

        # Check available disk space before saving (require at least 2GB free)
        checkpoint_dir = self.run_dir / 'checkpoints'
        stat = os.statvfs(checkpoint_dir)
        free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

        if free_space_gb < 2.0:
            self.log('error', f"Insufficient disk space: {free_space_gb:.2f}GB free, need at least 2GB")
            raise RuntimeError(f"Insufficient disk space for checkpoint: {free_space_gb:.2f}GB available")

        # Save latest checkpoint atomically
        latest_path = self.run_dir / 'checkpoints/latest_model.pt'
        self._atomic_save(checkpoint_data, latest_path)

        # Save step-specific checkpoint atomically
        step_dir = self.run_dir / f'checkpoints/step_{step}'
        step_dir.mkdir(exist_ok=True)
        step_path = step_dir / 'model.pt'
        self._atomic_save(checkpoint_data, step_path)

        # Save best checkpoint if applicable
        if is_best:
            best_path = self.run_dir / 'checkpoints/best_model.pt'
            self._atomic_save(checkpoint_data, best_path)
            self.log('training', f"Saved new best checkpoint with loss {loss:.6f}")

            # Update run metadata
            self._update_best_loss(loss)

        self.log('training', f"Saved checkpoint at step {step} (epoch {epoch})")

        # Return the path to the latest checkpoint
        return str(latest_path)

    def _atomic_save(self, checkpoint_data: Dict[str, Any], final_path: Path):
        """
        Atomically save checkpoint using temp file + rename pattern.

        Args:
            checkpoint_data: Checkpoint data to save
            final_path: Final destination path
        """
        import tempfile
        import shutil

        # Create temp file in same directory to ensure same filesystem
        temp_fd, temp_path = tempfile.mkstemp(
            dir=final_path.parent,
            prefix='.tmp_checkpoint_',
            suffix='.pt'
        )

        try:
            # Close the file descriptor, we'll use torch.save
            os.close(temp_fd)

            # Save to temp file
            torch.save(checkpoint_data, temp_path)

            # Atomic rename (only works on same filesystem)
            shutil.move(temp_path, final_path)

        except Exception as e:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except:
                pass
            raise RuntimeError(f"Failed to save checkpoint atomically: {e}") from e

    def _update_best_loss(self, loss: float):
        """Update the best validation loss in run metadata."""
        metadata_path = self.run_dir / 'configs/run_metadata.json'

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['best_val_loss'] = loss
        metadata['last_updated'] = datetime.now().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def finish_run(self, status: str = 'completed', final_metrics: Optional[Dict[str, float]] = None):
        """
        Mark the run as finished and save final metadata.

        Args:
            status: Final status ('completed', 'failed', 'interrupted')
            final_metrics: Final metrics to save
        """
        metadata_path = self.run_dir / 'configs/run_metadata.json'

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata['status'] = status
        metadata['end_time'] = datetime.now().isoformat()
        metadata['total_steps'] = len(self.metrics['training'])

        if final_metrics:
            metadata['final_metrics'] = final_metrics

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.log('training', f"Run {self.run_id} finished with status: {status}")

    def get_checkpoint_path(self, checkpoint_type: str = 'latest') -> Path:
        """
        Get the path to a specific checkpoint.

        Args:
            checkpoint_type: Type of checkpoint ('latest', 'best', or 'step_N')

        Returns:
            Path to the checkpoint file
        """
        if checkpoint_type == 'latest':
            return self.run_dir / 'checkpoints/latest_model.pt'
        elif checkpoint_type == 'best':
            return self.run_dir / 'checkpoints/best_model.pt'
        elif checkpoint_type.startswith('step_'):
            step_num = checkpoint_type.split('_')[1]
            return self.run_dir / f'checkpoints/step_{step_num}/model.pt'
        else:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")

    def save_generated_samples(self, samples: List[str], filename: str = 'generated_samples.txt'):
        """Save generated text samples from the model."""
        output_file = self.run_dir / 'outputs' / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Generated samples from run: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")

            for i, sample in enumerate(samples, 1):
                f.write(f"Sample {i}:\n{sample}\n\n")

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Clean up old step-specific checkpoints, keeping only the most recent ones.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoint_dir = self.run_dir / 'checkpoints'
        step_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]

        if len(step_dirs) <= keep_last_n:
            return

        # Sort by step number
        step_dirs.sort(key=lambda x: int(x.name.split('_')[1]))

        # Remove old checkpoints
        for dir_to_remove in step_dirs[:-keep_last_n]:
            shutil.rmtree(dir_to_remove)
            self.log('debug', f"Removed old checkpoint: {dir_to_remove.name}")

    @classmethod
    def list_runs(cls, base_output_dir: str = "/project/code/outputs") -> List[Dict[str, Any]]:
        """
        List all training runs with their metadata.

        Args:
            base_output_dir: Base directory containing runs

        Returns:
            List of run metadata dictionaries
        """
        runs_dir = Path(base_output_dir) / "runs"

        if not runs_dir.exists():
            return []

        runs = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / 'configs/run_metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata['run_directory'] = str(run_dir)
                    runs.append(metadata)

        # Sort by start time (most recent first)
        runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return runs

    @classmethod
    def load_run(cls, run_id: str, base_output_dir: str = "/project/code/outputs") -> 'RunManager':
        """
        Load an existing run for analysis or resuming.

        Args:
            run_id: ID of the run to load
            base_output_dir: Base directory containing runs

        Returns:
            RunManager instance for the existing run
        """
        run_dir = Path(base_output_dir) / "runs" / run_id

        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found in {base_output_dir}")

        metadata_file = run_dir / 'configs/run_metadata.json'
        if not metadata_file.exists():
            raise ValueError(f"Run metadata not found for {run_id}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Create a new RunManager instance but point to existing directory
        manager = cls.__new__(cls)
        manager.run_id = run_id
        manager.run_dir = run_dir
        manager.base_output_dir = Path(base_output_dir)
        manager.tags = metadata.get('tags', [])
        manager.description = metadata.get('description')

        # Setup logging for existing run
        manager.loggers = {}
        manager._setup_logging()

        # Load existing metrics
        manager.metrics = {'training': [], 'evaluation': [], 'loss_curves': {'train_loss': [], 'val_loss': [], 'steps': []}}
        manager._load_existing_metrics()

        return manager

    def _load_existing_metrics(self):
        """Load existing metrics from JSON files."""
        metrics_dir = self.run_dir / 'metrics'

        # Load training metrics
        training_file = metrics_dir / 'training_metrics.json'
        if training_file.exists():
            with open(training_file, 'r') as f:
                self.metrics['training'] = json.load(f)

        # Load evaluation metrics
        eval_file = metrics_dir / 'evaluation_metrics.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                self.metrics['evaluation'] = json.load(f)

        # Load loss curves
        loss_file = metrics_dir / 'loss_curves.json'
        if loss_file.exists():
            with open(loss_file, 'r') as f:
                self.metrics['loss_curves'] = json.load(f)