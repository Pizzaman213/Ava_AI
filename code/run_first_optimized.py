#!/usr/bin/env python3
"""Optimized initial setup script for LLM project with better error handling and performance."""

import subprocess
import sys
import os
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SetupState:
    """Manages setup state for resumable execution."""
    
    def __init__(self, state_file: Path = Path(".setup_state.json")):
        self.state_file = state_file
        self.state = self.load_state()
    
    def load_state(self) -> Dict:
        """Load previous state if exists."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "completed_steps": [],
            "failed_steps": [],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_state(self):
        """Save current state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_completed(self, step: str):
        """Mark a step as completed."""
        if step not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step)
            self.save_state()
    
    def mark_failed(self, step: str, error: str):
        """Mark a step as failed."""
        self.state["failed_steps"].append({
            "step": step,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.save_state()
    
    def is_completed(self, step: str) -> bool:
        """Check if a step was completed."""
        return step in self.state["completed_steps"]
    
    def reset(self):
        """Reset state."""
        self.state = {
            "completed_steps": [],
            "failed_steps": [],
            "timestamp": datetime.now().isoformat()
        }
        self.save_state()


class OptimizedSetup:
    """Optimized setup runner with better error handling and performance."""
    
    def __init__(self, project_dir: Path = None):
        self.project_dir = project_dir or Path(__file__).parent
        self.state = SetupState(self.project_dir / ".setup_state.json")
        self.total_steps = 0
        self.completed_steps = 0
        self.start_time = time.time()
        
    def check_prerequisites(self) -> bool:
        """Check if required tools are installed."""
        required_tools = ["python3", "git", "pip"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            return False
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        
        return True
    
    def check_disk_space(self, required_gb: int = 10) -> bool:
        """Check if sufficient disk space is available."""
        import shutil
        stat = shutil.disk_usage(self.project_dir)
        available_gb = stat.free / (1024 ** 3)
        
        if available_gb < required_gb:
            logger.warning(f"Low disk space: {available_gb:.1f}GB available, {required_gb}GB recommended")
            return False
        
        return True
    
    def check_resource_exists(self, path: Path) -> bool:
        """Enhanced check for resource existence."""
        if not path.exists():
            return False
        
        if path.is_dir():
            # Check if directory has content
            try:
                return any(path.iterdir())
            except PermissionError:
                return False
        else:
            # Check if file is not empty
            return path.stat().st_size > 0
    
    def run_command_with_retry(
        self, 
        cmd: str, 
        description: str, 
        skip_if_exists: Optional[Path] = None,
        max_retries: int = 2,
        timeout: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Run command with retry logic and timeout."""
        
        # Check if step was already completed
        step_id = hashlib.md5(cmd.encode()).hexdigest()[:8]
        if self.state.is_completed(step_id):
            logger.info(f"✓ Already completed: {description}")
            return True, description
        
        # Check if we should skip
        if skip_if_exists and self.check_resource_exists(skip_if_exists):
            logger.info(f"✓ Skipping: {description} (already exists)")
            self.state.mark_completed(step_id)
            return True, description
        
        # Try to run the command
        for attempt in range(max_retries):
            logger.info(f"{'='*60}")
            logger.info(f"[{self.completed_steps}/{self.total_steps}] {description}")
            if attempt > 0:
                logger.info(f"Retry attempt {attempt}/{max_retries}")
            logger.info(f"Command: {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
            
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    text=True,
                    capture_output=False,
                    timeout=timeout
                )
                
                logger.info(f"✓ {description} completed")
                self.state.mark_completed(step_id)
                self.completed_steps += 1
                return True, description
                
            except subprocess.TimeoutExpired:
                logger.error(f"✗ {description} timed out after {timeout}s")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                self.state.mark_failed(step_id, "Timeout")
                return False, description
                
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ {description} failed with error code {e.returncode}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                self.state.mark_failed(step_id, str(e))
                return False, description
        
        return False, description
    
    def run_parallel_commands(
        self, 
        commands: List[Tuple[str, str, Optional[Path]]],
        max_workers: int = 4
    ) -> bool:
        """Run multiple commands in parallel with better resource management."""
        
        # Adjust workers based on system resources
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(max_workers, cpu_count, len(commands))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {len(commands)} commands in parallel (max {max_workers} workers)")
        logger.info('='*60)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for cmd, desc, skip_path in commands:
                future = executor.submit(
                    self.run_command_with_retry,
                    cmd, desc, skip_path
                )
                futures.append(future)
            
            # Process results as they complete
            failed = []
            for future in as_completed(futures):
                success, desc = future.result()
                if not success:
                    failed.append(desc)
            
            if failed:
                logger.error(f"Failed tasks: {', '.join(failed)}")
                return False
        
        return True
    
    def validate_config(self, config_path: str) -> bool:
        """Validate configuration file before training."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['model', 'training']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field in config: {field}")
                    return False
            
            logger.info(f"✓ Config validated: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    def optimize_for_platform(self) -> Dict:
        """Detect platform and return optimized settings."""
        import platform
        
        system = platform.system()
        settings = {
            "num_workers": 4,
            "batch_size": 32,
            "device": "cpu"
        }
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                settings["device"] = "cuda"
                settings["batch_size"] = 64
                settings["num_workers"] = 8
            elif torch.backends.mps.is_available():
                settings["device"] = "mps"
                settings["batch_size"] = 64
                settings["num_workers"] = 8
        except ImportError:
            pass
        
        # Adjust for system
        if system == "Darwin":  # macOS
            settings["num_workers"] = min(settings["num_workers"], 6)
        elif system == "Windows":
            settings["num_workers"] = min(settings["num_workers"], 4)
        
        logger.info(f"Platform: {system}, Device: {settings['device']}")
        return settings
    
    def run(self, resume: bool = True):
        """Run the optimized setup process."""
        
        # Change to project directory
        os.chdir(self.project_dir)
        logger.info(f"Starting optimized setup in: {self.project_dir}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed")
            sys.exit(1)
        
        # Check disk space
        if not self.check_disk_space():
            response = input("Continue with low disk space? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Reset state if not resuming
        if not resume:
            self.state.reset()
        
        # Get platform settings
        platform_settings = self.optimize_for_platform()
        
        # Define all setup steps
        initial_commands = [
            ("python3 scripts/utils/setup_folders.py",
             "Setup project folders",
             None),
            
            ("python3 scripts/utils/fast_pip_install.py", 
             "Fast pip install",
             None),
        ]
        
        download_commands = [
            ("python3 scripts/data_download/download_full_datasets.py --datasets wikipedia code --max_samples 50000",
             "Download Wikipedia and code datasets (50k)",
             Path("data/pretraining/raw/full_datasets")),
            
            ("python3 scripts/data_download/download_full_datasets.py --datasets all --max_samples 10000",
             "Download all datasets (10k)",
             Path("data/pretraining/raw/full_datasets"))
        ]
        
        # Adjust batch size based on platform
        batch_size = platform_settings["batch_size"]
        
        final_commands = [
            ("python3 scripts/data_prep/prepare_data.py --input-path data/pretraining/raw/full_datasets "
             "--output-dir data/pretraining/processed --input-format json --output-format jsonl "
             "--max-length 512 --fast-mode --tokenizer gpt2",
             "Prepare data for training",
             Path("data/pretraining/processed")),
            
            (f"python3 scripts/training/train_with_data.py --config configs/mps/small.yaml "
             f"--epochs 10 --batch-size {batch_size} --tokenizer-path gpt2",
             "Train model with prepared data",
             None)
        ]
        
        # Calculate total steps
        self.total_steps = len(initial_commands) + len(download_commands) + len(final_commands)
        
        # Run initial setup
        logger.info("\n🚀 Phase 1: Initial Setup")
        for cmd, desc, skip_path in initial_commands:
            success, _ = self.run_command_with_retry(cmd, desc, skip_path)
            if not success:
                logger.error(f"Critical failure in: {desc}")
                sys.exit(1)
        
        # Run downloads in parallel
        logger.info("\n📥 Phase 2: Data Downloads")
        if not self.run_parallel_commands(download_commands, max_workers=2):
            logger.error("Download phase failed")
            sys.exit(1)
        
        # Validate config before training
        config_path = "configs/mps/small.yaml"
        if not self.validate_config(config_path):
            logger.error("Config validation failed")
            sys.exit(1)
        
        # Run final commands
        logger.info("\n🔨 Phase 3: Data Processing & Training")
        for cmd, desc, skip_path in final_commands:
            success, _ = self.run_command_with_retry(
                cmd, desc, skip_path, 
                timeout=3600 if "train" in cmd.lower() else None
            )
            if not success:
                logger.error(f"Failure in: {desc}")
                sys.exit(1)
        
        # Report completion
        elapsed_time = time.time() - self.start_time
        logger.info("\n" + "="*60)
        logger.info(f"✅ Setup completed successfully!")
        logger.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        logger.info(f"📊 Steps completed: {self.completed_steps}/{self.total_steps}")
        logger.info("="*60)
        
        # Cleanup state file
        self.state.state_file.unlink(missing_ok=True)


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized LLM Project Setup")
    parser.add_argument("--reset", action="store_true", help="Reset and start fresh")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = OptimizedSetup()
    
    try:
        setup.run(resume=not args.reset)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Setup interrupted by user")
        logger.info("Run again to resume from last successful step")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.info("Check .setup_state.json for details")
        sys.exit(1)


if __name__ == "__main__":
    main()