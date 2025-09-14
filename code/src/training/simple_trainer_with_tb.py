"""
Simple trainer with TensorBoard support
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from .simple_trainer import SimpleMoETrainer

class SimpleMoETrainerWithTB(SimpleMoETrainer):
    """Extended trainer with TensorBoard logging"""
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.output_dir / "logs")
        print(f"TensorBoard logs will be saved to: {self.output_dir / 'logs'}")
        
    def log_metrics(self, metrics, step):
        """Log metrics to both console and TensorBoard"""
        # Call parent method for console logging
        super().log_metrics(metrics, step)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"train/{key}", value, step)
        
        # Flush to ensure data is written
        self.tb_writer.flush()
    
    def evaluate(self, val_dataloader):
        """Evaluate model and log to TensorBoard"""
        avg_loss = super().evaluate(val_dataloader)
        
        # Log validation loss to TensorBoard
        self.tb_writer.add_scalar("val/loss", avg_loss, self.global_step)
        self.tb_writer.flush()
        
        return avg_loss
    
    def train(self):
        """Override to close TensorBoard writer after training"""
        try:
            super().train()
        finally:
            if hasattr(self, 'tb_writer'):
                self.tb_writer.close()
                print(f"\nTensorBoard logs saved. Run: tensorboard --logdir {self.output_dir / 'logs'}")