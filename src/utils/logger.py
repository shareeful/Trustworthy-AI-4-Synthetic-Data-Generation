"""
Logging utilities for experiment tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


class ExperimentLogger:
    """Custom logger for experiment tracking with file and console output."""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        level: int = logging.INFO
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save log files
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Initialized logger for experiment: {experiment_name}")
        self.logger.info(f"Log file: {log_file}")
        
        # Metrics storage
        self.metrics = {}
        self.config = {}
    
    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.config = config
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("="*80)
        self.logger.info(json.dumps(config, indent=2))
        self.logger.info("="*80)
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for an epoch."""
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        
        self.metrics[epoch].update(metrics)
        
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:3d} | {metric_str}")
    
    def log_compliance(self, epoch: int, compliance_status: dict):
        """Log compliance check results."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"COMPLIANCE CHECK - Epoch {epoch}")
        self.logger.info(f"{'='*80}")
        for metric, status in compliance_status.items():
            symbol = "✓" if status else "✗"
            self.logger.info(f"  {symbol} {metric}")
        self.logger.info(f"{'='*80}\n")
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        output = {
            'experiment': self.experiment_name,
            'config': self.config,
            'metrics': self.metrics
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Saved metrics to {metrics_file}")
    
    def get_logger(self) -> logging.Logger:
        """Get underlying logger object."""
        return self.logger


def setup_logging(
    experiment_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO
) -> ExperimentLogger:
    """
    Setup logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured ExperimentLogger instance
    """
    return ExperimentLogger(experiment_name, log_dir, level)

