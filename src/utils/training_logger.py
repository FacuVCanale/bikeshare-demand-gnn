"""
Minimal training logger utilities for machine learning experiments.

This module provides clean, minimalist logging functions with essential
information display for training progress and metrics.
"""

import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np


class TrainingLogger:
    """
    Minimal training logger with clean formatting.
    
    Provides essential logging methods with minimal visual clutter
    while maintaining all important information.
    """
    
    def __init__(self, logger: logging.Logger, width: int = 100):
        """
        Initialize training logger.
        
        Args:
            logger: Logger instance to use for output
            width: Total width for formatting lines
        """
        self.logger = logger
        self.width = width
        
    def log_section_header(self, title: str, icon: str = "", line_char: str = "-"):
        """
        Log a section header with minimal formatting.
        
        Args:
            title: Section title
            icon: Unused for minimal design
            line_char: Character for separator lines
        """
        self.logger.info("")
        self.logger.info(f"{title}")
        self.logger.info(line_char * len(title))
        
    def log_subsection_header(self, title: str, icon: str = "", line_char: str = "-"):
        """
        Log a subsection header.
        
        Args:
            title: Subsection title
            icon: Unused for minimal design
            line_char: Character for separator lines
        """
        self.logger.info("")
        self.logger.info(f"{title}")
        
    def log_config_table(self, config_data: Dict[str, Any], title: str = "TRAINING CONFIGURATION"):
        """
        Log configuration data in a clean format.
        
        Args:
            config_data: Dictionary of configuration items
            title: Table title
        """
        self.log_section_header(title)
        
        max_key_length = max(len(str(key)) for key in config_data.keys()) + 2
        
        for key, value in config_data.items():
            formatted_key = f"{str(key):<{max_key_length}}"
            self.logger.info(f"  {formatted_key}: {value}")
            
    def log_training_header(self, metrics_names: List[str], has_validation: bool = True):
        """
        Create and log training progress header.
        
        Args:
            metrics_names: List of metric names to display
            has_validation: Whether to include validation columns
        """
        self.log_section_header("TRAINING STARTED")
        return 80  # dummy return for compatibility
        
    def log_epoch_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        status_info: Dict[str, Any] = None,
        epoch_time: float = 0.0,
        metrics_order: List[str] = None
    ):
        """
        Log training progress for a single epoch in minimal format.
        
        Args:
            epoch: Current epoch number (1-indexed)
            total_epochs: Total number of epochs
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary (optional)
            status_info: Status information (best model, learning rate, etc.)
            epoch_time: Time taken for this epoch
            metrics_order: Order of metrics to display
        """
        # build minimal log line
        log_parts = []
        
        # epoch
        log_parts.append(f"epoch {epoch:03d}")
        
        # train loss
        train_loss = train_metrics.get('loss', 0.0)
        log_parts.append(f"train_loss {train_loss:.4f}")
        
        # validation loss
        if val_metrics is not None:
            val_loss = val_metrics.get('loss', 0.0)
            log_parts.append(f"val_loss {val_loss:.4f}")
        
        # train r²
        train_r2 = train_metrics.get('r2', 0.0)
        log_parts.append(f"train_r² {train_r2:.3f}")
        
        # validation r²
        if val_metrics is not None:
            val_r2 = val_metrics.get('r2', 0.0)
            log_parts.append(f"val_r² {val_r2:.3f}")
        
        # best model indicator
        if status_info and status_info.get('is_best', False):
            log_parts.append("★ NEW BEST")
        
        # timing
        log_parts.append(f"{epoch_time:.1f}s")
        
        # format final line
        log_line = "│ " + " │ ".join(log_parts) + " │"
        
        self.logger.info(log_line)
        
    def log_detailed_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        separator_length: int = 100
    ):
        """
        Log detailed metrics - disabled for minimal logging.
        """
        # minimal logging - skip detailed metrics
        pass
            
    def log_training_summary(
        self,
        total_time: float,
        total_epochs: int,
        final_train_metrics: Dict[str, float],
        final_val_metrics: Optional[Dict[str, float]] = None,
        best_info: Optional[Dict[str, Any]] = None,
        saved_files: Optional[Dict[str, str]] = None
    ):
        """
        Log comprehensive training completion summary.
        
        Args:
            total_time: Total training time in seconds
            total_epochs: Total number of epochs completed
            final_train_metrics: Final training metrics
            final_val_metrics: Final validation metrics (optional)
            best_info: Information about best model
            saved_files: Dictionary of saved file descriptions and paths
        """
        avg_time_per_epoch = total_time / max(total_epochs, 1)
        
        self.log_section_header("TRAINING COMPLETED")
        
        # timing information
        self.logger.info(f"  Duration: {total_time/60:.1f} minutes ({total_epochs:,} epochs)")
        self.logger.info(f"  Average: {avg_time_per_epoch:.3f} seconds per epoch")
            
        # performance summary
        self.log_subsection_header("FINAL METRICS")
        
        # headers
        self.logger.info(f"{'Dataset':<12} {'Loss':<10} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'MAPE':<8}")
        self.logger.info("-" * 54)
        
        # train metrics
        train_line = f"{'Train':<12} {final_train_metrics.get('loss', 0):<10.4f} "
        train_line += f"{final_train_metrics.get('r2', 0):<8.3f} "
        train_line += f"{final_train_metrics.get('mae', 0):<8.3f} "
        train_line += f"{final_train_metrics.get('rmse', 0):<8.3f} "
        train_line += f"{final_train_metrics.get('mape', 0):<8.2f}"
        self.logger.info(train_line)
        
        # validation metrics
        if final_val_metrics:
            val_line = f"{'Validation':<12} {final_val_metrics.get('loss', 0):<10.4f} "
            val_line += f"{final_val_metrics.get('r2', 0):<8.3f} "
            val_line += f"{final_val_metrics.get('mae', 0):<8.3f} "
            val_line += f"{final_val_metrics.get('rmse', 0):<8.3f} "
            val_line += f"{final_val_metrics.get('mape', 0):<8.2f}"
            self.logger.info(val_line)
        
        # best model information
        if best_info:
            self.logger.info("")
            self.logger.info("BEST MODEL:")
            
            best_loss = best_info.get('best_loss')
            best_epoch = best_info.get('best_epoch')
            if best_loss is not None and best_epoch is not None:
                self.logger.info(f"  Best loss: {best_loss:.6f} (epoch {best_epoch})")
                
            final_loss = best_info.get('final_loss')
            if final_loss is not None and best_loss is not None and not np.isnan(final_loss):
                improvement = ((final_loss - best_loss) / abs(best_loss)) * 100
                if improvement > 0:
                    self.logger.info(f"  Final vs best: {improvement:.2f}% worse")
                else:
                    self.logger.info(f"  Final vs best: {abs(improvement):.2f}% better")
        
        # saved files
        if saved_files:
            self.log_subsection_header("SAVED FILES")
            
            for description, path in saved_files.items():
                self.logger.info(f"  {description}: {path}")
        
        self.logger.info("")
        
    def log_evaluation_header(self, eval_time: float, num_samples: int):
        """
        Log evaluation start information.
        
        Args:
            eval_time: Time taken for evaluation
            num_samples: Number of samples evaluated
        """
        self.log_section_header("MODEL EVALUATION")
        
        self.logger.info(f"  Evaluation time: {eval_time:.3f} seconds")
        self.logger.info(f"  Test samples: {num_samples:,}")
        
    def log_evaluation_results(
        self,
        metrics: Dict[str, float],
        with_assessment: bool = True
    ):
        """
        Log evaluation results in a clean format.
        
        Args:
            metrics: Dictionary of evaluation metrics
            with_assessment: Whether to include performance assessment
        """
        self.log_subsection_header("TEST RESULTS")
        
        # results table
        self.logger.info(f"{'Metric':<15} {'Value':<15} {'Description':<25}")
        self.logger.info("-" * 55)
        
        # prioritized metrics order
        metric_order = ['test_loss', 'loss', 'r2', 'mae', 'rmse', 'mape']
        displayed_metrics = []
        
        metric_descriptions = {
            'test_loss': 'Test loss (lower better)',
            'loss': 'Loss (lower better)',
            'r2': 'R² score (1.0 = perfect)',
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Square Error',
            'mape': 'Mean Abs Percent Error'
        }
        
        for metric in metric_order:
            if metric in metrics:
                description = metric_descriptions.get(metric, 'Custom metric')
                value = metrics[metric]
                
                if metric in ['test_loss', 'loss']:
                    value_str = f"{value:.6f}"
                elif metric == 'mape':
                    value_str = f"{value:.2f}%"
                else:
                    value_str = f"{value:.4f}"
                
                metric_line = f"{metric.upper():<15} {value_str:<15} {description:<25}"
                self.logger.info(metric_line)
                displayed_metrics.append(metric)
        
        # remaining metrics
        for metric, value in metrics.items():
            if metric not in displayed_metrics:
                value_str = f"{value:.4f}"
                metric_line = f"{metric.upper():<15} {value_str:<15} {'Custom metric':<25}"
                self.logger.info(metric_line)
        
        # performance assessment
        if with_assessment:
            self.log_performance_assessment(metrics)
            
        self.logger.info("")
        
    def log_performance_assessment(self, metrics: Dict[str, float]):
        """
        Log performance assessment based on metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        self.logger.info("")
        self.logger.info("PERFORMANCE ASSESSMENT:")
        
        # r² assessment
        r2_score = metrics.get('r2', 0)
        if r2_score >= 0.9:
            performance = "Excellent"
        elif r2_score >= 0.8:
            performance = "Good" 
        elif r2_score >= 0.7:
            performance = "Fair"
        elif r2_score >= 0.5:
            performance = "Poor"
        else:
            performance = "Very Poor"
        
        self.logger.info(f"  Overall: {performance} (R² = {r2_score:.3f})")
        
        # mape assessment
        mape_score = metrics.get('mape', float('inf'))
        if mape_score <= 5:
            mape_assessment = "Very accurate"
        elif mape_score <= 10:
            mape_assessment = "Accurate"
        elif mape_score <= 20:
            mape_assessment = "Moderate"
        elif mape_score <= 50:
            mape_assessment = "Poor accuracy"
        else:
            mape_assessment = "Very poor accuracy"
        
        self.logger.info(f"  Accuracy: {mape_assessment} (MAPE = {mape_score:.1f}%)")


def create_training_logger(logger: logging.Logger, width: int = 100) -> TrainingLogger:
    """
    Factory function to create a TrainingLogger instance.
    
    Args:
        logger: Logger instance to use
        width: Formatting width
        
    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(logger, width)


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_eta(elapsed_time: float, current_step: int, total_steps: int) -> float:
    """
    Calculate estimated time of arrival.
    
    Args:
        elapsed_time: Time elapsed so far
        current_step: Current step number
        total_steps: Total number of steps
        
    Returns:
        Estimated remaining time in seconds
    """
    if current_step == 0:
        return 0.0
    
    avg_time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    return avg_time_per_step * remaining_steps 