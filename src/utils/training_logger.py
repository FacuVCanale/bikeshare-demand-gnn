"""
Professional training logger utilities for machine learning experiments.

This module provides reusable logging functions with consistent formatting
for training progress, metrics display, and performance assessment.
"""

import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np


class TrainingLogger:
    """
    Professional training logger with standardized formatting.
    
    Provides reusable methods for consistent logging across different
    training scenarios with customizable formatting options.
    """
    
    def __init__(self, logger: logging.Logger, width: int = 120):
        """
        Initialize training logger.
        
        Args:
            logger: Logger instance to use for output
            width: Total width for formatting lines
        """
        self.logger = logger
        self.width = width
        
    def log_section_header(self, title: str, icon: str = "🔸", line_char: str = "="):
        """
        Log a major section header with consistent formatting.
        
        Args:
            title: Section title
            icon: Icon to display with title
            line_char: Character for separator lines
        """
        self.logger.info(line_char * self.width)
        self.logger.info(f"{icon} {title}")
        self.logger.info(line_char * self.width)
        
    def log_subsection_header(self, title: str, icon: str = "📋", line_char: str = "─"):
        """
        Log a subsection header.
        
        Args:
            title: Subsection title
            icon: Icon to display with title  
            line_char: Character for separator lines
        """
        self.logger.info(line_char * self.width)
        self.logger.info(f"{icon} {title}")
        self.logger.info(line_char * self.width)
        
    def log_config_table(self, config_data: Dict[str, Any], title: str = "CONFIGURATION"):
        """
        Log configuration data in a formatted table.
        
        Args:
            config_data: Dictionary of configuration items
            title: Table title
        """
        self.log_section_header(title, "🚀")
        
        max_key_length = max(len(str(key)) for key in config_data.keys()) + 2
        
        for key, value in config_data.items():
            formatted_key = f"{str(key):<{max_key_length}}"
            self.logger.info(f"{formatted_key}: {value}")
            
    def log_training_header(self, metrics_names: List[str], has_validation: bool = True):
        """
        Create and log training progress table header.
        
        Args:
            metrics_names: List of metric names to display
            has_validation: Whether to include validation columns
        """
        self.log_section_header("TRAINING PROGRESS", "🎯")
        
        # create main header
        header_parts = ["EPOCH", "PROGRESS"]
        header_parts.append("TRAIN")
        if has_validation:
            header_parts.append("VALIDATION") 
        header_parts.extend(["STATUS", "TIME"])
        
        # calculate column widths
        epoch_width = 8
        progress_width = 12
        metrics_width = len(metrics_names) * 8 + 4  # space for each metric plus padding
        status_width = 25
        time_width = 8
        
        header = f"{'EPOCH':>{epoch_width}} │ {'PROGRESS':>{progress_width}} │ "
        header += f"{'TRAIN':>{metrics_width}} │"
        if has_validation:
            header += f" {'VALIDATION':>{metrics_width}} │"
        header += f" {'STATUS':>{status_width}} │ {'TIME':>{time_width}}"
        
        self.logger.info(header)
        self.logger.info("─" * len(header))
        
        # create subheader with metric names
        subheader = f"{'':>{epoch_width}} │ {'[████████]':>{progress_width}} │ "
        metrics_str = " ".join(f"{name:>7}" for name in metrics_names)
        subheader += f"{metrics_str:>{metrics_width}} │"
        if has_validation:
            subheader += f" {metrics_str:>{metrics_width}} │"
        subheader += f" {'Learning Rate':>{status_width}} │ {'(s)':>{time_width}}"
        
        self.logger.info(subheader)
        self.logger.info("─" * len(header))
        
        return len(header)  # return header length for consistent separators
        
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
        Log training progress for a single epoch.
        
        Args:
            epoch: Current epoch number (1-indexed)
            total_epochs: Total number of epochs
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary (optional)
            status_info: Status information (best model, learning rate, etc.)
            epoch_time: Time taken for this epoch
            metrics_order: Order of metrics to display
        """
        if metrics_order is None:
            metrics_order = ['loss', 'r2', 'mae']
            
        # format progress bar
        progress = epoch / total_epochs
        bar_length = 8
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        progress_str = f"[{bar}] {progress*100:4.0f}%"
        
        # format train metrics
        train_values = []
        for metric in metrics_order:
            value = train_metrics.get(metric, 0.0)
            train_values.append(f"{value:7.3f}")
        
        # format validation metrics
        val_values = []
        if val_metrics is not None:
            for metric in metrics_order:
                value = val_metrics.get(metric, 0.0)
                val_values.append(f"{value:7.3f}")
        
        # format status
        status_str = ""
        if status_info:
            if status_info.get('is_best', False):
                improvement = status_info.get('improvement', 0.0)
                status_str = f"🏆 NEW BEST (↓{improvement:.4f})"
            elif status_info.get('epochs_since_best', -1) >= 0:
                epochs_since = status_info.get('epochs_since_best')
                status_str = f"⏱️  {epochs_since:2d} epochs since best"
            else:
                lr = status_info.get('learning_rate', 0.0)
                status_str = f"📈 LR: {lr:.1e}"
        
        # build log line
        log_line = f"{epoch:8d} │ {progress_str:>12} │ "
        log_line += f"{' '.join(train_values):>32} │"
        
        if val_metrics is not None:
            log_line += f" {' '.join(val_values):>32} │"
            
        log_line += f" {status_str:>25} │ {epoch_time:8.2f}"
        
        self.logger.info(log_line)
        
    def log_detailed_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        separator_length: int = 120
    ):
        """
        Log detailed metrics in a readable format.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            additional_info: Additional information (LR, ETA, etc.)
            separator_length: Length of separator line
        """
        details = []
        
        # detailed train metrics
        train_details = []
        for metric, value in train_metrics.items():
            if metric not in ['loss', 'r2', 'mae']:  # skip already shown metrics
                if metric == 'mape':
                    train_details.append(f"{metric.upper()}: {value:.2f}%")
                else:
                    train_details.append(f"{metric.upper()}: {value:.4f}")
        
        if train_details:
            details.append(f"TRAIN → {', '.join(train_details)}")
        
        # detailed validation metrics
        if val_metrics:
            val_details = []
            for metric, value in val_metrics.items():
                if metric not in ['loss', 'r2', 'mae']:
                    if metric == 'mape':
                        val_details.append(f"{metric.upper()}: {value:.2f}%")
                    else:
                        val_details.append(f"{metric.upper()}: {value:.4f}")
            
            if val_details:
                details.append(f"VAL → {', '.join(val_details)}")
        
        # additional information
        if additional_info:
            add_details = []
            for key, value in additional_info.items():
                if key == 'learning_rate':
                    add_details.append(f"LR: {value:.1e}")
                elif key == 'eta':
                    eta_str = f"{value/60:.0f}m" if value > 60 else f"{value:.0f}s"
                    add_details.append(f"ETA: {eta_str}")
                else:
                    add_details.append(f"{key}: {value}")
            
            if add_details:
                details.append(", ".join(add_details))
        
        # log details
        for detail in details:
            self.logger.info(f"{'':>8} │ {'':>12} │ {detail:<65} │ {'':>25} │ {'':>8}")
        
        if details:
            self.logger.info("─" * separator_length)
            
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
        
        self.log_section_header("TRAINING COMPLETED", "✅")
        
        # timing information
        timing_config = {
            "Total duration": f"{total_time/60:.1f} minutes ({total_time:.1f} seconds)",
            "Total epochs": f"{total_epochs:,}",
            "Avg time per epoch": f"{avg_time_per_epoch:.3f}s"
        }
        
        for key, value in timing_config.items():
            self.logger.info(f"{key:<25}: {value}")
            
        # performance summary table
        self.log_subsection_header("FINAL PERFORMANCE SUMMARY", "📊")
        
        metrics_header = f"{'DATASET':<12} │ {'LOSS':<12} │ {'R²':<12} │ {'MAE':<12} │ {'RMSE':<12} │ {'MAPE':<12}"
        self.logger.info(metrics_header)
        self.logger.info("─" * len(metrics_header))
        
        # train metrics row
        train_line = f"{'TRAIN':<12} │ {final_train_metrics.get('loss', 0):<12.4f} │ "
        train_line += f"{final_train_metrics.get('r2', 0):<12.3f} │ "
        train_line += f"{final_train_metrics.get('mae', 0):<12.4f} │ "
        train_line += f"{final_train_metrics.get('rmse', 0):<12.4f} │ "
        train_line += f"{final_train_metrics.get('mape', 0):<12.2f}"
        self.logger.info(train_line)
        
        # validation metrics row
        if final_val_metrics:
            val_line = f"{'VALIDATION':<12} │ {final_val_metrics.get('loss', 0):<12.4f} │ "
            val_line += f"{final_val_metrics.get('r2', 0):<12.3f} │ "
            val_line += f"{final_val_metrics.get('mae', 0):<12.4f} │ "
            val_line += f"{final_val_metrics.get('rmse', 0):<12.4f} │ "
            val_line += f"{final_val_metrics.get('mape', 0):<12.2f}"
            self.logger.info(val_line)
        
        # best model information
        if best_info:
            self.logger.info("─" * len(metrics_header))
            self.logger.info(f"{'🏆 BEST MODEL':<30}")
            
            best_loss = best_info.get('best_loss')
            best_epoch = best_info.get('best_epoch')
            if best_loss is not None and best_epoch is not None:
                self.logger.info(f"{'Best validation loss':<25}: {best_loss:.6f} (epoch {best_epoch})")
                
            final_loss = best_info.get('final_loss')
            if final_loss is not None and best_loss is not None and not np.isnan(final_loss):
                improvement = ((final_loss - best_loss) / abs(best_loss)) * 100
                if improvement > 0:
                    self.logger.info(f"{'Final vs best':<25}: {improvement:.2f}% worse")
                else:
                    self.logger.info(f"{'Final vs best':<25}: {abs(improvement):.2f}% better")
        
        # saved files information
        if saved_files:
            self.log_subsection_header("SAVING MODEL FILES", "💾")
            
            for description, path in saved_files.items():
                self.logger.info(f"{description:<25}: saved to {path}")
                
        self.log_section_header("", "", "=")  # final separator
        
    def log_evaluation_header(self, eval_time: float, num_samples: int):
        """
        Log evaluation start with basic information.
        
        Args:
            eval_time: Time taken for evaluation
            num_samples: Number of samples evaluated
        """
        self.log_section_header("EVALUATING MODEL ON TEST DATA", "🧪")
        
        self.logger.info(f"{'Evaluation time':<25}: {eval_time:.3f} seconds")
        self.logger.info(f"{'Test samples':<25}: {num_samples:,}")
        
    def log_evaluation_results(
        self,
        metrics: Dict[str, float],
        with_assessment: bool = True
    ):
        """
        Log evaluation results in a professional format.
        
        Args:
            metrics: Dictionary of evaluation metrics
            with_assessment: Whether to include performance assessment
        """
        self.log_subsection_header("TEST RESULTS", "📈")
        
        # metric interpretations
        metric_interpretations = {
            'test_loss': 'Lower is better',
            'loss': 'Lower is better',
            'r2': 'Coefficient of determination (1.0 = perfect)',
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Square Error',
            'mape': 'Mean Absolute Percentage Error (%)'
        }
        
        # results table
        results_header = f"{'METRIC':<15} │ {'VALUE':<15} │ {'INTERPRETATION':<30}"
        self.logger.info(results_header)
        self.logger.info("─" * len(results_header))
        
        # prioritized metrics order
        metric_order = ['test_loss', 'loss', 'r2', 'mae', 'rmse', 'mape']
        displayed_metrics = []
        
        for metric in metric_order:
            if metric in metrics:
                interpretation = metric_interpretations.get(metric, 'Custom metric')
                value = metrics[metric]
                
                if metric in ['test_loss', 'loss']:
                    value_str = f"{value:.6f}"
                elif metric == 'mape':
                    value_str = f"{value:.4f}%"
                else:
                    value_str = f"{value:.4f}"
                
                metric_line = f"{metric.upper():<15} │ {value_str:<15} │ {interpretation:<30}"
                self.logger.info(metric_line)
                displayed_metrics.append(metric)
        
        # remaining metrics
        for metric, value in metrics.items():
            if metric not in displayed_metrics:
                value_str = f"{value:.4f}"
                metric_line = f"{metric.upper():<15} │ {value_str:<15} │ {'Custom metric':<30}"
                self.logger.info(metric_line)
        
        # performance assessment
        if with_assessment:
            self.log_performance_assessment(metrics)
            
        self.log_section_header("", "", "=")  # final separator
        
    def log_performance_assessment(self, metrics: Dict[str, float]):
        """
        Log automatic performance assessment based on metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        self.logger.info("─" * 62)  # length of results header
        self.logger.info(f"{'🎯 PERFORMANCE ASSESSMENT':<30}")
        
        # r² assessment
        r2_score = metrics.get('r2', 0)
        if r2_score >= 0.9:
            performance = "🟢 Excellent"
        elif r2_score >= 0.8:
            performance = "🟡 Good" 
        elif r2_score >= 0.7:
            performance = "🟠 Fair"
        elif r2_score >= 0.5:
            performance = "🔴 Poor"
        else:
            performance = "⚫ Very Poor"
        
        self.logger.info(f"{'Overall performance':<25}: {performance} (R² = {r2_score:.3f})")
        
        # mape assessment
        mape_score = metrics.get('mape', float('inf'))
        if mape_score <= 5:
            mape_assessment = "🟢 Very accurate"
        elif mape_score <= 10:
            mape_assessment = "🟡 Accurate"
        elif mape_score <= 20:
            mape_assessment = "🟠 Moderate"
        elif mape_score <= 50:
            mape_assessment = "🔴 Poor accuracy"
        else:
            mape_assessment = "⚫ Very poor accuracy"
        
        self.logger.info(f"{'Prediction accuracy':<25}: {mape_assessment} (MAPE = {mape_score:.1f}%)")


def create_training_logger(logger: logging.Logger, width: int = 120) -> TrainingLogger:
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