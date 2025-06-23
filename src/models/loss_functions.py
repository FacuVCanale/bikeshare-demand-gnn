"""
Custom Loss Functions for handling sparse/zero-inflated bike demand data.

This module implements various loss functions designed to handle the challenges
of bike demand prediction where many time periods have zero arrivals/departures.
Includes Zero-Inflated Loss, Weighted MSE, Focal Loss for Regression, and others.

Author: EcoBici-AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class ZeroInflatedLoss(nn.Module):
    """
    Zero-Inflated Loss for handling data with excess zeros.
    
    This loss function models two separate processes:
    1. Binary classification: Is the value zero or non-zero?
    2. Regression: What is the actual value when non-zero?
    
    Perfect for bike demand where many periods have zero activity.
    """
    
    def __init__(
        self, 
        zero_weight: float = 1.0,
        nonzero_weight: float = 2.0,
        regression_weight: float = 1.0,
        zero_threshold: float = 1e-6,
        use_log_transform: bool = True
    ):
        """
        Initialize Zero-Inflated Loss.
        
        Args:
            zero_weight: Weight for zero predictions
            nonzero_weight: Weight for non-zero predictions (higher = focus more on activity)
            regression_weight: Weight for regression component vs classification component
            zero_threshold: Values below this are considered "zero"
            use_log_transform: Apply log(1+x) transform for regression part
        """
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.regression_weight = regression_weight
        self.zero_threshold = zero_threshold
        self.use_log_transform = use_log_transform
        
        # loss components
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute zero-inflated loss.
        
        Args:
            predictions: Model predictions [batch_size, num_targets]
            targets: Ground truth targets [batch_size, num_targets]
            
        Returns:
            Combined loss value
        """
        # create binary masks for zero/non-zero targets
        is_zero = (targets.abs() <= self.zero_threshold).float()
        is_nonzero = 1.0 - is_zero
        
        # 1. Binary classification component: predict zero vs non-zero
        # convert predictions to "probability of being non-zero"
        pred_nonzero_logits = torch.log(torch.abs(predictions) + 1e-8)  # log of absolute predicted value
        binary_targets = is_nonzero  # 1 if non-zero, 0 if zero
        
        # weighted binary cross-entropy
        pos_weight = torch.tensor(self.nonzero_weight / self.zero_weight, device=predictions.device)
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        classification_loss = bce_loss_fn(pred_nonzero_logits, binary_targets)
        
        # 2. Regression component: predict actual values for non-zero targets
        if is_nonzero.sum() > 0:  # only if we have non-zero targets
            # filter to non-zero samples
            nonzero_mask = is_nonzero.bool()
            nonzero_preds = predictions[nonzero_mask]
            nonzero_targets = targets[nonzero_mask]
            
            if self.use_log_transform:
                # apply log(1+x) transform for better handling of skewed distributions
                log_preds = torch.log1p(torch.abs(nonzero_preds))
                log_targets = torch.log1p(torch.abs(nonzero_targets))
                regression_loss = self.mse_loss(log_preds, log_targets)
            else:
                regression_loss = self.mse_loss(nonzero_preds, nonzero_targets)
        else:
            regression_loss = torch.tensor(0.0, device=predictions.device)
        
        # 3. Combine both components
        total_loss = classification_loss + self.regression_weight * regression_loss
        
        return total_loss


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss that gives higher weight to non-zero targets.
    
    Simple but effective approach for handling zero-inflated data.
    """
    
    def __init__(
        self, 
        zero_weight: float = 0.5,
        nonzero_weight: float = 2.0,
        zero_threshold: float = 1e-6
    ):
        """
        Initialize Weighted MSE Loss.
        
        Args:
            zero_weight: Weight for zero targets (lower = less focus on zeros)
            nonzero_weight: Weight for non-zero targets (higher = more focus on activity)
            zero_threshold: Values below this are considered "zero"
        """
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.zero_threshold = zero_threshold
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss."""
        # create binary mask for zero/non-zero targets
        is_zero = (targets.abs() <= self.zero_threshold).float()
        is_nonzero = 1.0 - is_zero
        
        # compute element-wise squared errors
        squared_errors = (predictions - targets) ** 2
        
        # apply weights
        weights = is_zero * self.zero_weight + is_nonzero * self.nonzero_weight
        weighted_errors = squared_errors * weights
        
        # return mean weighted error
        return weighted_errors.mean()


class FocalRegressionLoss(nn.Module):
    """
    Focal Loss adapted for regression tasks.
    
    Focuses learning on "hard" examples by down-weighting easy predictions.
    Good for imbalanced regression where you want to focus on difficult cases.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Regression Loss.
        
        Args:
            alpha: Scaling factor
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal regression loss."""
        # compute base MSE loss
        mse_loss = (predictions - targets) ** 2
        
        # compute normalized error (0 to 1 scale)
        # normalize by target magnitude to make focal weight meaningful
        target_scale = torch.abs(targets) + 1.0  # add 1 to avoid division by zero
        normalized_error = torch.sqrt(mse_loss) / target_scale
        normalized_error = torch.clamp(normalized_error, 0.0, 1.0)
        
        # compute focal weight: (1 - p_t)^gamma where p_t is "correctness"
        correctness = 1.0 - normalized_error
        focal_weight = (1.0 - correctness) ** self.gamma
        
        # apply focal weighting
        focal_loss = self.alpha * focal_weight * mse_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class RobustL1Loss(nn.Module):
    """
    Robust L1 Loss with higher weight for non-zero targets.
    
    Less sensitive to outliers than MSE, good for noisy bike demand data.
    """
    
    def __init__(
        self, 
        zero_weight: float = 0.3,
        nonzero_weight: float = 1.5,
        zero_threshold: float = 1e-6
    ):
        """Initialize Robust L1 Loss."""
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.zero_threshold = zero_threshold
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted L1 loss."""
        # create binary mask for zero/non-zero targets
        is_zero = (targets.abs() <= self.zero_threshold).float()
        is_nonzero = 1.0 - is_zero
        
        # compute element-wise absolute errors
        abs_errors = torch.abs(predictions - targets)
        
        # apply weights
        weights = is_zero * self.zero_weight + is_nonzero * self.nonzero_weight
        weighted_errors = abs_errors * weights
        
        return weighted_errors.mean()


class AdaptiveZeroInflatedLoss(nn.Module):
    """
    Adaptive Zero-Inflated Loss that automatically adjusts weights based on data distribution.
    
    Dynamically adapts the balance between zero and non-zero examples based on
    the proportion of zeros in each batch.
    """
    
    def __init__(
        self, 
        base_nonzero_weight: float = 2.0,
        adaptive_factor: float = 0.5,
        zero_threshold: float = 1e-6,
        use_log_transform: bool = True
    ):
        """
        Initialize Adaptive Zero-Inflated Loss.
        
        Args:
            base_nonzero_weight: Base weight for non-zero targets
            adaptive_factor: How much to adapt based on zero proportion (0-1)
            zero_threshold: Values below this are considered "zero"
            use_log_transform: Apply log transform for regression component
        """
        super().__init__()
        self.base_nonzero_weight = base_nonzero_weight
        self.adaptive_factor = adaptive_factor
        self.zero_threshold = zero_threshold
        self.use_log_transform = use_log_transform
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptive zero-inflated loss."""
        # calculate proportion of zeros in this batch
        is_zero = (targets.abs() <= self.zero_threshold).float()
        zero_proportion = is_zero.mean().item()
        
        # adapt weights based on zero proportion
        # more zeros -> higher weight for non-zeros
        adaptive_nonzero_weight = self.base_nonzero_weight * (1.0 + self.adaptive_factor * zero_proportion)
        adaptive_zero_weight = 1.0 / (1.0 + self.adaptive_factor * zero_proportion)
        
        # use weighted MSE with adaptive weights
        is_nonzero = 1.0 - is_zero
        squared_errors = (predictions - targets) ** 2
        
        if self.use_log_transform:
            # apply log transform for better handling of skewed data
            log_preds = torch.sign(predictions) * torch.log1p(torch.abs(predictions))
            log_targets = torch.sign(targets) * torch.log1p(torch.abs(targets))
            squared_errors = (log_preds - log_targets) ** 2
        
        # apply adaptive weights
        weights = is_zero * adaptive_zero_weight + is_nonzero * adaptive_nonzero_weight
        weighted_errors = squared_errors * weights
        
        return weighted_errors.mean()


def create_loss_function(
    loss_type: str,
    **kwargs
) -> nn.Module:
    """
    Factory function to create custom loss functions.
    
    Args:
        loss_type: Type of loss function to create
        **kwargs: Additional parameters for the loss function
        
    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'zero_inflated':
        return ZeroInflatedLoss(**kwargs)
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(**kwargs)
    elif loss_type == 'focal_regression':
        return FocalRegressionLoss(**kwargs)
    elif loss_type == 'robust_l1':
        return RobustL1Loss(**kwargs)
    elif loss_type == 'adaptive_zero_inflated':
        return AdaptiveZeroInflatedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def analyze_target_distribution(targets: torch.Tensor, zero_threshold: float = 1e-6) -> dict:
    """
    Analyze the distribution of target values to help choose appropriate loss function.
    
    Args:
        targets: Target tensor to analyze
        zero_threshold: Threshold for considering values as zero
        
    Returns:
        Dictionary with distribution statistics
    """
    targets_np = targets.detach().cpu().numpy().flatten()
    
    # basic statistics
    is_zero = np.abs(targets_np) <= zero_threshold
    zero_count = np.sum(is_zero)
    nonzero_count = len(targets_np) - zero_count
    zero_proportion = zero_count / len(targets_np)
    
    # statistics for non-zero values
    if nonzero_count > 0:
        nonzero_values = targets_np[~is_zero]
        nonzero_mean = np.mean(nonzero_values)
        nonzero_std = np.std(nonzero_values)
        nonzero_min = np.min(nonzero_values)
        nonzero_max = np.max(nonzero_values)
    else:
        nonzero_mean = nonzero_std = nonzero_min = nonzero_max = 0.0
    
    stats = {
        'total_samples': len(targets_np),
        'zero_count': zero_count,
        'nonzero_count': nonzero_count,
        'zero_proportion': zero_proportion,
        'nonzero_mean': nonzero_mean,
        'nonzero_std': nonzero_std,
        'nonzero_min': nonzero_min,
        'nonzero_max': nonzero_max,
        'overall_mean': np.mean(targets_np),
        'overall_std': np.std(targets_np)
    }
    
    # recommendations
    if zero_proportion > 0.7:
        stats['recommendation'] = 'zero_inflated'
        stats['reason'] = f"High zero proportion ({zero_proportion:.1%}) - use Zero-Inflated Loss"
    elif zero_proportion > 0.4:
        stats['recommendation'] = 'weighted_mse'
        stats['reason'] = f"Moderate zero proportion ({zero_proportion:.1%}) - use Weighted MSE"
    else:
        stats['recommendation'] = 'focal_regression'
        stats['reason'] = f"Low zero proportion ({zero_proportion:.1%}) - use Focal Regression"
    
    return stats 