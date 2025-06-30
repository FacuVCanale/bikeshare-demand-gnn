"""
Temporal Graph Neural Network Trainer.

This module provides training functionality for temporal GNN models that process
sequential graph data with dynamic adjacency matrices, specifically designed
for the Gated GCN model and bike demand prediction.

Key features:
- Handles temporal sequences with dynamic graphs
- Prevents data leakage through proper temporal ordering
- Supports various loss functions and optimization strategies
- Integrates with existing GNN training infrastructure
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from tqdm import tqdm
import json
import warnings

# import dataset and model components
from ..dataset.temporal_graph_dataset import TemporalGraphDataset, TemporalGraphData
from ..models.gated_gcn import GateGCN
from .gnn_trainer import GNNTrainer


class TemporalGNNTrainer:
    """
    Trainer for temporal Graph Neural Networks with dynamic adjacency matrices.
    
    This trainer handles the complexity of temporal graph sequences while ensuring
    no data leakage occurs during training and evaluation.
    """
    
    def __init__(
        self,
        model: GateGCN,
        device: str = 'auto',
        save_dir: str = 'experiments/temporal_gnn',
        logging_level: str = 'INFO'
    ):
        """
        Initialize temporal GNN trainer.
        
        Args:
            model: Gated GCN model for temporal prediction
            device: Device to use ('cuda', 'cpu', or 'auto')
            save_dir: Directory to save models and logs
            logging_level: Logging level
        """
        # setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # setup model
        self.model = model.to(self.device)
        
        # setup directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # setup logging
        logging.basicConfig(level=getattr(logging, logging_level))
        self.logger = logging.getLogger(__name__)
        
        # training state
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        self.logger.info(f"Temporal GNN Trainer initialized on {self.device}")
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Save directory: {self.save_dir}")
    
    def setup_training(
        self,
        optimizer_name: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_name: str = 'plateau',
        loss_function: str = 'mse',
        loss_params: Optional[Dict] = None
    ):
        """
        Setup training components (optimizer, scheduler, loss function).
        
        Args:
            optimizer_name: Optimizer type ('adam', 'adamw', 'sgd')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler_name: Learning rate scheduler ('plateau', 'cosine', 'none')
            loss_function: Loss function name ('mse', 'mae', 'huber')
            loss_params: Additional loss function parameters
        """
        # setup optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # setup scheduler
        if scheduler_name.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif scheduler_name.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,
                eta_min=1e-6
            )
        elif scheduler_name.lower() == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        # setup loss function
        loss_params = loss_params or {}
        
        if loss_function.lower() == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_function.lower() == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_function.lower() == 'huber':
            self.loss_fn = nn.HuberLoss(delta=loss_params.get('delta', 1.0))
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        self.logger.info(f"Training setup complete:")
        self.logger.info(f"  Optimizer: {optimizer_name} (lr={learning_rate}, wd={weight_decay})")
        self.logger.info(f"  Scheduler: {scheduler_name}")
        self.logger.info(f"  Loss function: {loss_function}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_frequency: int = 10
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training sequences
            epoch: Current epoch number
            log_frequency: How often to log progress
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch} Training",
            leave=False
        )
        
        for batch_idx, temporal_data in enumerate(progress_bar):
            # NEW: support batch_size>1 where temporal_data is a list of sequences
            if isinstance(temporal_data, list):
                seq_list = temporal_data
            else:
                seq_list = [temporal_data]

            batch_loss = 0.0
            for seq in seq_list:
                # move data to device
                seq = seq.to(self.device)

                # zero gradients (only once per optimiser step)
                self.optimizer.zero_grad()

                predictions = self.model(seq.x, seq.edge_index)
                loss = self.loss_fn(predictions, seq.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss += loss.item()

            # use average loss for logging when batch>1
            loss_value = batch_loss / len(seq_list)

            total_loss += loss_value
            num_batches += 1

            progress_bar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: DataLoader for validation sequences
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for temporal_data in tqdm(val_loader, desc="Validation", leave=False):
                if isinstance(temporal_data, list):
                    seq_list = temporal_data
                else:
                    seq_list = [temporal_data]

                for seq in seq_list:
                    seq = seq.to(self.device)
                    try:
                        predictions = self.model(seq.x, seq.edge_index)
                        loss = self.loss_fn(predictions, seq.y)
                        total_loss += loss.item()
                        num_batches += 1
                        all_predictions.append(predictions.cpu())
                        all_targets.append(seq.y.cpu())
                    except Exception as e:
                        self.logger.warning(f"Error in validation batch: {str(e)}")
                        continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # compute additional metrics
        if all_predictions and all_targets:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            metrics = self._compute_metrics(all_predictions, all_targets)
        else:
            metrics = {}
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        with torch.no_grad():
            # mean squared error
            mse = torch.mean((predictions - targets) ** 2).item()
            
            # mean absolute error
            mae = torch.mean(torch.abs(predictions - targets)).item()
            
            # root mean squared error
            rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
            
            # r-squared
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
    
    def fit(
        self,
        train_dataset: TemporalGraphDataset,
        val_dataset: TemporalGraphDataset,
        epochs: int = 100,
        batch_size: int = 1,  # typically 1 for temporal sequences
        early_stopping_patience: int = 15,
        save_best_model: bool = True,
        log_frequency: int = 10,
        min_improvement: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train the temporal GNN model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size (typically 1 for temporal data)
            early_stopping_patience: Patience for early stopping
            save_best_model: Whether to save the best model
            log_frequency: How often to log progress
            min_improvement: Minimum improvement for early stopping
            
        Returns:
            Training history and results
        """
        # create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_temporal_batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_temporal_batch
        )
        
        self.logger.info(f"Training temporal GNN for {epochs} epochs")
        self.logger.info(f"Train sequences: {len(train_dataset)}")
        self.logger.info(f"Val sequences: {len(val_dataset)}")
        
        # training loop
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # train epoch
            train_loss = self.train_epoch(train_loader, epoch, log_frequency)
            self.train_losses.append(train_loss)
            
            # validate epoch
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # logging
            if epoch % log_frequency == 0:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                self.logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"{metrics_str}"
                )
            
            # check for improvement
            if val_loss < self.best_val_loss - min_improvement:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if save_best_model:
                    self.best_model_state = self.model.state_dict().copy()
                    self._save_checkpoint(epoch, val_loss, val_metrics)
            else:
                patience_counter += 1
            
            # early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping after {epoch} epochs")
                break
        
        # restore best model
        if save_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_val_metrics': val_metrics,
            'epochs_trained': epoch
        }
    
    def _collate_temporal_batch(self, batch: List[TemporalGraphData]) -> Union[TemporalGraphData, List[TemporalGraphData]]:
        """
        Collate function now supports real batching.

        If ``batch_size`` is 1 the historical behaviour is preserved and the
        unique ``TemporalGraphData`` element is returned directly. For larger
        batch sizes we simply return the list of ``TemporalGraphData`` objects
        so that the training loop can iterate over each sequence.  This keeps
        the memory representation compact and avoids complex padding logic.
        A future optimisation could stack sequences with the same number of
        nodes into a single tensor, but for most GNN workloads iterating over
        a handful of sequences per optimisation step already offers a
        significant speed-up compared to processing them one-by-one at the
        DataLoader level.
        """
        if len(batch) == 1:
            # backward-compatibility path
            return batch[0]
        # for real batching just return the list so the outer loop can handle it
        return batch
    
    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = self.save_dir / 'best_model.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # save metrics separately for easy reading
        metrics_path = self.save_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'val_loss': val_loss,
                'metrics': metrics,
                'train_losses': self.train_losses[-10:],  # last 10 for brevity
                'val_losses': self.val_losses[-10:]
            }, f, indent=2)
    
    def predict(self, test_dataset: TemporalGraphDataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._collate_temporal_batch
        )
        
        with torch.no_grad():
            for temporal_data in tqdm(test_loader, desc="Predicting"):
                temporal_data = temporal_data.to(self.device)
                
                try:
                    predictions = self.model(temporal_data.x, temporal_data.edge_index)
                    
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(temporal_data.y.cpu().numpy())
                    
                except Exception as e:
                    self.logger.warning(f"Error in prediction: {str(e)}")
                    continue
        
        if all_predictions and all_targets:
            return np.concatenate(all_predictions), np.concatenate(all_targets)
        else:
            return np.array([]), np.array([])


def train_temporal_gnn_experiment(
    train_dataset: TemporalGraphDataset,
    val_dataset: TemporalGraphDataset,
    test_dataset: TemporalGraphDataset,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    experiment_name: str = "temporal_gnn_experiment"
) -> Dict[str, Any]:
    """
    Run a complete temporal GNN training experiment.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model_config: Model configuration parameters
        training_config: Training configuration parameters
        experiment_name: Name for the experiment
        
    Returns:
        Complete experiment results
    """
    # get dataset info
    train_info = train_dataset.get_data_info()
    
    # create model
    model = GateGCN(
        num_features=train_info['num_features'],
        num_targets=train_info['num_targets'],
        **model_config
    )
    
    # create trainer
    trainer = TemporalGNNTrainer(
        model=model,
        save_dir=f"experiments/temporal_gnn/{experiment_name}"
    )
    
    # setup training
    trainer.setup_training(**training_config)
    
    # train model
    training_history = trainer.fit(train_dataset, val_dataset)
    
    # evaluate on test set
    test_predictions, test_targets = trainer.predict(test_dataset)
    
    if len(test_predictions) > 0:
        test_metrics = trainer._compute_metrics(
            torch.tensor(test_predictions),
            torch.tensor(test_targets)
        )
    else:
        test_metrics = {}
    
    return {
        'model': model,
        'trainer': trainer,
        'training_history': training_history,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'test_metrics': test_metrics,
        'experiment_name': experiment_name,
        'config': {
            'model_config': model_config,
            'training_config': training_config
        }
    } 