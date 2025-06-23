"""
Training module for Graph Neural Network models in EcoBici demand prediction.

This module provides comprehensive training functionality for GNN models including
training loops, validation, early stopping, model checkpointing, and experiment tracking.
Supports both full-batch training (for small graphs) and mini-batch training with 
neighbor sampling (for large graphs).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, DataLoader as GeometricDataLoader
from torch_geometric.utils import trim_to_layer

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime

from src.models.gnn_models import (
    TemporalGCN, SpatialGAT, GraphSAGE, GraphTransformer, 
    HybridSpatioTemporalGNN, calculate_gnn_metrics, create_gnn_model
)


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop early
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class GNNTrainer:
    """
    Comprehensive trainer for Graph Neural Network models.
    
    Handles training, validation, model checkpointing, and experiment tracking
    for various GNN architectures on bike demand prediction. Supports both
    full-batch training (small graphs) and mini-batch training with neighbor
    sampling (large graphs).
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        experiment_name: str = None,
        save_dir: str = 'experiments',
        batch_size: int = None,
        num_neighbors: List[int] = None,
        use_neighbor_sampling: bool = 'auto'
    ):
        """
        Initialize GNN trainer.
        
        Args:
            model: GNN model to train
            device: Device to use ('cuda', 'cpu', or 'auto')
            experiment_name: Name for this experiment
            save_dir: Directory to save experiment results
            batch_size: Batch size for mini-batch training (None for full-batch)
            num_neighbors: Number of neighbors to sample per layer for NeighborLoader
            use_neighbor_sampling: Whether to use neighbor sampling ('auto', True, False)
        """
        # set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # move model to device
        self.model = model.to(self.device)
        
        # experiment setup
        self.experiment_name = experiment_name or f"gnn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # batching configuration
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [15, 10, 5]  # default neighbor sampling
        self.use_neighbor_sampling = use_neighbor_sampling
        
        # setup logging
        self.setup_logging()
        
        # training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        self.logger.info(f"Initialized GNN trainer for {model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Neighbor sampling: {self.use_neighbor_sampling}")
        
    def setup_logging(self):
        """Setup logging for the experiment"""
        log_file = self.save_dir / 'training.log'
        
        # create logger
        self.logger = logging.getLogger(f'gnn_trainer_{self.experiment_name}')
        self.logger.setLevel(logging.INFO)
        
        # clear existing handlers
        self.logger.handlers.clear()
        
        # create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def setup_training(
        self,
        optimizer_name: str = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_name: str = 'plateau',
        loss_function: str = 'mse',
        **optimizer_kwargs
    ):
        """
        Setup training components (optimizer, scheduler, loss function).
        
        Args:
            optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler_name: Learning rate scheduler ('plateau', 'cosine', 'none')
            loss_function: Loss function ('mse', 'mae', 'huber')
            **optimizer_kwargs: Additional optimizer arguments
        """
        # setup optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=optimizer_kwargs.get('momentum', 0.9),
                **{k: v for k, v in optimizer_kwargs.items() if k != 'momentum'}
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # setup scheduler
        if scheduler_name.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_name.lower() == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        # setup loss function
        if loss_function.lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function.lower() == 'mae':
            self.criterion = nn.L1Loss()
        elif loss_function.lower() == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        self.logger.info(f"Training setup complete:")
        self.logger.info(f"  Optimizer: {optimizer_name} (lr={learning_rate}, wd={weight_decay})")
        self.logger.info(f"  Scheduler: {scheduler_name}")
        self.logger.info(f"  Loss: {loss_function}")
    
    def _determine_batching_strategy(self, data: Data) -> bool:
        """
        Determine whether to use neighbor sampling based on graph size.
        
        Args:
            data: Graph data
            
        Returns:
            True if should use neighbor sampling, False for full-batch
        """
        if self.use_neighbor_sampling == 'auto':
            # use neighbor sampling for graphs with >500 nodes or >5000 edges
            num_nodes = data.x.shape[0] if data.x is not None else data.num_nodes
            num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0
            
            should_use_sampling = num_nodes > 500 or num_edges > 5000
            self.logger.info(f"Auto-detected batching strategy: {'neighbor sampling' if should_use_sampling else 'full-batch'}")
            self.logger.info(f"  Graph size: {num_nodes} nodes, {num_edges} edges")
            return should_use_sampling
        else:
            return bool(self.use_neighbor_sampling)
    
    def _create_data_loader(self, data: Data, shuffle: bool = False, input_nodes: torch.Tensor = None):
        """
        Create appropriate data loader based on batching strategy.
        
        Args:
            data: Graph data
            shuffle: Whether to shuffle data
            input_nodes: Specific nodes to sample from (for train/val split)
            
        Returns:
            DataLoader or iterable
        """
        use_sampling = self._determine_batching_strategy(data)
        
        if use_sampling and self.batch_size is not None:
            # use neighbor sampling for large graphs
            loader = NeighborLoader(
                data,
                num_neighbors=self.num_neighbors,
                batch_size=self.batch_size,
                input_nodes=input_nodes,
                shuffle=shuffle,
                num_workers=0,  # avoid multiprocessing issues
                replace=False,
                directed=True
            )
            self.logger.info(f"Created NeighborLoader: batch_size={self.batch_size}, num_neighbors={self.num_neighbors}")
            return loader
        else:
            # use full-batch training for small graphs
            if input_nodes is not None:
                # create subset of data for train/val split
                data_subset = Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    y=data.y,
                    input_nodes=input_nodes
                )
                self.logger.info(f"Created full-batch loader with input_nodes subset")
                return [data_subset]
            else:
                self.logger.info(f"Created full-batch loader")
                return [data]
    
    def train_epoch(self, train_loader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch in train_loader:
            # move batch to device
            batch = batch.to(self.device)
            
            # forward pass
            self.optimizer.zero_grad()
            
            # check if this is a neighbor sampled batch or full graph
            if hasattr(batch, 'num_sampled_nodes') and hasattr(batch, 'num_sampled_edges'):
                # neighbor sampled batch - use hierarchical sampling
                predictions = self._forward_with_sampling(batch)
                # for neighbor sampling, only compute loss on seed nodes
                if hasattr(batch, 'batch_size'):
                    targets = batch.y[:batch.batch_size]
                    predictions = predictions[:batch.batch_size]
                else:
                    targets = batch.y
            else:
                # full-batch or regular batch
                predictions = self.model(batch)
                targets = batch.y
                
                # if using input_nodes subset, filter predictions and targets
                if hasattr(batch, 'input_nodes') and batch.input_nodes is not None:
                    input_nodes = batch.input_nodes
                    predictions = predictions[input_nodes]
                    targets = targets[input_nodes]
            
            # calculate loss
            loss = self.criterion(predictions, targets)
            
            # backward pass
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # optimizer step
            self.optimizer.step()
            
            # accumulate metrics
            total_loss += loss.item()
            num_batches += 1
            all_predictions.append(predictions.detach())
            all_targets.append(targets.detach())
        
        # calculate epoch metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metrics = calculate_gnn_metrics(all_predictions, all_targets)
        else:
            metrics = {}
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, metrics
    
    def _forward_with_sampling(self, batch: Data) -> torch.Tensor:
        """
        Forward pass with hierarchical neighbor sampling using trim_to_layer.
        
        Args:
            batch: Sampled batch with num_sampled_nodes/edges attributes
            
        Returns:
            Model predictions
        """
        x, edge_index = batch.x, batch.edge_index
        num_sampled_nodes = batch.num_sampled_nodes
        num_sampled_edges = batch.num_sampled_edges
        
        # check if model supports hierarchical sampling
        if hasattr(self.model, 'convs') and hasattr(self.model.convs, '__iter__'):
            # apply hierarchical sampling for models with conv layers
            for i, conv in enumerate(self.model.convs):
                # trim to current layer
                if i < len(num_sampled_edges):
                    x, edge_index, _ = trim_to_layer(
                        i, num_sampled_nodes, num_sampled_edges, x, edge_index
                    )
                
                # apply conv layer with appropriate activation
                if hasattr(conv, '__call__'):
                    x = conv(x, edge_index)
                    
                    # apply activation and dropout if not last layer
                    if i < len(self.model.convs) - 1:
                        if hasattr(self.model, 'activation'):
                            x = self.model.activation(x)
                        else:
                            x = torch.relu(x)
                        x = torch.nn.functional.dropout(x, p=getattr(self.model, 'dropout', 0.2), training=self.training)
            
            # apply final prediction layer if exists
            if hasattr(self.model, 'predictor'):
                x = self.model.predictor(x)
            elif hasattr(self.model, 'lin'):
                x = self.model.lin(x)
                
            return x
        else:
            # fallback to regular forward for models without hierarchical support
            return self.model(batch)
        
    def validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # move batch to device
                batch = batch.to(self.device)
                
                # forward pass
                if hasattr(batch, 'num_sampled_nodes') and hasattr(batch, 'num_sampled_edges'):
                    # neighbor sampled batch
                    predictions = self._forward_with_sampling(batch)
                    if hasattr(batch, 'batch_size'):
                        targets = batch.y[:batch.batch_size]
                        predictions = predictions[:batch.batch_size]
                    else:
                        targets = batch.y
                else:
                    # full-batch
                    predictions = self.model(batch)
                    targets = batch.y
                    
                    # filter if using input_nodes
                    if hasattr(batch, 'input_nodes') and batch.input_nodes is not None:
                        input_nodes = batch.input_nodes
                        predictions = predictions[input_nodes]
                        targets = targets[input_nodes]
                
                # calculate loss
                loss = self.criterion(predictions, targets)
                
                # accumulate metrics
                total_loss += loss.item()
                num_batches += 1
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # calculate epoch metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metrics = calculate_gnn_metrics(all_predictions, all_targets)
        else:
            metrics = {}
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, metrics
    
    def fit(
        self,
        train_data: Data,
        val_data: Optional[Data] = None,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best_model: bool = True,
        save_checkpoints: bool = True,
        checkpoint_frequency: int = 10,
        verbose: bool = True,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        log_frequency: int = 10,
        min_improvement: float = 0.001
    ) -> Dict[str, Any]:
        """
        Train the GNN model.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_best_model: Whether to save the best model
            save_checkpoints: Whether to save periodic checkpoints
            checkpoint_frequency: How often to save checkpoints
            verbose: Whether to print progress
            train_mask: Mask for training nodes (for node-level tasks)
            val_mask: Mask for validation nodes (for node-level tasks)
            log_frequency: How often to log progress (every N epochs)
            min_improvement: Minimum improvement required to save best model
            
        Returns:
            Training history dictionary
        """
        if self.optimizer is None:
            self.setup_training()
        
        # create data loaders
        if train_mask is not None:
            train_nodes = train_mask.nonzero().squeeze() if train_mask.dtype == torch.bool else train_mask
        else:
            train_nodes = None
            
        if val_mask is not None:
            val_nodes = val_mask.nonzero().squeeze() if val_mask.dtype == torch.bool else val_mask
        else:
            val_nodes = None
        
        train_loader = self._create_data_loader(train_data, shuffle=True, input_nodes=train_nodes)
        
        if val_data is not None:
            val_loader = self._create_data_loader(val_data, shuffle=False, input_nodes=val_nodes)
        else:
            val_loader = None
        
        # setup early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            restore_best_weights=True
        ) if val_loader is not None else None
        
        best_val_loss = float('inf')
        last_model_save_epoch = -1
        start_time = time.time()
        
        # model parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info("=" * 120)
        self.logger.info(f"{'🚀 TRAINING CONFIGURATION':<30}")
        self.logger.info("=" * 120)
        self.logger.info(f"{'Model':<25}: {self.model.__class__.__name__} ({trainable_params:,} trainable parameters)")
        self.logger.info(f"{'Training epochs':<25}: {epochs:,}")
        self.logger.info(f"{'Strategy':<25}: {'Mini-batch' if isinstance(train_loader, NeighborLoader) else 'Full-batch'}")
        if val_loader is not None:
            self.logger.info(f"{'Early stopping':<25}: {early_stopping_patience} epochs patience, {min_improvement:.4f} min improvement")
        self.logger.info(f"{'Logging frequency':<25}: every {log_frequency} epochs")
        self.logger.info(f"{'Optimizer':<25}: {self.optimizer.__class__.__name__} (lr={self.optimizer.param_groups[0]['lr']:.1e})")
        self.logger.info(f"{'Loss function':<25}: {self.criterion.__class__.__name__}")
        if self.scheduler is not None:
            self.logger.info(f"{'Scheduler':<25}: {self.scheduler.__class__.__name__}")
        self.logger.info("=" * 120)
        
        # create header for training logs
        header = f"{'EPOCH':>8} │ {'PROGRESS':>12} │ {'TRAIN':>32} │"
        if val_loader is not None:
            header += f" {'VALIDATION':>32} │"
        header += f" {'STATUS':>25} │ {'TIME':>8}"
        
        self.logger.info(header)
        self.logger.info("─" * len(header))
        
        subheader = f"{'':>8} │ {'[████████]':>12} │ {'Loss':>12} {'R²':>8} {'MAE':>8} │"
        if val_loader is not None:
            subheader += f" {'Loss':>12} {'R²':>8} {'MAE':>8} │"
        subheader += f" {'Learning Rate':>25} │ {'(s)':>8}"
        
        self.logger.info(subheader)
        self.logger.info("─" * len(header))
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # training step
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # validation step
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            else:
                val_loss, val_metrics = float('nan'), {}
            
            # update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # determine if this is the best model so far
            is_best = False
            improvement = 0.0
            if save_best_model and val_loader is not None:
                improvement = best_val_loss - val_loss
                if improvement > min_improvement:
                    best_val_loss = val_loss
                    is_best = True
                    last_model_save_epoch = epoch
                    self.save_model('best_model.pt', silent=True)
            
            # enhanced logging
            epoch_time = time.time() - epoch_start
            should_log = verbose and ((epoch + 1) % log_frequency == 0 or is_best or epoch == 0 or epoch == epochs - 1)
            
            if should_log:
                elapsed_time = time.time() - start_time
                avg_epoch_time = elapsed_time / (epoch + 1)
                eta = avg_epoch_time * (epochs - epoch - 1)
                
                # format progress bar
                progress = (epoch + 1) / epochs
                bar_length = 8
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                progress_str = f"[{bar}] {progress*100:4.0f}%"
                
                # format train metrics
                train_loss_str = f"{train_loss:8.4f}"
                train_r2_str = f"{train_metrics.get('r2', 0):7.3f}"
                train_mae_str = f"{train_metrics.get('mae', 0):7.3f}"
                
                # format validation metrics  
                if val_loader is not None:
                    val_loss_str = f"{val_loss:8.4f}"
                    val_r2_str = f"{val_metrics.get('r2', 0):7.3f}"
                    val_mae_str = f"{val_metrics.get('mae', 0):7.3f}"
                
                # format status
                if is_best:
                    status_str = f"🏆 NEW BEST (↓{improvement:.4f})"
                elif val_loader is not None and last_model_save_epoch >= 0:
                    epochs_since_best = epoch - last_model_save_epoch
                    status_str = f"⏱️  {epochs_since_best:2d} epochs since best"
                else:
                    status_str = f"📈 LR: {current_lr:.1e}"
                
                # build main log line
                log_line = f"{epoch+1:8d} │ {progress_str:>12} │ {train_loss_str} {train_r2_str} {train_mae_str} │"
                
                if val_loader is not None:
                    log_line += f" {val_loss_str} {val_r2_str} {val_mae_str} │"
                
                log_line += f" {status_str:>25} │ {epoch_time:8.2f}"
                
                self.logger.info(log_line)
                
                # additional detailed metrics (less frequent)
                if (epoch + 1) % (log_frequency * 3) == 0 or is_best:
                    detail_metrics = []
                    
                    # train detailed metrics
                    train_rmse = train_metrics.get('rmse', 0)
                    train_mape = train_metrics.get('mape', 0)
                    detail_metrics.append(f"TRAIN → RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%")
                    
                    # validation detailed metrics
                    if val_loader is not None:
                        val_rmse = val_metrics.get('rmse', 0)
                        val_mape = val_metrics.get('mape', 0)
                        detail_metrics.append(f"VAL → RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%")
                    
                    # learning rate and ETA
                    eta_str = f"{eta/60:.0f}m" if eta > 60 else f"{eta:.0f}s"
                    detail_metrics.append(f"LR: {current_lr:.1e}, ETA: {eta_str}")
                    
                    for detail in detail_metrics:
                        self.logger.info(f"{'':>8} │ {'':>12} │ {detail:<65} │ {'':>25} │ {'':>8}")
                    
                    if detail_metrics:
                        self.logger.info("─" * len(log_line))
            
            # save checkpoints
            if save_checkpoints and (epoch + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pt')
            
            # early stopping
            if early_stopping is not None and early_stopping(val_loss, self.model):
                self.logger.info("=" * 120)
                self.logger.info(f"🛑 EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                self.logger.info(f"Best validation loss: {best_val_loss:.4f} (epoch {last_model_save_epoch + 1})")
                self.logger.info("=" * 120)
                break
        
        total_time = time.time() - start_time
        total_epochs = len(self.history['train_loss'])
        avg_time_per_epoch = total_time / max(total_epochs, 1)
        
        # final training summary
        self.logger.info("=" * 120)
        self.logger.info(f"{'✅ TRAINING COMPLETED':<30}")
        self.logger.info("=" * 120)
        
        # timing information
        self.logger.info(f"{'Total duration':<25}: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        self.logger.info(f"{'Total epochs':<25}: {total_epochs:,}")
        self.logger.info(f"{'Avg time per epoch':<25}: {avg_time_per_epoch:.3f}s")
        
        # model performance summary
        final_train_metrics = self.history['train_metrics'][-1] if self.history['train_metrics'] else {}
        final_val_metrics = self.history['val_metrics'][-1] if self.history['val_metrics'] else {}
        
        self.logger.info("─" * 120)
        self.logger.info(f"{'📊 FINAL PERFORMANCE SUMMARY':<30}")
        self.logger.info("─" * 120)
        
        # create performance table
        metrics_header = f"{'DATASET':<12} │ {'LOSS':<12} │ {'R²':<12} │ {'MAE':<12} │ {'RMSE':<12} │ {'MAPE':<12}"
        self.logger.info(metrics_header)
        self.logger.info("─" * len(metrics_header))
        
        # train metrics
        train_line = f"{'TRAIN':<12} │ {self.history['train_loss'][-1]:<12.4f} │ {final_train_metrics.get('r2', 0):<12.3f} │ "
        train_line += f"{final_train_metrics.get('mae', 0):<12.4f} │ {final_train_metrics.get('rmse', 0):<12.4f} │ "
        train_line += f"{final_train_metrics.get('mape', 0):<12.2f}"
        self.logger.info(train_line)
        
        # validation metrics
        if val_loader is not None:
            val_line = f"{'VALIDATION':<12} │ {self.history['val_loss'][-1]:<12.4f} │ {final_val_metrics.get('r2', 0):<12.3f} │ "
            val_line += f"{final_val_metrics.get('mae', 0):<12.4f} │ {final_val_metrics.get('rmse', 0):<12.4f} │ "
            val_line += f"{final_val_metrics.get('mape', 0):<12.2f}"
            self.logger.info(val_line)
        
        # best model information
        if val_loader is not None and last_model_save_epoch >= 0:
            self.logger.info("─" * len(metrics_header))
            self.logger.info(f"{'🏆 BEST MODEL':<30}")
            self.logger.info(f"{'Best validation loss':<25}: {best_val_loss:.6f} (epoch {last_model_save_epoch + 1})")
            
            final_val_loss = self.history['val_loss'][-1]
            if not np.isnan(final_val_loss):
                improvement_from_best = ((final_val_loss - best_val_loss) / abs(best_val_loss)) * 100
                if improvement_from_best > 0:
                    self.logger.info(f"{'Final vs best':<25}: {improvement_from_best:.2f}% worse")
                else:
                    self.logger.info(f"{'Final vs best':<25}: {abs(improvement_from_best):.2f}% better")
        
        # save model files
        self.logger.info("─" * 120)
        self.logger.info(f"{'💾 SAVING MODEL FILES':<30}")
        self.logger.info("─" * 120)
        
        self.save_model('final_model.pt')
        self.save_model_architecture('model_architecture.json')
        self.save_history()
        
        self.logger.info(f"{'Final model':<25}: saved to {self.save_dir}/final_model.pt")
        if val_loader is not None and last_model_save_epoch >= 0:
            self.logger.info(f"{'Best model':<25}: saved to {self.save_dir}/best_model.pt")
        self.logger.info(f"{'Model architecture':<25}: saved to {self.save_dir}/model_architecture.json")
        self.logger.info(f"{'Training history':<25}: saved to {self.save_dir}/training_history.json")
        
        self.logger.info("=" * 120)
        
        return self.history
    
    def save_model(self, filename: str, silent: bool = False):
        """Save model state dict and architecture information"""
        model_path = self.save_dir / filename
        
        # collect model initialization parameters
        model_init_params = {}
        
        # get basic parameters available in all models
        model_init_params['num_features'] = getattr(self.model, 'num_features', None)
        model_init_params['hidden_dim'] = getattr(self.model, 'hidden_dim', None)
        model_init_params['num_targets'] = getattr(self.model, 'num_targets', None)
        model_init_params['dropout'] = getattr(self.model, 'dropout', None)
        
        # get model-specific parameters
        model_class_name = self.model.__class__.__name__
        
        if model_class_name == 'TemporalGCN':
            model_init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None),
                'activation': 'relu'  # default from model
            })
        elif model_class_name == 'SpatialGAT':
            model_init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'num_heads': getattr(self.model, 'num_heads', None),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None),
                'attention_dropout': getattr(self.model, 'attention_dropout', 0.1)
            })
        elif model_class_name == 'GraphSAGE':
            model_init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'aggregation': getattr(self.model, 'aggregation', 'mean'),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None)
            })
        elif model_class_name == 'GraphTransformer':
            model_init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'num_heads': getattr(self.model, 'num_heads', None),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None)
            })
        elif model_class_name == 'HybridSpatioTemporalGNN':
            model_init_params.update({
                'use_temporal': getattr(self.model, 'use_temporal', True),
                'temporal_dim': getattr(self.model, 'temporal_dim', 64)
            })
        
        # save complete model information
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': model_class_name,
            'model_init_params': model_init_params,
            'model_config': getattr(self.model, 'config', {}),
            'training_config': {
                'batch_size': self.batch_size,
                'num_neighbors': self.num_neighbors,
                'use_neighbor_sampling': self.use_neighbor_sampling
            }
        }, model_path)
        
        if not silent:
            self.logger.info(f"Model saved to {model_path}")
    
    def save_model_architecture(self, filename: str = 'model_architecture.json'):
        """Save detailed model architecture as JSON"""
        arch_path = self.save_dir / filename
        
        # collect comprehensive architecture information
        architecture_info = {
            'model_class': self.model.__class__.__name__,
            'model_module': self.model.__class__.__module__,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_structure': str(self.model),
            'training_config': {
                'batch_size': self.batch_size,
                'num_neighbors': self.num_neighbors,
                'use_neighbor_sampling': self.use_neighbor_sampling
            },
            'initialization_parameters': {}
        }
        
        # get model-specific initialization parameters
        model_class_name = self.model.__class__.__name__
        init_params = architecture_info['initialization_parameters']
        
        # common parameters for all models
        init_params['num_features'] = getattr(self.model, 'num_features', None)
        init_params['hidden_dim'] = getattr(self.model, 'hidden_dim', None)
        init_params['num_targets'] = getattr(self.model, 'num_targets', None)
        init_params['dropout'] = getattr(self.model, 'dropout', None)
        
        # model-specific parameters
        if model_class_name == 'TemporalGCN':
            init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None),
                'activation': 'relu'
            })
        elif model_class_name == 'SpatialGAT':
            init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'num_heads': getattr(self.model, 'num_heads', None),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None),
                'attention_dropout': getattr(self.model, 'attention_dropout', 0.1)
            })
        elif model_class_name == 'GraphSAGE':
            init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'aggregation': getattr(self.model, 'aggregation', 'mean'),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None)
            })
        elif model_class_name == 'GraphTransformer':
            init_params.update({
                'num_layers': getattr(self.model, 'num_layers', None),
                'num_heads': getattr(self.model, 'num_heads', None),
                'use_batch_norm': getattr(self.model, 'use_batch_norm', None)
            })
        elif model_class_name == 'HybridSpatioTemporalGNN':
            init_params.update({
                'use_temporal': getattr(self.model, 'use_temporal', True),
                'temporal_dim': getattr(self.model, 'temporal_dim', 64)
            })
        
        # add layer information for more complex architectures
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                })
        architecture_info['layers'] = layer_info
        
        # save to JSON
        with open(arch_path, 'w') as f:
            json.dump(architecture_info, f, indent=2, default=str)
        
        self.logger.info(f"Model architecture saved to {arch_path}")
        
        return architecture_info
    
    def save_checkpoint(self, epoch: int, filename: str):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'training_config': {
                'batch_size': self.batch_size,
                'num_neighbors': self.num_neighbors,
                'use_neighbor_sampling': self.use_neighbor_sampling
            }
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        if 'training_config' in checkpoint:
            config = checkpoint['training_config']
            self.batch_size = config.get('batch_size', self.batch_size)
            self.num_neighbors = config.get('num_neighbors', self.num_neighbors)
            self.use_neighbor_sampling = config.get('use_neighbor_sampling', self.use_neighbor_sampling)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)
    
    def save_history(self):
        """Save training history"""
        history_path = self.save_dir / 'training_history.json'
        
        # convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_json[key] = [
                    item.tolist() if hasattr(item, 'tolist') else item 
                    for item in value
                ]
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def evaluate(self, test_data: Data, test_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data
            test_mask: Mask for test nodes (for node-level tasks)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("=" * 120)
        self.logger.info(f"{'🧪 EVALUATING MODEL ON TEST DATA':<30}")
        self.logger.info("=" * 120)
        
        eval_start_time = time.time()
        self.model.eval()
        
        # create test loader
        if test_mask is not None:
            test_nodes = test_mask.nonzero().squeeze() if test_mask.dtype == torch.bool else test_mask
        else:
            test_nodes = None
        
        test_loader = self._create_data_loader(test_data, shuffle=False, input_nodes=test_nodes)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # forward pass
                if hasattr(batch, 'num_sampled_nodes') and hasattr(batch, 'num_sampled_edges'):
                    predictions = self._forward_with_sampling(batch)
                    if hasattr(batch, 'batch_size'):
                        targets = batch.y[:batch.batch_size]
                        predictions = predictions[:batch.batch_size]
                    else:
                        targets = batch.y
                else:
                    predictions = self.model(batch)
                    targets = batch.y
                    
                    if hasattr(batch, 'input_nodes') and batch.input_nodes is not None:
                        input_nodes = batch.input_nodes
                        predictions = predictions[input_nodes]
                        targets = targets[input_nodes]
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_gnn_metrics(all_predictions, all_targets)
        
        # add test loss
        metrics['test_loss'] = self.criterion(all_predictions, all_targets).item()
        eval_time = time.time() - eval_start_time
        
        # log evaluation results in professional format
        self.logger.info(f"{'Evaluation time':<25}: {eval_time:.3f} seconds")
        self.logger.info(f"{'Test samples':<25}: {len(all_predictions):,}")
        
        self.logger.info("─" * 120)
        self.logger.info(f"{'📈 TEST RESULTS':<30}")
        self.logger.info("─" * 120)
        
        # create results table
        results_header = f"{'METRIC':<15} │ {'VALUE':<15} │ {'INTERPRETATION':<30}"
        self.logger.info(results_header)
        self.logger.info("─" * len(results_header))
        
        # prioritize important metrics with interpretations
        metric_interpretations = {
            'test_loss': 'Lower is better',
            'r2': 'Coefficient of determination (1.0 = perfect)',
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Square Error',
            'mape': 'Mean Absolute Percentage Error (%)'
        }
        
        metric_order = ['test_loss', 'r2', 'mae', 'rmse', 'mape']
        displayed_metrics = []
        
        for metric in metric_order:
            if metric in metrics:
                interpretation = metric_interpretations.get(metric, 'Custom metric')
                value_str = f"{metrics[metric]:.6f}" if metric == 'test_loss' else f"{metrics[metric]:.4f}"
                if metric == 'mape':
                    value_str += "%"
                
                metric_line = f"{metric.upper():<15} │ {value_str:<15} │ {interpretation:<30}"
                self.logger.info(metric_line)
                displayed_metrics.append(metric)
        
        # display any remaining metrics
        for metric, value in metrics.items():
            if metric not in displayed_metrics:
                value_str = f"{value:.4f}"
                metric_line = f"{metric.upper():<15} │ {value_str:<15} │ {'Custom metric':<30}"
                self.logger.info(metric_line)
        
        # performance assessment
        self.logger.info("─" * len(results_header))
        self.logger.info(f"{'🎯 PERFORMANCE ASSESSMENT':<30}")
        
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
        
        mape_score = metrics.get('mape', 0)
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
        
        self.logger.info("=" * 120)
        return metrics
    
    def predict(self, data: Data, input_nodes: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data
            input_nodes: Specific nodes to predict for
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        # create data loader
        data_loader = self._create_data_loader(data, shuffle=False, input_nodes=input_nodes)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                if hasattr(batch, 'num_sampled_nodes') and hasattr(batch, 'num_sampled_edges'):
                    predictions = self._forward_with_sampling(batch)
                    if hasattr(batch, 'batch_size'):
                        predictions = predictions[:batch.batch_size]
                else:
                    predictions = self.model(batch)
                    if hasattr(batch, 'input_nodes') and batch.input_nodes is not None:
                        predictions = predictions[batch.input_nodes]
                
                all_predictions.append(predictions)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        return all_predictions.cpu().numpy()


def train_gnn_experiment(
    model_type: str,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    config: Dict[str, Any],
    experiment_name: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Run a complete GNN training experiment with automatic batching strategy.
    
    Args:
        model_type: Type of GNN model to train
        train_data: Training data
        val_data: Validation data  
        test_data: Test data
        config: Configuration dictionary
        experiment_name: Name for the experiment
        
    Returns:
        Tuple of (trained_model, results_dict)
    """
    # create model
    num_features = train_data.x.shape[1]
    num_targets = train_data.y.shape[1] if len(train_data.y.shape) > 1 else 1
    
    model = create_gnn_model(
        model_type=model_type,
        num_features=num_features,
        num_targets=num_targets,
        **config.get('model_params', {})
    )
    
    # create trainer with batching configuration
    trainer_params = config.get('trainer_params', {})
    trainer = GNNTrainer(
        model=model,
        experiment_name=experiment_name,
        batch_size=trainer_params.get('batch_size', None),
        num_neighbors=trainer_params.get('num_neighbors', [15, 10, 5]),
        use_neighbor_sampling=trainer_params.get('use_neighbor_sampling', 'auto'),
        **{k: v for k, v in trainer_params.items() 
           if k not in ['batch_size', 'num_neighbors', 'use_neighbor_sampling']}
    )
    
    # setup training
    trainer.setup_training(**config.get('training_params', {}))
    
    # train model
    fit_params = config.get('fit_params', {})
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        train_mask=fit_params.get('train_mask'),
        val_mask=fit_params.get('val_mask'),
        log_frequency=fit_params.get('log_frequency', 10),
        min_improvement=fit_params.get('min_improvement', 0.001),
        **{k: v for k, v in fit_params.items() 
           if k not in ['train_mask', 'val_mask', 'log_frequency', 'min_improvement']}
    )
    
    # evaluate on test set
    test_metrics = trainer.evaluate(
        test_data, 
        test_mask=fit_params.get('test_mask')
    )
    
    results = {
        'model_type': model_type,
        'history': history,
        'test_metrics': test_metrics,
        'config': config,
        'experiment_name': trainer.experiment_name
    }
    
    return model, results 