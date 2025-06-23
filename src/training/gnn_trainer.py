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
from src.utils.training_logger import create_training_logger, calculate_eta


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
                # deep copy to avoid reference issues
                import copy
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                print(f"Early stopping: Restored best model weights (best loss: {self.best_loss:.4f})")
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
        self.training_logger = create_training_logger(self.logger)
        
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
        
        # create file handler with detailed format
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # create console handler with minimal format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
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
        Determine whether to use neighbor sampling based on graph size and user preferences.
        
        Args:
            data: Graph data
            
        Returns:
            True if should use neighbor sampling, False for full-batch
        """
        # if user explicitly set use_neighbor_sampling, respect their choice
        if self.use_neighbor_sampling != 'auto':
            should_use_sampling = bool(self.use_neighbor_sampling)
            strategy_reason = "user-specified"
        else:
            # auto-detection logic
            num_nodes = data.x.shape[0] if data.x is not None else data.num_nodes
            num_edges = data.edge_index.shape[1] if data.edge_index is not None else 0
            
            # if user specified batch_size, they want mini-batch training
            if self.batch_size is not None:
                # for very large graphs, always use neighbor sampling
                if num_nodes > 5000 or num_edges > 50000:
                    should_use_sampling = True
                    strategy_reason = "large graph detected, using neighbor sampling"
                # for medium graphs, use neighbor sampling for efficiency  
                elif num_nodes > 1000 or num_edges > 10000:
                    should_use_sampling = True
                    strategy_reason = "medium graph with batch_size specified, using neighbor sampling"
                # for small graphs, use simple batching without neighbor sampling
                else:
                    should_use_sampling = False
                    strategy_reason = "small graph with batch_size specified, using simple mini-batch"
            else:
                # no batch_size specified, use traditional thresholds for full-batch vs sampling
                should_use_sampling = num_nodes > 500 or num_edges > 5000
                strategy_reason = "auto-detected based on graph size for full-batch training"
        
        self.logger.info(f"Batching strategy: {'neighbor sampling' if should_use_sampling else 'simple mini-batch' if self.batch_size else 'full-batch'}")
        self.logger.info(f"  Reason: {strategy_reason}")
        self.logger.info(f"  Graph size: {data.x.shape[0] if data.x is not None else data.num_nodes} nodes, {data.edge_index.shape[1] if data.edge_index is not None else 0} edges")
        if self.batch_size:
            self.logger.info(f"  Batch size: {self.batch_size}")
        
        return should_use_sampling
    
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
        elif not use_sampling and self.batch_size is not None:
            # use simple mini-batch training without neighbor sampling for small/medium graphs
            from torch_geometric.loader import DataLoader as GeometricDataLoader
            
            # create mini-batch indices for node-level tasks
            if input_nodes is not None:
                # node-level batching
                num_nodes = len(input_nodes)
                indices = torch.arange(num_nodes)
                
                # create batched data loader
                batched_data = []
                for i in range(0, num_nodes, self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    batch_nodes = input_nodes[batch_indices]
                    
                    # create mini-batch data
                    batch_data = Data(
                        x=data.x,
                        edge_index=data.edge_index,
                        y=data.y,
                        input_nodes=batch_nodes,
                        batch_size=len(batch_nodes)
                    )
                    batched_data.append(batch_data)
                
                if shuffle:
                    import random
                    random.shuffle(batched_data)
                
                self.logger.info(f"Created simple mini-batch loader: {len(batched_data)} batches of size ~{self.batch_size}")
                return batched_data
            else:
                # graph-level batching - create multiple copies with different subsets
                num_nodes = data.x.shape[0] if data.x is not None else data.num_nodes
                indices = torch.arange(num_nodes)
                
                batched_data = []
                for i in range(0, num_nodes, self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    
                    # create mini-batch data
                    batch_data = Data(
                        x=data.x,
                        edge_index=data.edge_index,
                        y=data.y,
                        input_nodes=batch_indices,
                        batch_size=len(batch_indices)
                    )
                    batched_data.append(batch_data)
                
                if shuffle:
                    import random
                    random.shuffle(batched_data)
                
                self.logger.info(f"Created simple mini-batch loader: {len(batched_data)} batches of size ~{self.batch_size}")
                return batched_data
        else:
            # use full-batch training for small graphs when no batch_size specified
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
                # full-batch or simple mini-batch
                predictions = self.model(batch)
                targets = batch.y
                
                # if using input_nodes subset (simple mini-batch), filter predictions and targets
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
                    # full-batch or simple mini-batch
                    predictions = self.model(batch)
                    targets = batch.y
                    
                    # filter if using input_nodes (simple mini-batch)
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
            save_checkpoints: Whether to save checkpoints (only when better model found)
            checkpoint_frequency: DEPRECATED - Checkpoints now saved only when better model found
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
        first_validation = True
        
        # model parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # log training configuration
        # determine training strategy for display
        if isinstance(train_loader, NeighborLoader):
            strategy_display = "Neighbor sampling mini-batch"
        elif isinstance(train_loader, list) and len(train_loader) > 1:
            strategy_display = "Simple mini-batch"
        else:
            strategy_display = "Full-batch"
        
        config_data = {
            "Model": f"{self.model.__class__.__name__} ({trainable_params:,} trainable parameters)",
            "Training epochs": f"{epochs:,}",
            "Strategy": strategy_display,
            "Logging frequency": f"every {log_frequency} epochs",
            "Optimizer": f"{self.optimizer.__class__.__name__} (lr={self.optimizer.param_groups[0]['lr']:.1e})",
            "Loss function": self.criterion.__class__.__name__
        }
        
        # add batch size information if using mini-batch
        if self.batch_size is not None:
            config_data["Batch size"] = f"{self.batch_size}"
            if isinstance(train_loader, list):
                config_data["Number of batches"] = f"{len(train_loader)}"
        
        if val_loader is not None:
            config_data["Early stopping"] = f"{early_stopping_patience} epochs patience, {min_improvement:.4f} min improvement"
        
        if save_checkpoints:
            config_data["Checkpoints"] = "Saved only when better model found (max 3 kept)"
        
        if self.scheduler is not None:
            config_data["Scheduler"] = self.scheduler.__class__.__name__
            
        self.training_logger.log_config_table(config_data, "TRAINING CONFIGURATION")
        
        # create training progress header
        metrics_names = ['Loss', 'R²', 'MAE']
        header_length = self.training_logger.log_training_header(metrics_names, val_loader is not None)
        
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
                if first_validation:
                    # first validation - always save as best
                    best_val_loss = val_loss
                    is_best = True
                    last_model_save_epoch = epoch
                    improvement = 0.0  # no improvement to show on first epoch
                    first_validation = False
                    self.save_model('best_model.pt', silent=True)
                else:
                    # subsequent validations - check for improvement
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
                eta = calculate_eta(elapsed_time, epoch + 1, epochs)
                
                # prepare status information
                status_info = {
                    'is_best': is_best and improvement > 0,  # only show "NEW BEST" when there's actual improvement
                    'improvement': improvement if is_best else 0.0,
                    'epochs_since_best': epoch - last_model_save_epoch if last_model_save_epoch >= 0 else -1,
                    'learning_rate': current_lr
                }
                
                # prepare metrics with standard names
                train_log_metrics = {
                    'loss': train_loss,
                    'r2': train_metrics.get('r2', 0),
                    'mae': train_metrics.get('mae', 0)
                }
                
                val_log_metrics = None
                if val_loader is not None:
                    val_log_metrics = {
                        'loss': val_loss,
                        'r2': val_metrics.get('r2', 0),
                        'mae': val_metrics.get('mae', 0)
                    }
                
                # log epoch progress
                self.training_logger.log_epoch_progress(
                    epoch + 1, epochs, train_log_metrics, val_log_metrics,
                    status_info, epoch_time, ['loss', 'r2', 'mae']
                )
            
            # save checkpoints only when finding better models
            if save_checkpoints and is_best and improvement > 0:
                checkpoint_filename = f'checkpoint_best_epoch_{epoch+1}_loss_{val_loss:.6f}.pt'
                self.save_checkpoint(epoch, checkpoint_filename)
                self.logger.info(f"Saved checkpoint for new best model: {checkpoint_filename}")
                
                # optionally clean up old checkpoints (keep only last 3 best)
                self._cleanup_old_checkpoints(keep_last=3)
            
            # early stopping
            if early_stopping is not None and early_stopping(val_loss, self.model):
                self.training_logger.log_section_header(
                    f"EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}"
                )
                self.logger.info(f"Best validation loss: {best_val_loss:.4f} (epoch {last_model_save_epoch + 1})")
                break
        
        total_time = time.time() - start_time
        total_epochs = len(self.history['train_loss'])
        
        # determine which metrics to use for final summary
        use_best_metrics = (early_stopping and early_stopping.restore_best_weights and 
                           early_stopping.best_weights is not None and 
                           early_stopping.counter >= early_stopping.patience and
                           last_model_save_epoch >= 0)
        
        if use_best_metrics:
            # use metrics from the best epoch (when early stopping restored weights)
            final_train_metrics = self.history['train_metrics'][last_model_save_epoch] if self.history['train_metrics'] else {}
            final_val_metrics = self.history['val_metrics'][last_model_save_epoch] if self.history['val_metrics'] else {}
            
            # add loss to metrics for consistency
            final_train_metrics = dict(final_train_metrics)
            final_train_metrics['loss'] = self.history['train_loss'][last_model_save_epoch]
            
            if val_loader is not None:
                final_val_metrics = dict(final_val_metrics)  
                final_val_metrics['loss'] = self.history['val_loss'][last_model_save_epoch]
            
            self.logger.info(f"Using metrics from best epoch ({last_model_save_epoch + 1}) for final summary")
        else:
            # use metrics from last epoch (normal case)
            final_train_metrics = self.history['train_metrics'][-1] if self.history['train_metrics'] else {}
            final_val_metrics = self.history['val_metrics'][-1] if self.history['val_metrics'] else {}
            
            # add loss to metrics for consistency
            final_train_metrics = dict(final_train_metrics)
            final_train_metrics['loss'] = self.history['train_loss'][-1]
            
            if val_loader is not None:
                final_val_metrics = dict(final_val_metrics)  
                final_val_metrics['loss'] = self.history['val_loss'][-1]
        
        # prepare best model information
        best_info = None
        if val_loader is not None and last_model_save_epoch >= 0:
            # determine final loss based on which metrics we're using
            if use_best_metrics:
                final_loss = self.history['val_loss'][last_model_save_epoch]
            else:
                final_loss = self.history['val_loss'][-1]
                
            best_info = {
                'best_loss': best_val_loss,
                'best_epoch': last_model_save_epoch + 1,
                'final_loss': final_loss,
                'using_best_metrics': use_best_metrics
            }
        
        # prepare saved files information
        saved_files = {
            'Final model': f"{self.save_dir}/final_model.pt",
            'Model architecture': f"{self.save_dir}/model_architecture.json",
            'Training history': f"{self.save_dir}/training_history.json"
        }
        
        if val_loader is not None and last_model_save_epoch >= 0:
            saved_files['Best model'] = f"{self.save_dir}/best_model.pt"
        
        # save model files
        # if early stopping was triggered and restored best weights, update best_model.pt
        if (early_stopping and early_stopping.restore_best_weights and 
            early_stopping.best_weights is not None and early_stopping.counter >= early_stopping.patience):
            self.logger.info("Early stopping restored best weights - updating best_model.pt")
            self.save_model('best_model.pt', silent=True)
            self.logger.info("Note: Final model contains restored best weights from early stopping")
        elif val_loader is not None and last_model_save_epoch >= 0:
            self.logger.info(f"Note: Best model is from epoch {last_model_save_epoch + 1} (saved as best_model.pt)")
        
        # save final model and other files
        self.save_model('final_model.pt')
        self.save_model_architecture('model_architecture.json')
        self.save_history()
        
        # log comprehensive training summary
        self.training_logger.log_training_summary(
            total_time, total_epochs, final_train_metrics, final_val_metrics,
            best_info, saved_files
        )
        
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
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """
        Clean up old checkpoint files, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        import glob
        import os
        
        # find all checkpoint files in the experiment directory
        checkpoint_pattern = str(self.save_dir / 'checkpoint_best_*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= keep_last:
            return  # not enough files to clean up
        
        # sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # remove old checkpoints (keep only the newest ones)
        files_to_remove = checkpoint_files[keep_last:]
        
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                filename = Path(file_path).name
                self.logger.info(f"Removed old checkpoint: {filename}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
    
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
                    # neighbor sampled batch
                    predictions = self._forward_with_sampling(batch)
                    if hasattr(batch, 'batch_size'):
                        targets = batch.y[:batch.batch_size]
                        predictions = predictions[:batch.batch_size]
                    else:
                        targets = batch.y
                else:
                    # full-batch or simple mini-batch
                    predictions = self.model(batch)
                    targets = batch.y
                    
                    # filter if using input_nodes (simple mini-batch)
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
        
        # log evaluation using training logger
        self.training_logger.log_evaluation_header(eval_time, len(all_predictions))
        self.training_logger.log_evaluation_results(metrics, with_assessment=True)
        
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
    
    # ensure we're using the best model for evaluation
    best_model_path = trainer.save_dir / 'best_model.pt'
    final_model_path = trainer.save_dir / 'final_model.pt'
    
    # determine which model to use for evaluation
    if best_model_path.exists():
        trainer.logger.info("Loading best model for evaluation...")
        try:
            checkpoint = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.logger.info("Best model loaded successfully")
        except Exception as e:
            trainer.logger.warning(f"Failed to load best model: {e}")
            trainer.logger.info("Using current model state for evaluation")
    else:
        # fallback to current model state (which may already contain best weights from early stopping)
        trainer.logger.info("No best model file found, using current model state for evaluation")
        if final_model_path.exists():
            trainer.logger.info("(Current model state may contain best weights if early stopping was triggered)")
    
    # evaluate on test set with best model
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
    
    return trainer.model, results 