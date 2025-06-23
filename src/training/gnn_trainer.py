"""
Training module for Graph Neural Network models in EcoBici demand prediction.

This module provides comprehensive training functionality for GNN models including
training loops, validation, early stopping, model checkpointing, and experiment tracking.
Multi-GPU support is included for accelerated training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import gc

from src.models.gnn_models import (
    TemporalGCN, SpatialGAT, GraphSAGE, GraphTransformer, 
    HybridSpatioTemporalGNN, calculate_gnn_metrics, create_gnn_model
)


def clear_gpu_memory():
    """Clear GPU memory cache and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - allocated
    
    return {
        'allocated': allocated,
        'reserved': reserved, 
        'free': free,
        'total': total
    }


def enable_memory_optimizations():
    """Enable PyTorch memory optimizations"""
    # enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("Flash attention enabled")
    except:
        pass
    
    # set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # enable memory mapping for large tensors
    torch.multiprocessing.set_sharing_strategy('file_system')


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information for multi-GPU setup.
    
    Returns:
        Dictionary with device information and configuration
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'gpu_names': [],
        'total_memory': [],
        'use_multi_gpu': False,
        'primary_device': 'cpu'
    }
    
    if device_info['cuda_available']:
        device_info['primary_device'] = 'cuda:0'
        
        # collect GPU information
        for i in range(device_info['num_gpus']):
            device_info['gpu_names'].append(torch.cuda.get_device_name(i))
            device_info['total_memory'].append(torch.cuda.get_device_properties(i).total_memory)
        
        # enable multi-GPU if more than one GPU available
        if device_info['num_gpus'] > 1:
            device_info['use_multi_gpu'] = True
    
    return device_info


def setup_device(device_spec: str = 'auto') -> Tuple[torch.device, bool, Dict[str, Any]]:
    """
    Setup device configuration with multi-GPU support.
    
    Args:
        device_spec: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', 'multi-gpu')
        
    Returns:
        Tuple of (primary_device, use_multi_gpu, device_info)
    """
    device_info = get_device_info()
    use_multi_gpu = False
    
    if device_spec == 'auto':
        if device_info['use_multi_gpu']:
            primary_device = torch.device('cuda:0')
            use_multi_gpu = True
        elif device_info['cuda_available']:
            primary_device = torch.device('cuda:0')
        else:
            primary_device = torch.device('cpu')
    elif device_spec == 'multi-gpu':
        if device_info['num_gpus'] > 1:
            primary_device = torch.device('cuda:0')
            use_multi_gpu = True
        elif device_info['cuda_available']:
            primary_device = torch.device('cuda:0')
        else:
            primary_device = torch.device('cpu')
    elif device_spec == 'cpu':
        primary_device = torch.device('cpu')
    elif device_spec.startswith('cuda'):
        if device_info['cuda_available']:
            primary_device = torch.device(device_spec)
        else:
            primary_device = torch.device('cpu')
    else:
        primary_device = torch.device(device_spec)
    
    return primary_device, use_multi_gpu, device_info


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
                # handle DataParallel models
                if isinstance(model, nn.DataParallel):
                    self.best_weights = model.module.state_dict().copy()
                else:
                    self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(self.best_weights)
                else:
                    model.load_state_dict(self.best_weights)
            return True
        return False


class GNNTrainer:
    """
    Comprehensive trainer for Graph Neural Network models with memory optimization and multi-GPU support.
    
    Handles training, validation, model checkpointing, and experiment tracking
    for various GNN architectures on bike demand prediction with memory-efficient techniques.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        experiment_name: str = None,
        save_dir: str = 'experiments',
        use_gradient_checkpointing: bool = True,
        max_memory_gb: float = 20.0
    ):
        """
        Initialize GNN trainer with memory optimization and multi-GPU support.
        
        Args:
            model: GNN model to train
            device: Device specification ('auto', 'cpu', 'cuda', 'multi-gpu')
            experiment_name: Name for this experiment
            save_dir: Directory to save experiment results
            use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
            max_memory_gb: Maximum memory to use per GPU (GB)
        """
        # enable memory optimizations
        enable_memory_optimizations()
        clear_gpu_memory()
        
        # setup device configuration
        self.device, self.use_multi_gpu, self.device_info = setup_device(device)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_memory_gb = max_memory_gb
        
        # store original model reference
        self.original_model = model
        
        # enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for memory efficiency")
        
        # setup multi-GPU with memory considerations
        if self.use_multi_gpu:
            print(f"Setting up memory-optimized multi-GPU training on {self.device_info['num_gpus']} GPUs:")
            for i, name in enumerate(self.device_info['gpu_names']):
                memory_gb = self.device_info['total_memory'][i] / (1024**3)
                print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
            
            # For GNNs, use single GPU with manual batching instead of DataParallel
            # DataParallel replicates the entire graph, causing memory issues
            print("WARNING: Using single GPU for GNN training due to memory efficiency.")
            print("DataParallel with large graphs can cause OOM. Consider using smaller models or graph sampling.")
            self.use_multi_gpu = False
            self.model = model.to(self.device)
        else:
            self.model = model.to(self.device)
        
        # clear memory after model setup
        clear_gpu_memory()
        
        # experiment setup
        self.experiment_name = experiment_name or f"gnn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(save_dir) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # log device configuration
        self.logger.info(f"Initialized GNN trainer for {self.original_model.__class__.__name__}")
        self.logger.info(f"Primary device: {self.device}")
        if self.use_multi_gpu:
            self.logger.info(f"Multi-GPU training enabled on {self.device_info['num_gpus']} GPUs")
            for i, name in enumerate(self.device_info['gpu_names']):
                memory_gb = self.device_info['total_memory'][i] / (1024**3)
                self.logger.info(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
        self.logger.info(f"Experiment: {self.experiment_name}")
        
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
        
    def train_epoch(self, train_data: Data) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch with memory optimization.
        
        Args:
            train_data: Training data
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # check memory before training
        memory_before = get_memory_usage()
        if memory_before['allocated'] > self.max_memory_gb:
            clear_gpu_memory()
            self.logger.warning(f"High memory usage detected: {memory_before['allocated']:.1f}GB. Cleared cache.")
        
        # move data to device with memory monitoring
        try:
            train_data = train_data.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_gpu_memory()
                self.logger.error(f"OOM during data transfer. Memory: {get_memory_usage()}")
                raise MemoryError(f"Cannot fit data on GPU. Try reducing batch size or model size.")
            raise e
        
        # forward pass with memory checkpointing
        self.optimizer.zero_grad()
        
        try:
            if self.use_gradient_checkpointing:
                # use gradient checkpointing for memory efficiency
                predictions = torch.utils.checkpoint.checkpoint(
                    self._forward_with_checkpointing, 
                    train_data,
                    use_reentrant=False
                )
            else:
                predictions = self.model(train_data)
                
            targets = train_data.y
            
            # calculate loss
            loss = self.criterion(predictions, targets)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                memory_usage = get_memory_usage()
                self.logger.error(f"OOM during forward pass. Memory: {memory_usage}")
                clear_gpu_memory()
                raise MemoryError(f"Out of memory during forward pass. Memory usage: {memory_usage['allocated']:.1f}GB")
            raise e
        
        # backward pass with memory monitoring
        try:
            loss.backward()
        except RuntimeError as e:
            if "out of memory" in str(e):
                memory_usage = get_memory_usage()
                self.logger.error(f"OOM during backward pass. Memory: {memory_usage}")
                clear_gpu_memory()
                raise MemoryError(f"Out of memory during backward pass. Memory usage: {memory_usage['allocated']:.1f}GB")
            raise e
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # optimizer step
        self.optimizer.step()
        
        # store results and clear intermediate tensors
        total_loss += loss.item()
        all_predictions.append(predictions.detach().cpu())  # move to CPU to save GPU memory
        all_targets.append(targets.detach().cpu())
        
        # clear GPU memory
        del predictions, targets, loss
        clear_gpu_memory()
        
        # calculate metrics on CPU
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_gnn_metrics(all_predictions.to(self.device), all_targets.to(self.device))
        
        # log memory usage
        memory_after = get_memory_usage()
        metrics['memory_allocated_gb'] = memory_after['allocated']
        metrics['memory_free_gb'] = memory_after['free']
        
        return total_loss, metrics
        
    def _forward_with_checkpointing(self, data: Data) -> torch.Tensor:
        """Helper method for gradient checkpointing"""
        return self.model(data)
        
    def validate_epoch(self, val_data: Data) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch with memory optimization.
        
        Args:
            val_data: Validation data
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # clear memory before validation
        clear_gpu_memory()
        
        with torch.no_grad():
            try:
                # move data to device with memory monitoring
                val_data = val_data.to(self.device)
                
                # forward pass
                predictions = self.model(val_data)
                targets = val_data.y
                
                # calculate loss
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                # move to CPU to save GPU memory
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_usage = get_memory_usage()
                    self.logger.error(f"OOM during validation. Memory: {memory_usage}")
                    clear_gpu_memory()
                    raise MemoryError(f"Out of memory during validation. Memory usage: {memory_usage['allocated']:.1f}GB")
                raise e
            
            # clear intermediate tensors
            del predictions, targets, loss
            clear_gpu_memory()
        
        # calculate metrics on CPU
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_gnn_metrics(all_predictions.to(self.device), all_targets.to(self.device))
        
        # log memory usage
        memory_after = get_memory_usage()
        metrics['memory_allocated_gb'] = memory_after['allocated']
        metrics['memory_free_gb'] = memory_after['free']
        
        return total_loss, metrics
    
    def fit(
        self,
        train_data: Data,
        val_data: Optional[Data] = None,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best_model: bool = True,
        save_checkpoints: bool = True,
        checkpoint_frequency: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the GNN model with multi-GPU support.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_best_model: Whether to save the best model
            save_checkpoints: Whether to save periodic checkpoints
            checkpoint_frequency: How often to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        if self.optimizer is None:
            self.setup_training()
        
        # setup early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            restore_best_weights=True
        ) if val_data is not None else None
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Training samples: {train_data.x.shape[0]}")
        if val_data is not None:
            self.logger.info(f"Validation samples: {val_data.x.shape[0]}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # training step
            train_loss, train_metrics = self.train_epoch(train_data)
            
            # validation step
            if val_data is not None:
                val_loss, val_metrics = self.validate_epoch(val_data)
            else:
                val_loss, val_metrics = float('nan'), {}
            
            # update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_data is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # logging with memory information
            epoch_time = time.time() - epoch_start
            if verbose and (epoch + 1) % 5 == 0:
                log_msg = f"Epoch {epoch+1:3d}/{epochs} | "
                log_msg += f"Train Loss: {train_loss:.4f} | "
                if val_data is not None:
                    log_msg += f"Val Loss: {val_loss:.4f} | "
                log_msg += f"Train R²: {train_metrics.get('r2', 0):.4f} | "
                if val_data is not None:
                    log_msg += f"Val R²: {val_metrics.get('r2', 0):.4f} | "
                log_msg += f"Time: {epoch_time:.2f}s"
                
                # add memory usage information
                if torch.cuda.is_available():
                    train_mem = train_metrics.get('memory_allocated_gb', 0)
                    train_free = train_metrics.get('memory_free_gb', 0)
                    log_msg += f" | GPU Mem: {train_mem:.1f}GB used, {train_free:.1f}GB free"
                    
                    # memory warning if getting close to limit
                    if train_mem > self.max_memory_gb * 0.9:
                        log_msg += " ⚠️ HIGH MEMORY"
                
                self.logger.info(log_msg)
                
            # periodic memory cleanup
            if (epoch + 1) % 10 == 0:
                clear_gpu_memory()
                memory_usage = get_memory_usage()
                self.logger.info(f"Memory cleanup - Allocated: {memory_usage['allocated']:.1f}GB, Free: {memory_usage['free']:.1f}GB")
            
            # save best model
            if save_best_model and val_data is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pt')
            
            # save checkpoints
            if save_checkpoints and (epoch + 1) % checkpoint_frequency == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pt')
            
            # early stopping
            if early_stopping is not None and early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # save final model, architecture, and history
        self.save_model('final_model.pt')
        self.save_model_architecture('model_architecture.json')
        self.save_history()
        
        return self.history
    
    def save_model(self, filename: str):
        """Save model state dict and architecture information"""
        model_path = self.save_dir / filename
        
        # get the actual model (unwrap DataParallel if needed)
        model_to_save = self.original_model if self.use_multi_gpu else self.model
        
        # collect model initialization parameters
        model_init_params = {}
        
        # get basic parameters available in all models
        model_init_params['num_features'] = getattr(model_to_save, 'num_features', None)
        model_init_params['hidden_dim'] = getattr(model_to_save, 'hidden_dim', None)
        model_init_params['num_targets'] = getattr(model_to_save, 'num_targets', None)
        model_init_params['dropout'] = getattr(model_to_save, 'dropout', None)
        
        # get model-specific parameters
        model_class_name = model_to_save.__class__.__name__
        
        if model_class_name == 'TemporalGCN':
            model_init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None),
                'activation': 'relu'  # default from model
            })
        elif model_class_name == 'SpatialGAT':
            model_init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'num_heads': getattr(model_to_save, 'num_heads', None),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None),
                'attention_dropout': getattr(model_to_save, 'attention_dropout', 0.1)
            })
        elif model_class_name == 'GraphSAGE':
            model_init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'aggregation': getattr(model_to_save, 'aggregation', 'mean'),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None)
            })
        elif model_class_name == 'GraphTransformer':
            model_init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'num_heads': getattr(model_to_save, 'num_heads', None),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None)
            })
        elif model_class_name == 'HybridSpatioTemporalGNN':
            model_init_params.update({
                'use_temporal': getattr(model_to_save, 'use_temporal', True),
                'temporal_dim': getattr(model_to_save, 'temporal_dim', 64)
            })
        
        # save complete model information (unwrap DataParallel state dict)
        state_dict = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()
        
        torch.save({
            'model_state_dict': state_dict,
            'model_class': model_class_name,
            'model_init_params': model_init_params,
            'model_config': getattr(model_to_save, 'config', {}),
            'multi_gpu_trained': self.use_multi_gpu,
            'device_info': self.device_info
        }, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def save_model_architecture(self, filename: str = 'model_architecture.json'):
        """Save detailed model architecture as JSON"""
        arch_path = self.save_dir / filename
        
        # get the actual model (unwrap DataParallel if needed)
        model_to_save = self.original_model if self.use_multi_gpu else self.model
        
        # collect comprehensive architecture information
        architecture_info = {
            'model_class': model_to_save.__class__.__name__,
            'model_module': model_to_save.__class__.__module__,
            'total_parameters': sum(p.numel() for p in model_to_save.parameters()),
            'trainable_parameters': sum(p.numel() for p in model_to_save.parameters() if p.requires_grad),
            'model_structure': str(model_to_save),
            'multi_gpu_trained': self.use_multi_gpu,
            'device_info': self.device_info,
            'initialization_parameters': {}
        }
        
        # get model-specific initialization parameters
        model_class_name = model_to_save.__class__.__name__
        init_params = architecture_info['initialization_parameters']
        
        # common parameters for all models
        init_params['num_features'] = getattr(model_to_save, 'num_features', None)
        init_params['hidden_dim'] = getattr(model_to_save, 'hidden_dim', None)
        init_params['num_targets'] = getattr(model_to_save, 'num_targets', None)
        init_params['dropout'] = getattr(model_to_save, 'dropout', None)
        
        # model-specific parameters
        if model_class_name == 'TemporalGCN':
            init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None),
                'activation': 'relu'
            })
        elif model_class_name == 'SpatialGAT':
            init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'num_heads': getattr(model_to_save, 'num_heads', None),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None),
                'attention_dropout': getattr(model_to_save, 'attention_dropout', 0.1)
            })
        elif model_class_name == 'GraphSAGE':
            init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'aggregation': getattr(model_to_save, 'aggregation', 'mean'),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None)
            })
        elif model_class_name == 'GraphTransformer':
            init_params.update({
                'num_layers': getattr(model_to_save, 'num_layers', None),
                'num_heads': getattr(model_to_save, 'num_heads', None),
                'use_batch_norm': getattr(model_to_save, 'use_batch_norm', None)
            })
        elif model_class_name == 'HybridSpatioTemporalGNN':
            init_params.update({
                'use_temporal': getattr(model_to_save, 'use_temporal', True),
                'temporal_dim': getattr(model_to_save, 'temporal_dim', 64)
            })
        
        # add layer information for more complex architectures
        layer_info = []
        for name, module in model_to_save.named_modules():
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
        
        # save state dict properly (unwrap DataParallel if needed)
        model_state_dict = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'multi_gpu_trained': self.use_multi_gpu,
            'device_info': self.device_info
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # load state dict properly (handle DataParallel)
        if self.use_multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
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
        
        # add device information
        history_json['device_info'] = self.device_info
        history_json['multi_gpu_used'] = self.use_multi_gpu
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")
    
    def evaluate(self, test_data: Data) -> Dict[str, float]:
        """
        Evaluate model on test data with multi-GPU support.
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        test_data = test_data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(test_data)
            targets = test_data.y
            
            # calculate metrics
            metrics = calculate_gnn_metrics(predictions, targets)
            
            # add additional evaluation metrics
            metrics['test_loss'] = self.criterion(predictions, targets).item()
            
        self.logger.info("Test Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, data: Data) -> np.ndarray:
        """
        Make predictions on new data with multi-GPU support.
        
        Args:
            data: Input data
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data)
            return predictions.cpu().numpy()


def train_gnn_experiment(
    model_type: str,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    config: Dict[str, Any],
    experiment_name: str = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Run a complete GNN training experiment with multi-GPU support.
    
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
    
    # create trainer with multi-GPU support
    trainer = GNNTrainer(
        model=model,
        experiment_name=experiment_name,
        **config.get('trainer_params', {})
    )
    
    # setup training
    trainer.setup_training(**config.get('training_params', {}))
    
    # train model
    history = trainer.fit(
        train_data=train_data,
        val_data=val_data,
        **config.get('fit_params', {})
    )
    
    # evaluate on test set
    test_metrics = trainer.evaluate(test_data)
    
    results = {
        'model_type': model_type,
        'history': history,
        'test_metrics': test_metrics,
        'config': config,
        'experiment_name': trainer.experiment_name,
        'device_info': trainer.device_info,
        'multi_gpu_used': trainer.use_multi_gpu
    }
    
    return model, results 