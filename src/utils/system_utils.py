"""
System utilities and memory management for EcoBici-AI.

This module contains functions for:
- Memory usage monitoring and management
- Checkpoint creation and loading
- System resource management
- Performance optimization utilities
"""

import os
import gc
import psutil
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from src.utils.path_utils import temp_path, data_path


def log_memory_usage(step_name: str) -> None:
    """
    Log current memory usage for a specific step.
    
    Args:
        step_name: Name of the current processing step
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"  -> Memory usage after {step_name}: {memory_mb:.1f} MB")


def check_memory_and_abort_if_needed(operation_name: str, max_memory_mb: int = 8000) -> bool:
    """
    Check current memory usage and abort if it exceeds the limit.
    
    Args:
        operation_name: Name of the operation being performed
        max_memory_mb: Maximum allowed memory in MB
        
    Returns:
        True if operation should abort, False if safe to continue
    """
    process = psutil.Process(os.getpid())
    current_memory_mb = process.memory_info().rss / 1024 / 1024
    
    if current_memory_mb > max_memory_mb:
        print(f"❌ MEMORY LIMIT EXCEEDED for {operation_name}")
        print(f"   Current: {current_memory_mb:.1f} MB | Limit: {max_memory_mb} MB")
        print(f"   Aborting operation to prevent system crash.")
        return True
    
    return False


def force_garbage_collection() -> None:
    """Force garbage collection to free up memory."""
    gc.collect()


def clear_checkpoint_directory(checkpoint_dir: Union[str, Path] = data_path("feature_checkpoints")) -> None:
    """
    Clear all files in the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if checkpoint_path.exists():
        print(f"clearing checkpoint directory: {checkpoint_dir}")
        for file_path in checkpoint_path.glob("*.parquet"):
            try:
                file_path.unlink()
                print(f"  removed: {file_path.name}")
            except Exception as e:
                print(f"  failed to remove {file_path.name}: {e}")
    else:
        print(f"checkpoint directory does not exist: {checkpoint_dir}")


def create_checkpoint_directory(checkpoint_dir: str | Path = data_path("feature_checkpoints")) -> Path:
    """
    Create checkpoint directory if it doesn't exist.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Path object for the checkpoint directory
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def save_checkpoint(df: Union[pl.DataFrame, pd.DataFrame], 
                   path: str, 
                   description: str,
                   use_streaming: bool = True) -> None:
    """
    Save a checkpoint of the DataFrame.
    
    Args:
        df: DataFrame to save
        path: Path to save the checkpoint
        description: Description of the checkpoint
        use_streaming: Whether to use streaming write for Polars
    """
    try:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(df, pl.DataFrame):
            if use_streaming:
                df.write_parquet(str(checkpoint_path), use_pyarrow=True)
            else:
                df.write_parquet(str(checkpoint_path))
        elif isinstance(df, pd.DataFrame):
            df.to_parquet(str(checkpoint_path), index=False)
        else:
            raise ValueError(f"Unsupported DataFrame type: {type(df)}")
        
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Saved {description}: {file_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"  ❌ Failed to save {description}: {e}")


def load_checkpoint_if_exists(path: str, description: str) -> Optional[pl.DataFrame]:
    """
    Load a checkpoint if it exists.
    
    Args:
        path: Path to the checkpoint file
        description: Description of the checkpoint
        
    Returns:
        DataFrame if checkpoint exists, None otherwise
    """
    checkpoint_path = Path(path)
    
    if checkpoint_path.exists():
        try:
            df = pl.read_parquet(str(checkpoint_path))
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Loaded {description}: {file_size_mb:.1f} MB")
            return df
        except Exception as e:
            print(f"  ❌ Failed to load {description}: {e}")
            return None
    else:
        print(f"  ⏭️  Checkpoint not found for {description}")
        return None


def get_system_info() -> dict:
    """
    Get basic system information.
    
    Returns:
        Dictionary with system information
    """
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_free_gb': psutil.disk_usage('/').free / (1024**3)
    }


def print_system_info() -> None:
    """Print system information."""
    info = get_system_info()
    print("System Information:")
    print(f"  CPU cores: {info['cpu_count']}")
    print(f"  Total memory: {info['memory_total_gb']:.1f} GB")
    print(f"  Available memory: {info['memory_available_gb']:.1f} GB")
    print(f"  Memory usage: {info['memory_percent']:.1f}%")
    print(f"  Free disk space: {info['disk_free_gb']:.1f} GB")


def ensure_safe_thread_pool() -> None:
    """
    Ensure safe thread pool configuration for Polars.
    
    This prevents thread pool exhaustion issues during intensive operations.
    """
    try:
        import polars as pl
        
        # Get system info
        cpu_count = psutil.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Conservative thread pool sizing
        if available_memory_gb < 4:
            # Low memory systems - use fewer threads
            max_threads = min(2, cpu_count // 2)
        elif available_memory_gb < 8:
            # Medium memory systems
            max_threads = min(4, cpu_count // 2)
        else:
            # High memory systems - can use more threads
            max_threads = min(8, cpu_count)
        
        # Set Polars thread pool
        os.environ['POLARS_MAX_THREADS'] = str(max(1, max_threads))
        
        print(f"  -> Set POLARS_MAX_THREADS to {max_threads}")
        
    except Exception as e:
        print(f"  -> Warning: Could not configure thread pool: {e}")


def optimize_polars_settings() -> None:
    """Optimize Polars settings for better performance."""
    # Set streaming buffer size
    os.environ['POLARS_STREAMING_CHUNK_SIZE'] = '100000'
    
    # Enable string cache for categorical optimization
    pl.enable_string_cache()
    
    print("  -> Optimized Polars settings")


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    path = Path(file_path)
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def cleanup_temp_files(temp_dir: Union[str, Path] = temp_path()) -> None:
    """
    Clean up temporary files in the specified directory.
    
    Args:
        temp_dir: Directory containing temporary files
    """
    temp_path = Path(temp_dir)
    
    if temp_path.exists():
        temp_files = list(temp_path.glob("*"))
        if temp_files:
            print(f"cleaning up {len(temp_files)} temporary files...")
            for file_path in temp_files:
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"  failed to remove {file_path}: {e}")
        else:
            print("no temporary files to clean up")


def setup_environment() -> None:
    """Set up optimal environment for data processing."""
    print("Setting up environment...")
    
    # Print system info
    print_system_info()
    
    # Optimize settings
    ensure_safe_thread_pool()
    optimize_polars_settings()
    
    # Force garbage collection
    force_garbage_collection()
    
    print("Environment setup complete!")


def monitor_memory_during_operation(operation_name: str, max_memory_mb: int = 8000):
    """
    Context manager for monitoring memory during operations.
    
    Args:
        operation_name: Name of the operation
        max_memory_mb: Maximum allowed memory in MB
    """
    class MemoryMonitor:
        def __init__(self, name: str, max_mem: int):
            self.name = name
            self.max_memory = max_mem
            self.start_memory = None
            
        def __enter__(self):
            process = psutil.Process(os.getpid())
            self.start_memory = process.memory_info().rss / 1024 / 1024
            print(f"Starting {self.name} (memory: {self.start_memory:.1f} MB)")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            process = psutil.Process(os.getpid())
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - self.start_memory
            
            print(f"Completed {self.name}")
            print(f"  Memory: {self.start_memory:.1f} -> {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
            
            if end_memory > self.max_memory:
                print(f"  ⚠️ Memory usage exceeds limit ({self.max_memory} MB)")
            
    return MemoryMonitor(operation_name, max_memory_mb) 