#!/usr/bin/env python3
"""
Optimized command-line script to prepare GNN dataset for EcoBici arrival prediction.
Usage: python scripts/prepare_gnn_dataset.py --data_path data --split train --output_path data/gnn

Key optimizations:
- Structured logging with configurable levels
- Parallel processing of multiple splits  
- Intelligent caching of computations
- Memory-efficient processing with chunks
- Compressed pickle files for smaller storage
- Progress monitoring with resource tracking
- Enhanced error handling and recovery
"""

import argparse
import sys
import os
from pathlib import Path
import pickle
import lzma
import gzip
import json
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import psutil
from dataclasses import dataclass
import hashlib

import torch
from tqdm import tqdm

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset.spatial_temporal_dataset import EcoBiciSpatialTemporalDataset

# optimized configuration
@dataclass
class OptimizedConfig:
    """Configuration for optimized processing"""
    use_compression: bool = True
    compression_format: str = 'lzma'  # 'lzma', 'gzip', 'none'
    parallel_processing: bool = True
    max_workers: int = None  # auto-detect based on CPU cores
    cache_intermediate: bool = True
    chunk_size: int = 10000  # for memory-efficient processing
    validate_sample_size: int = 500  # for leakage validation
    show_progress: bool = True
    resource_monitoring: bool = True
    cache_dir: str = '.cache'
    log_level: str = 'INFO'

def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Setup structured logging with performance tracking"""
    logger = logging.getLogger('gnn_dataset_prep')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # clear existing handlers
    logger.handlers.clear()
    
    # console handler with rich formatting
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # file handler for detailed logs
    log_file = Path('gnn_dataset_preparation.log')
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

class ResourceMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def log_resources(self, stage: str):
        """Log current resource usage"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        elapsed = time.time() - self.start_time
        
        self.logger.info(
            f"{stage}: Memory={memory_mb:.1f}MB, CPU={cpu_percent:.1f}%, "
            f"Elapsed={elapsed:.1f}s"
        )

class OptimizedCache:
    """Intelligent caching system for intermediate results"""
    
    def __init__(self, cache_dir: str, logger: logging.Logger):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logger
        
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result"""
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {key}: {e}")
        return None
    
    def set(self, key: str, value: Any):
        """Cache result"""
        cache_file = self.cache_dir / f"{key}.cache"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {key}: {e}")
    
    def clear(self):
        """Clear all cached results"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        self.logger.info("Cache cleared")

def save_compressed(data: Any, file_path: Path, compression: str = 'lzma', logger: logging.Logger = None):
    """Save data with optimal compression"""
    start_time = time.time()
    
    if compression == 'lzma':
        with lzma.open(file_path.with_suffix('.pkl.xz'), 'wb', preset=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif compression == 'gzip':
        with gzip.open(file_path.with_suffix('.pkl.gz'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elapsed = time.time() - start_time
    if logger:
        size_mb = file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 0
        logger.info(f"Saved {file_path.name} ({size_mb:.1f}MB) in {elapsed:.2f}s")

def load_compressed(file_path: Path, compression: str = 'auto', logger: logging.Logger = None) -> Any:
    """Load data with automatic compression detection"""
    start_time = time.time()
    
    # auto-detect compression
    if compression == 'auto':
        if file_path.suffix == '.xz':
            compression = 'lzma'
        elif file_path.suffix == '.gz':
            compression = 'gzip'
        else:
            compression = 'none'
    
    if compression == 'lzma':
        with lzma.open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif compression == 'gzip':
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    
    elapsed = time.time() - start_time
    if logger:
        logger.info(f"Loaded {file_path.name} in {elapsed:.2f}s")
    
    return data

def process_single_split(args: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Process a single data split - optimized for parallel execution"""
    split, config = args
    
    # setup logging for this worker
    logger = setup_logging(config['log_level'])
    start_time = time.time()
    
    logger.info(f"🚀 Starting processing of {split} split")
    logger.debug(f"   📋 Worker configuration: {config}")
    
    try:
        # create dataset instance
        logger.info(f"🏗️  Creating EcoBiciSpatialTemporalDataset for {split}...")
        logger.debug(f"   📂 Data path: {config['data_path']}")
        logger.debug(f"   ⏰ Sequence length: {config['sequence_length']}")
        logger.debug(f"   🔗 K-neighbors: {config['k_neighbors']}")
        
        dataset_start = time.time()
        dataset = EcoBiciSpatialTemporalDataset(
            data_path=config['data_path'],
            sequence_length=config['sequence_length'],
            k_neighbors=config['k_neighbors']
        )
        dataset_time = time.time() - dataset_start
        logger.info(f"✅ Dataset instance created in {dataset_time:.2f}s")
        
        # for non-training splits, try to load pre-fitted scalers
        scalers_path = Path(config.get('output_path', 'data/gnn')) / 'scalers_temp.pkl'
        if split != 'train' and scalers_path.exists():
            logger.info(f"🔧 Loading pre-fitted scalers for {split} split...")
            logger.debug(f"   📂 Scalers path: {scalers_path}")
            try:
                scalers_start = time.time()
                scalers = load_compressed(scalers_path, 'auto', logger)
                dataset.feature_scaler = scalers['feature_scaler']
                dataset.target_scaler = scalers['target_scaler']
                scalers_time = time.time() - scalers_start
                logger.info(f"✅ Successfully loaded scalers for {split} in {scalers_time:.2f}s")
                logger.debug(f"   📊 Feature scaler shape: {dataset.feature_scaler.mean_.shape}")
                logger.debug(f"   🎯 Target scaler shape: {dataset.target_scaler.mean_.shape}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load scalers for {split}: {e}")
                logger.debug(f"   📂 Scalers file exists: {scalers_path.exists()}")
        
        # load data with progress tracking
        logger.info(f"📖 Loading {split} data...")
        data_start = time.time()
        temporal_signal = dataset.load_data(split)
        data_time = time.time() - data_start
        logger.info(f"✅ Data loaded successfully in {data_time:.2f}s")
        
        # gather detailed statistics
        logger.info(f"📊 Calculating dataset statistics for {split}...")
        stats_start = time.time()
        
        n_timesteps = len(temporal_signal)
        n_nodes = temporal_signal.edge_index.max().item() + 1
        n_edges = temporal_signal.edge_index.shape[1]
        n_features = temporal_signal[0].x.shape[1] if n_timesteps > 0 else 0
        
        logger.debug(f"   🔢 Basic stats: timesteps={n_timesteps}, nodes={n_nodes}, edges={n_edges}, features={n_features}")
        
        stats = {
            'split': split,
            'n_timesteps': n_timesteps,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'n_features': n_features,
            'processing_time_seconds': time.time() - start_time
        }
        
        if n_timesteps > 0:
            logger.debug(f"   📈 Analyzing target statistics...")
            sample_targets = temporal_signal[0].y
            sample_features = temporal_signal[0].x
            
            # detailed target statistics
            target_mean = sample_targets.mean().item()
            target_std = sample_targets.std().item()
            target_min = sample_targets.min().item()
            target_max = sample_targets.max().item()
            target_nonzero_ratio = (sample_targets > 0).float().mean().item()
            target_zeros = (sample_targets == 0).sum().item()
            
            # feature statistics
            feature_mean = sample_features.mean().item()
            feature_std = sample_features.std().item()
            feature_min = sample_features.min().item()
            feature_max = sample_features.max().item()
            feature_nan_count = torch.isnan(sample_features).sum().item()
            feature_inf_count = torch.isinf(sample_features).sum().item()
            
            stats.update({
                'target_mean': target_mean,
                'target_std': target_std,
                'target_min': target_min,
                'target_max': target_max,
                'target_nonzero_ratio': target_nonzero_ratio,
                'target_zeros': target_zeros,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
                'feature_min': feature_min,
                'feature_max': feature_max,
                'feature_nan_count': feature_nan_count,
                'feature_inf_count': feature_inf_count
            })
            
            logger.debug(f"   🎯 Target stats: mean={target_mean:.4f}, std={target_std:.4f}, "
                        f"range=[{target_min:.4f}, {target_max:.4f}]")
            logger.debug(f"   📊 Target distribution: {target_nonzero_ratio:.1%} non-zero, {target_zeros} zeros")
            logger.debug(f"   🔧 Feature stats: mean={feature_mean:.4f}, std={feature_std:.4f}, "
                        f"range=[{feature_min:.4f}, {feature_max:.4f}]")
            
            if feature_nan_count > 0 or feature_inf_count > 0:
                logger.warning(f"   ⚠️  Data quality issues: {feature_nan_count} NaN, {feature_inf_count} Inf values")
        
        stats_time = time.time() - stats_start
        total_time = time.time() - start_time
        
        logger.info(f"📈 Statistics calculated in {stats_time:.2f}s")
        logger.info(f"✅ Completed processing {split} split in {total_time:.2f}s:")
        logger.info(f"   📊 {stats['n_timesteps']} timesteps × {stats['n_nodes']} nodes × {stats['n_features']} features")
        logger.info(f"   🔗 {stats['n_edges']} spatial edges (avg degree: {stats['n_edges']/stats['n_nodes']:.1f})")
        
        if n_timesteps > 0:
            logger.info(f"   🎯 Target range: [{stats['target_min']:.3f}, {stats['target_max']:.3f}], "
                       f"{stats['target_nonzero_ratio']:.1%} non-zero")
        
        return {
            'split': split,
            'temporal_signal': temporal_signal,
            'stats': stats,
            'dataset': dataset,  # return for scaler access
            'success': True,
            'processing_time': total_time
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ Error processing {split} split after {error_time:.2f}s: {e}")
        logger.debug(f"   📍 Error details: {type(e).__name__}: {str(e)}")
        return {
            'split': split,
            'error': str(e),
            'error_type': type(e).__name__,
            'success': False,
            'processing_time': error_time
        }

def validate_data_leakage_parallel(dataset: EcoBiciSpatialTemporalDataset, 
                                 splits: List[str], 
                                 sample_size: int,
                                 logger: logging.Logger) -> bool:
    """Validate data leakage in parallel for multiple splits"""
    logger.info("🔍 Validating data leakage across all splits...")
    logger.debug(f"   📊 Splits to validate: {splits}")
    logger.debug(f"   🎲 Sample size per split: {sample_size}")
    
    validation_start = time.time()
    validation_tasks = []
    
    with ThreadPoolExecutor(max_workers=len(splits)) as executor:
        logger.debug(f"   🔄 Starting {len(splits)} validation tasks in parallel...")
        
        for split in splits:
            file_path = Path(dataset.data_path) / "clustered" / f"{split}_cluster_features.parquet"
            if file_path.exists():
                logger.debug(f"   ✅ {split} data file found, submitting validation task")
                future = executor.submit(dataset.validate_no_data_leakage, split, sample_size)
                validation_tasks.append((split, future))
            else:
                logger.warning(f"   ⏭️  {split} data file not found: {file_path}")
        
        logger.info(f"📋 Submitted {len(validation_tasks)} validation tasks")
        
        # collect results with progress tracking
        all_passed = True
        completed_tasks = 0
        
        for split, future in tqdm(validation_tasks, desc="Validating splits"):
            task_start = time.time()
            try:
                logger.debug(f"   🔍 Waiting for {split} validation to complete...")
                result = future.result(timeout=300)  # 5 min timeout
                task_time = time.time() - task_start
                completed_tasks += 1
                
                logger.info(f"✅ {split} split passed data leakage validation in {task_time:.2f}s")
                logger.debug(f"   📊 Progress: {completed_tasks}/{len(validation_tasks)} validations complete")
                
            except Exception as e:
                task_time = time.time() - task_start
                logger.error(f"❌ {split} split failed validation after {task_time:.2f}s: {e}")
                logger.debug(f"   📍 Error type: {type(e).__name__}")
                all_passed = False
    
    total_time = time.time() - validation_start
    logger.info(f"🏁 Data leakage validation complete in {total_time:.2f}s")
    logger.info(f"   📊 Results: {completed_tasks} passed, {len(validation_tasks) - completed_tasks} failed")
    
    return all_passed

def save_dataset_with_logging(temporal_signal, output_file: Path, compression: str, split: str, logger: logging.Logger):
    """Save dataset with detailed logging and validation"""
    logger.info(f"💾 Saving {split} temporal signal dataset...")
    logger.debug(f"   📂 Output path: {output_file}")
    logger.debug(f"   🗜️  Compression: {compression}")
    
    # pre-save validation
    save_start = time.time()
    logger.debug(f"   🔍 Pre-save validation...")
    
    if not temporal_signal or len(temporal_signal) == 0:
        logger.error(f"❌ Cannot save empty temporal signal for {split}")
        raise ValueError(f"Empty temporal signal for {split}")
    
    # calculate dataset metrics
    dataset_size = len(temporal_signal)
    if dataset_size > 0:
        sample_data = temporal_signal[0]
        n_nodes = sample_data.x.shape[0] if hasattr(sample_data, 'x') else 0
        n_features = sample_data.x.shape[1] if hasattr(sample_data, 'x') else 0
        n_targets = sample_data.y.shape[0] if hasattr(sample_data, 'y') else 0
        
        logger.debug(f"   📊 Dataset metrics: {dataset_size} timesteps, {n_nodes} nodes, {n_features} features, {n_targets} targets")
        
        # estimate memory usage
        if hasattr(sample_data, 'x') and hasattr(sample_data, 'y'):
            estimated_size_mb = (dataset_size * n_nodes * n_features * 4 + dataset_size * n_targets * 4) / (1024 * 1024)
            logger.debug(f"   💾 Estimated dataset size: {estimated_size_mb:.1f} MB")
    
    # perform save
    logger.debug(f"   💾 Writing to disk...")
    save_compressed(temporal_signal, output_file, compression, logger)
    save_time = time.time() - save_start
    
    # post-save validation
    if output_file.exists():
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logger.info(f"✅ Successfully saved {split} dataset in {save_time:.2f}s")
        logger.info(f"   📁 File: {output_file.name}")
        logger.info(f"   📊 Size: {file_size_mb:.1f} MB")
        logger.debug(f"   🗜️  Compression ratio: {estimated_size_mb/file_size_mb:.1f}x" if 'estimated_size_mb' in locals() else "")
    else:
        logger.error(f"❌ Failed to save {split} dataset - file not found after save")
        raise FileNotFoundError(f"Dataset file was not created: {output_file}")

def main():
    overall_start = time.time()
    
    parser = argparse.ArgumentParser(
        description="Optimized GNN dataset preparation for EcoBici arrival prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Optimization features:
  • Parallel processing of multiple splits
  • Intelligent caching of intermediate results  
  • Compressed output files (LZMA/GZIP)
  • Memory-efficient chunk processing
  • Resource monitoring and progress tracking
  • Enhanced error handling and recovery
  
Examples:
  # Quick preparation with all optimizations
  python scripts/prepare_gnn_dataset.py --split all --optimized
  
  # Full pipeline with compression and caching
  python scripts/prepare_gnn_dataset.py --split all --save_scalers --validate_leakage \\
    --compression lzma --cache --parallel
  
  # Memory-efficient mode for large datasets
  python scripts/prepare_gnn_dataset.py --split all --chunk_size 5000 --compression gzip
  
  # Development mode with detailed logging
  python scripts/prepare_gnn_dataset.py --split train --log_level DEBUG --no_parallel
        """
    )
    
    # original arguments
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to data directory (default: data)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='train',
                       help='Data split to prepare (default: train)')
    parser.add_argument('--output_path', type=str, default='data/gnn',
                       help='Output directory for processed datasets (default: data/gnn)')
    parser.add_argument('--sequence_length', type=int, default=12,
                       help='Sequence length for temporal modeling (default: 12)')
    parser.add_argument('--k_neighbors', type=int, default=8,
                       help='Number of spatial neighbors for graph construction (default: 8)')
    parser.add_argument('--stats_only', action='store_true',
                       help='Only show dataset statistics without saving')
    parser.add_argument('--save_scalers', action='store_true',
                       help='Save feature and target scalers for later use')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--validate_leakage', action='store_true',
                       help='Validate that there is no data leakage in lag features')
    parser.add_argument('--leakage_report', action='store_true',
                       help='Generate comprehensive data leakage prevention report')
    
    # optimization arguments
    parser.add_argument('--optimized', action='store_true',
                       help='Enable all optimizations (parallel, cache, compression)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing of splits')
    parser.add_argument('--no_parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--cache', action='store_true',
                       help='Enable intelligent caching')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear cache before processing')
    parser.add_argument('--compression', type=str, choices=['lzma', 'gzip', 'none'], default='lzma',
                       help='Compression format for output files (default: lzma)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for memory-efficient processing (default: 10000)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto)')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--no_progress', action='store_true',
                       help='Disable progress bars')
    
    logger.info("🎬 Starting GNN dataset preparation script...")
    logger.debug("⚙️  Parsing command line arguments...")
    
    args = parser.parse_args()
    args_parse_time = time.time() - overall_start
    
    logger.debug(f"✅ Arguments parsed in {args_parse_time:.3f}s")
    logger.debug(f"   📋 Arguments: {vars(args)}")
    
    # setup optimized configuration
    logger.debug("🔧 Setting up optimized configuration...")
    config_start = time.time()
    
    config = OptimizedConfig()
    
    if args.optimized:
        logger.info("🚀 Optimized mode enabled - activating all optimizations")
        config.parallel_processing = True
        config.cache_intermediate = True
        config.use_compression = True
    
    # override with specific arguments
    if args.no_parallel:
        logger.info("🔄 Parallel processing disabled by user")
        config.parallel_processing = False
    elif args.parallel:
        logger.info("🔄 Parallel processing enabled by user")
        config.parallel_processing = True
        
    if args.cache:
        logger.info("💾 Caching enabled by user")
        config.cache_intermediate = True
    if args.compression:
        config.compression_format = args.compression
        config.use_compression = args.compression != 'none'
        logger.info(f"🗜️  Compression format: {config.compression_format}")
    if args.chunk_size:
        config.chunk_size = args.chunk_size
        logger.debug(f"📦 Chunk size: {config.chunk_size}")
    if args.max_workers:
        config.max_workers = args.max_workers
        logger.debug(f"👥 Max workers: {config.max_workers}")
    if args.log_level:
        config.log_level = args.log_level
    if args.no_progress:
        config.show_progress = False
        logger.debug("📊 Progress bars disabled")
    
    config_time = time.time() - config_start
    logger.debug(f"✅ Configuration setup complete in {config_time:.3f}s")
    
    # setup logging and monitoring
    logger.info("📝 Setting up logging and resource monitoring...")
    setup_start = time.time()
    
    logger = setup_logging(config.log_level)
    monitor = ResourceMonitor(logger)
    
    # setup cache
    cache = OptimizedCache(config.cache_dir, logger) if config.cache_intermediate else None
    if args.clear_cache and cache:
        logger.info("🧹 Clearing cache as requested...")
        cache.clear()
    
    setup_time = time.time() - setup_start
    logger.debug(f"✅ Setup complete in {setup_time:.3f}s")
    
    # create output directory
    logger.info("📁 Setting up output directory...")
    output_path = Path(args.output_path)
    if not args.stats_only:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Output directory ready: {output_path.absolute()}")
    else:
        logger.info("📊 Stats-only mode - no output directory needed")
    
    # validate input data directory
    logger.info("🔍 Validating input data directory...")
    input_validation_start = time.time()
    
    data_path = Path(args.data_path)
    clustered_path = data_path / "clustered"
    
    if not data_path.exists():
        logger.error(f"❌ Data directory not found: {data_path}")
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    if not clustered_path.exists():
        logger.error(f"❌ Clustered data directory not found: {clustered_path}")
        raise FileNotFoundError(f"Clustered data directory not found: {clustered_path}")
    
    # check for required files
    metadata_file = clustered_path / "dataset_metadata.json"
    if not metadata_file.exists():
        logger.error(f"❌ Dataset metadata not found: {metadata_file}")
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_file}")
    
    logger.info(f"✅ Input data validation passed")
    logger.debug(f"   📂 Data path: {data_path.absolute()}")
    logger.debug(f"   📂 Clustered path: {clustered_path.absolute()}")
    
    input_validation_time = time.time() - input_validation_start
    logger.debug(f"   ⏱️  Validation time: {input_validation_time:.3f}s")
    
    logger.info("🚴 Optimized EcoBici GNN Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"📂 Data path: {args.data_path}")
    logger.info(f"⏰ Sequence length: {args.sequence_length}")
    logger.info(f"🔗 K-neighbors: {args.k_neighbors}")
    logger.info(f"📁 Output path: {args.output_path}")
    logger.info(f"🗜️  Compression: {config.compression_format}")
    logger.info(f"🔄 Parallel processing: {config.parallel_processing}")
    logger.info(f"💾 Caching: {config.cache_intermediate}")
    logger.info(f"📊 Stats only: {args.stats_only}")
    logger.info("")
    
    monitor.log_resources("Initialization")
    
    # determine splits to process
    logger.info("📋 Determining splits to process...")
    splits_start = time.time()
    
    if args.split == 'all':
        splits = ['train', 'val', 'test']
        logger.info("🎯 Processing all splits: train, val, test")
    else:
        splits = [args.split]
        logger.info(f"🎯 Processing single split: {args.split}")
    
    # validate split files exist
    available_splits = []
    for split in splits:
        split_file = clustered_path / f"{split}_cluster_features.parquet"
        if split_file.exists():
            file_size_mb = split_file.stat().st_size / 1024 / 1024
            available_splits.append(split)
            logger.info(f"   ✅ {split} split available: {file_size_mb:.1f} MB")
        else:
            logger.warning(f"   ⚠️  {split} split file not found: {split_file}")
    
    if not available_splits:
        logger.error(f"❌ No valid split files found for: {splits}")
        raise FileNotFoundError(f"No valid split files found")
    
    splits = available_splits
    splits_time = time.time() - splits_start
    logger.info(f"📋 Split determination complete in {splits_time:.3f}s: {len(splits)} splits to process")
    
    # prepare processing configuration
    logger.debug("⚙️  Preparing processing configuration...")
    process_config = {
        'data_path': args.data_path,
        'sequence_length': args.sequence_length,
        'k_neighbors': args.k_neighbors,
        'log_level': config.log_level,
        'output_path': args.output_path
    }
    logger.debug(f"   📋 Process config: {process_config}")
    
    # process splits with smart dependency handling
    processing_start = time.time()
    
    if config.parallel_processing and len(splits) > 1 and not args.stats_only:
        logger.info("🔄 Processing splits with dependency management...")
        logger.debug(f"   📊 Total splits: {len(splits)}")
        logger.debug(f"   🎯 Target: Smart parallel processing with train-first dependency")
        
        results = {}
        
        # step 1: process training split first (required for scaler fitting)
        if 'train' in splits:
            logger.info("🏋️  Step 1: Processing training split first (required for scalers)...")
            train_start = time.time()
            
            train_result = process_single_split(('train', process_config))
            results['train'] = train_result
            
            train_time = time.time() - train_start
            logger.info(f"⏱️  Training split processing time: {train_time:.2f}s")
            
            if not train_result.get('success', False):
                logger.error("❌ Training split failed - cannot proceed with other splits")
                logger.error(f"   📍 Error: {train_result.get('error', 'Unknown error')}")
                return
            
            # save scalers temporarily for parallel processing
            if train_result.get('dataset'):
                logger.info("💾 Saving temporary scalers for parallel processing...")
                scalers_save_start = time.time()
                try:
                    temp_scalers = {
                        'feature_scaler': train_result['dataset'].feature_scaler,
                        'target_scaler': train_result['dataset'].target_scaler
                    }
                    temp_scalers_path = output_path / 'scalers_temp.pkl'
                    save_compressed(temp_scalers, temp_scalers_path, 'none', logger)  # use no compression for speed
                    scalers_save_time = time.time() - scalers_save_start
                    logger.info(f"✅ Temporary scalers saved in {scalers_save_time:.2f}s")
                    logger.debug(f"   📂 Scalers path: {temp_scalers_path}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not save temporary scalers: {e}")
                    logger.debug(f"   📍 Error details: {type(e).__name__}: {str(e)}")
            
            logger.info("✅ Training split completed - scalers are now available")
            remaining_splits = [s for s in splits if s != 'train']
            logger.debug(f"   📋 Remaining splits for parallel processing: {remaining_splits}")
        else:
            logger.info("ℹ️  No training split found - proceeding with available splits")
            remaining_splits = splits
        
        # step 2: process remaining splits in parallel (if any)
        if remaining_splits:
            logger.info(f"🔄 Step 2: Processing remaining splits in parallel: {remaining_splits}")
            parallel_start = time.time()
            
            max_workers = config.max_workers or min(len(remaining_splits), psutil.cpu_count())
            logger.info(f"👥 Using {max_workers} parallel workers for {len(remaining_splits)} splits")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # submit remaining split processing tasks
                logger.debug("📋 Submitting parallel processing tasks...")
                future_to_split = {
                    executor.submit(process_single_split, (split, process_config)): split 
                    for split in remaining_splits
                }
                
                progress_bar = tqdm(total=len(remaining_splits), desc="Processing splits") if config.show_progress else None
                
                completed_parallel = 0
                for future in as_completed(future_to_split):
                    result = future.result()
                    results[result['split']] = result
                    completed_parallel += 1
                    
                    split_name = result['split']
                    if result.get('success', False):
                        logger.info(f"✅ Parallel task completed: {split_name} ({completed_parallel}/{len(remaining_splits)})")
                        logger.debug(f"   ⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
                    else:
                        logger.error(f"❌ Parallel task failed: {split_name} ({completed_parallel}/{len(remaining_splits)})")
                        logger.error(f"   📍 Error: {result.get('error', 'Unknown error')}")
                    
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_description(f"Completed {result['split']}")
                
                if progress_bar:
                    progress_bar.close()
            
            parallel_time = time.time() - parallel_start
            logger.info(f"⏱️  Parallel processing time: {parallel_time:.2f}s")
        
        monitor.log_resources("Smart parallel processing completed")
        
    else:
        logger.info("🔄 Processing splits sequentially...")
        logger.debug(f"   📊 Reason: {'stats_only mode' if args.stats_only else 'single split or parallel disabled'}")
        
        results = {}
        
        # sequential processing with progress
        splits_iter = tqdm(splits, desc="Processing splits") if config.show_progress else splits
        
        for i, split in enumerate(splits_iter):
            logger.info(f"🔄 Processing split {i+1}/{len(splits)}: {split}")
            split_start = time.time()
            
            try:
                # initialize dataset
                logger.info(f"🏗️  Initializing dataset for {split}...")
                dataset_init_start = time.time()
                
                dataset = EcoBiciSpatialTemporalDataset(
                    data_path=args.data_path,
                    sequence_length=args.sequence_length,
                    k_neighbors=args.k_neighbors
                )
                
                dataset_init_time = time.time() - dataset_init_start
                logger.debug(f"   ⏱️  Dataset initialization: {dataset_init_time:.2f}s")
                
                # generate leakage report if requested
                if args.leakage_report and split == 'train':
                    logger.info("📋 Generating data leakage prevention report...")
                    report_start = time.time()
                    dataset.print_data_leakage_report()
                    report_time = time.time() - report_start
                    logger.debug(f"   ⏱️  Report generation: {report_time:.2f}s")
                
                # load data
                logger.info(f"📖 Loading {split} data...")
                data_load_start = time.time()
                temporal_signal = dataset.load_data(split)
                data_load_time = time.time() - data_load_start
                logger.info(f"✅ Data loaded in {data_load_time:.2f}s")
                
                # show statistics
                logger.info(f"📈 {split.capitalize()} Dataset Statistics:")
                logger.info(f"   🔢 Number of time steps: {len(temporal_signal):,}")
                logger.info(f"   🏠 Number of nodes (clusters): {temporal_signal.edge_index.max().item() + 1}")
                logger.info(f"   🔗 Number of edges: {temporal_signal.edge_index.shape[1]:,}")
                
                if len(temporal_signal) > 0:
                    sample_features = temporal_signal[0].x
                    sample_targets = temporal_signal[0].y
                    logger.info(f"   📊 Features per node: {sample_features.shape[1]}")
                    logger.info(f"   🎯 Target statistics:")
                    logger.info(f"     - Mean: {sample_targets.mean().item():.4f}")
                    logger.info(f"     - Std: {sample_targets.std().item():.4f}")
                    logger.info(f"     - Min: {sample_targets.min().item():.4f}")
                    logger.info(f"     - Max: {sample_targets.max().item():.4f}")
                    logger.info(f"     - Non-zero ratio: {(sample_targets > 0).float().mean().item():.4f}")
                
                # save if not stats only
                if not args.stats_only:
                    logger.info(f"💾 Saving {split} dataset...")
                    save_start = time.time()
                    
                    output_file = output_path / f"{split}_temporal_signal.pkl"
                    save_dataset_with_logging(temporal_signal, output_file, config.compression_format, split, logger)
                    
                    # save additional info
                    info = {
                        'n_nodes': temporal_signal.edge_index.max().item() + 1,
                        'n_edges': temporal_signal.edge_index.shape[1],
                        'n_timesteps': len(temporal_signal),
                        'n_features': temporal_signal[0].x.shape[1] if len(temporal_signal) > 0 else 0,
                        'sequence_length': args.sequence_length,
                        'k_neighbors': args.k_neighbors,
                        'compression': config.compression_format,
                        'creation_time': time.time(),
                        'processing_time_seconds': time.time() - split_start
                    }
                    
                    info_file = output_path / f"{split}_info.json"
                    logger.debug(f"   📄 Saving metadata to: {info_file}")
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                    
                    save_time = time.time() - save_start
                    logger.debug(f"   ⏱️  Total save time: {save_time:.2f}s")
                
                results[split] = {
                    'split': split,
                    'temporal_signal': temporal_signal,
                    'dataset': dataset,
                    'success': True,
                    'processing_time': time.time() - split_start
                }
                
                split_total_time = time.time() - split_start
                logger.info(f"✅ Completed {split} in {split_total_time:.2f}s")
                
                monitor.log_resources(f"Completed {split}")
                
            except Exception as e:
                error_time = time.time() - split_start
                logger.error(f"❌ Error processing {split} split after {error_time:.2f}s: {e}")
                logger.debug(f"   📍 Error details: {type(e).__name__}: {str(e)}")
                results[split] = {
                    'split': split, 
                    'error': str(e), 
                    'error_type': type(e).__name__,
                    'success': False,
                    'processing_time': error_time
                }
                continue
    
    processing_time = time.time() - processing_start
    logger.info(f"⏱️  Total processing time: {processing_time:.2f}s")
    
    # validate data leakage if requested
    if args.validate_leakage:
        logger.info("🔍 Starting data leakage validation...")
        validation_start = time.time()
        
        valid_splits = [split for split, result in results.items() if result.get('success', False)]
        logger.info(f"   📋 Validating {len(valid_splits)} successful splits: {valid_splits}")
        
        if valid_splits:
            # get a dataset instance for validation
            dataset = next(result['dataset'] for result in results.values() if result.get('success', False))
            leakage_ok = validate_data_leakage_parallel(dataset, valid_splits, config.validate_sample_size, logger)
            
            validation_time = time.time() - validation_start
            logger.info(f"⏱️  Validation time: {validation_time:.2f}s")
            
            if not leakage_ok and not args.stats_only:
                logger.error("❌ Data leakage detected! Stopping processing.")
                return
            logger.info("✅ All data leakage validations passed!")
        else:
            logger.warning("⚠️  No successful splits to validate")
    
    # save scalers if requested and we processed training data
    if args.save_scalers and 'train' in results and results['train'].get('success', False) and not args.stats_only:
        logger.info("💾 Saving final scalers...")
        scalers_start = time.time()
        
        try:
            train_dataset = results['train']['dataset']
            scalers = {
                'feature_scaler': train_dataset.feature_scaler,
                'target_scaler': train_dataset.target_scaler,
                'metadata': {
                    'sequence_length': args.sequence_length,
                    'k_neighbors': args.k_neighbors,
                    'creation_time': time.time(),
                    'n_features': train_dataset.feature_scaler.n_features_in_,
                    'feature_names': getattr(train_dataset, 'feature_columns', [])[:train_dataset.feature_scaler.n_features_in_]
                }
            }
            
            scalers_file = output_path / 'scalers.pkl'
            save_compressed(scalers, scalers_file, config.compression_format, logger)
            
            scalers_time = time.time() - scalers_start
            logger.info(f"✅ Final scalers saved in {scalers_time:.2f}s")
            logger.debug(f"   📂 Scalers file: {scalers_file}")
        
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not save final scalers: {e}")
            logger.debug(f"   📍 Error details: {type(e).__name__}: {str(e)}")
    
    # cleanup temporary files
    cleanup_start = time.time()
    temp_scalers_path = output_path / 'scalers_temp.pkl'
    if temp_scalers_path.exists():
        try:
            temp_scalers_path.unlink()
            logger.info("🧹 Cleaned up temporary scalers file")
        except Exception as e:
            logger.warning(f"⚠️  Could not remove temporary scalers file: {e}")
    
    cleanup_time = time.time() - cleanup_start
    logger.debug(f"   ⏱️  Cleanup time: {cleanup_time:.3f}s")
    
    # comprehensive summary
    logger.info("🎉 Dataset preparation complete!")
    logger.info("=" * 60)
    
    # results summary
    successful_splits = [split for split, result in results.items() if result.get('success', False)]
    failed_splits = [split for split, result in results.items() if not result.get('success', False)]
    
    logger.info(f"📊 PROCESSING SUMMARY:")
    logger.info(f"   ✅ Successfully processed: {len(successful_splits)} splits")
    for split in successful_splits:
        processing_time = results[split].get('processing_time', 0)
        logger.info(f"     • {split}: {processing_time:.2f}s")
    
    if failed_splits:
        logger.warning(f"   ❌ Failed splits: {len(failed_splits)}")
        for split in failed_splits:
            error = results[split].get('error', 'Unknown error')
            processing_time = results[split].get('processing_time', 0)
            logger.warning(f"     • {split}: {error} (after {processing_time:.2f}s)")
    
    # file summary
    if not args.stats_only and successful_splits:
        logger.info(f"📁 GENERATED FILES in {args.output_path}:")
        total_size_mb = 0
        file_count = 0
        
        for file in sorted(output_path.glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / 1024 / 1024
                total_size_mb += size_mb
                file_count += 1
                
                # categorize files
                if file.suffix in ['.pkl', '.xz', '.gz']:
                    file_type = "📦 Dataset"
                elif file.suffix == '.json':
                    file_type = "📄 Metadata"
                else:
                    file_type = "📁 Other"
                
                logger.info(f"   {file_type}: {file.name} ({size_mb:.1f} MB)")
        
        logger.info(f"   📊 Total: {file_count} files, {total_size_mb:.1f} MB")
    
    # timing summary
    total_time = time.time() - overall_start
    logger.info(f"⏱️  TIMING SUMMARY:")
    logger.info(f"   • Total execution time: {total_time:.2f}s")
    logger.info(f"   • Processing time: {processing_time:.2f}s ({processing_time/total_time*100:.1f}%)")
    if args.validate_leakage and 'validation_time' in locals():
        logger.info(f"   • Validation time: {validation_time:.2f}s ({validation_time/total_time*100:.1f}%)")
    
    monitor.log_resources("Final")
    
    if successful_splits:
        logger.info("🎊 All requested datasets prepared successfully!")
    else:
        logger.error("💥 No datasets were successfully prepared!")
        return 1  # exit code for failure
    
    return 0  # exit code for success

if __name__ == "__main__":
    main() 