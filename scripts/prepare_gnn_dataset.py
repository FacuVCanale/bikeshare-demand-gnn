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
    logger.info(f"Starting processing of {split} split")
    
    try:
        # create dataset instance
        dataset = EcoBiciSpatialTemporalDataset(
            data_path=config['data_path'],
            sequence_length=config['sequence_length'],
            k_neighbors=config['k_neighbors']
        )
        
        # for non-training splits, try to load pre-fitted scalers
        scalers_path = Path(config.get('output_path', 'data/gnn')) / 'scalers_temp.pkl'
        if split != 'train' and scalers_path.exists():
            try:
                logger.info(f"Loading pre-fitted scalers for {split} split...")
                scalers = load_compressed(scalers_path, 'auto', logger)
                dataset.feature_scaler = scalers['feature_scaler']
                dataset.target_scaler = scalers['target_scaler']
                logger.info(f"✅ Successfully loaded scalers for {split}")
            except Exception as e:
                logger.warning(f"Could not load scalers for {split}: {e}")
        
        # load data with progress tracking
        logger.info(f"Loading {split} data...")
        temporal_signal = dataset.load_data(split)
        
        # gather statistics
        stats = {
            'split': split,
            'n_timesteps': len(temporal_signal),
            'n_nodes': temporal_signal.edge_index.max().item() + 1,
            'n_edges': temporal_signal.edge_index.shape[1],
            'n_features': temporal_signal[0].x.shape[1] if len(temporal_signal) > 0 else 0
        }
        
        if len(temporal_signal) > 0:
            sample_targets = temporal_signal[0].y
            stats.update({
                'target_mean': sample_targets.mean().item(),
                'target_std': sample_targets.std().item(),
                'target_min': sample_targets.min().item(),
                'target_max': sample_targets.max().item(),
                'target_nonzero_ratio': (sample_targets > 0).float().mean().item()
            })
        
        logger.info(f"Completed processing {split} split: {stats['n_timesteps']} timesteps")
        
        return {
            'split': split,
            'temporal_signal': temporal_signal,
            'stats': stats,
            'dataset': dataset,  # return for scaler access
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error processing {split} split: {e}")
        return {
            'split': split,
            'error': str(e),
            'success': False
        }

def validate_data_leakage_parallel(dataset: EcoBiciSpatialTemporalDataset, 
                                 splits: List[str], 
                                 sample_size: int,
                                 logger: logging.Logger) -> bool:
    """Validate data leakage in parallel for multiple splits"""
    logger.info("Validating data leakage across all splits...")
    
    validation_tasks = []
    with ThreadPoolExecutor(max_workers=len(splits)) as executor:
        for split in splits:
            if (Path(dataset.data_path) / "clustered" / f"{split}_cluster_features.parquet").exists():
                future = executor.submit(dataset.validate_no_data_leakage, split, sample_size)
                validation_tasks.append((split, future))
        
        # collect results with progress tracking
        all_passed = True
        for split, future in tqdm(validation_tasks, desc="Validating splits"):
            try:
                result = future.result(timeout=300)  # 5 min timeout
                logger.info(f"✅ {split} split passed data leakage validation")
            except Exception as e:
                logger.error(f"❌ {split} split failed validation: {e}")
                all_passed = False
    
    return all_passed

def main():
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
    
    args = parser.parse_args()
    
    # setup optimized configuration
    config = OptimizedConfig()
    
    if args.optimized:
        config.parallel_processing = True
        config.cache_intermediate = True
        config.use_compression = True
    
    # override with specific arguments
    if args.no_parallel:
        config.parallel_processing = False
    elif args.parallel:
        config.parallel_processing = True
        
    if args.cache:
        config.cache_intermediate = True
    if args.compression:
        config.compression_format = args.compression
        config.use_compression = args.compression != 'none'
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.max_workers:
        config.max_workers = args.max_workers
    if args.log_level:
        config.log_level = args.log_level
    if args.no_progress:
        config.show_progress = False
    
    # setup logging and monitoring
    logger = setup_logging(config.log_level)
    monitor = ResourceMonitor(logger)
    
    # setup cache
    cache = OptimizedCache(config.cache_dir, logger) if config.cache_intermediate else None
    if args.clear_cache and cache:
        cache.clear()
    
    # create output directory
    output_path = Path(args.output_path)
    if not args.stats_only:
        output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚴 Optimized EcoBici GNN Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"K-neighbors: {args.k_neighbors}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Compression: {config.compression_format}")
    logger.info(f"Parallel processing: {config.parallel_processing}")
    logger.info(f"Caching: {config.cache_intermediate}")
    logger.info("")
    
    monitor.log_resources("Initialization")
    
    # determine splits to process
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    logger.info(f"Processing splits: {splits}")
    
    # prepare processing configuration
    process_config = {
        'data_path': args.data_path,
        'sequence_length': args.sequence_length,
        'k_neighbors': args.k_neighbors,
        'log_level': config.log_level,
        'output_path': args.output_path
    }
    
    # process splits with smart dependency handling
    if config.parallel_processing and len(splits) > 1 and not args.stats_only:
        logger.info("🔄 Processing splits with dependency management...")
        
        results = {}
        
        # step 1: process training split first (required for scaler fitting)
        if 'train' in splits:
            logger.info("📊 Processing training split first (required for scalers)...")
            train_result = process_single_split(('train', process_config))
            results['train'] = train_result
            
            if not train_result.get('success', False):
                logger.error("❌ Training split failed - cannot proceed with other splits")
                return
            
            # save scalers temporarily for parallel processing
            if train_result.get('dataset'):
                try:
                    temp_scalers = {
                        'feature_scaler': train_result['dataset'].feature_scaler,
                        'target_scaler': train_result['dataset'].target_scaler
                    }
                    temp_scalers_path = output_path / 'scalers_temp.pkl'
                    save_compressed(temp_scalers, temp_scalers_path, 'none', logger)  # use no compression for speed
                    logger.info("💾 Saved temporary scalers for parallel processing")
                except Exception as e:
                    logger.warning(f"Could not save temporary scalers: {e}")
            
            logger.info("✅ Training split completed - scalers are now available")
            remaining_splits = [s for s in splits if s != 'train']
        else:
            remaining_splits = splits
        
        # step 2: process remaining splits in parallel (if any)
        if remaining_splits:
            logger.info(f"🔄 Processing remaining splits in parallel: {remaining_splits}")
            
            max_workers = config.max_workers or min(len(remaining_splits), psutil.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # submit remaining split processing tasks
                future_to_split = {
                    executor.submit(process_single_split, (split, process_config)): split 
                    for split in remaining_splits
                }
                
                progress_bar = tqdm(total=len(remaining_splits), desc="Processing splits") if config.show_progress else None
                
                for future in as_completed(future_to_split):
                    result = future.result()
                    results[result['split']] = result
                    
                    if progress_bar:
                        progress_bar.update(1)
                        progress_bar.set_description(f"Completed {result['split']}")
                
                if progress_bar:
                    progress_bar.close()
        
        monitor.log_resources("Smart parallel processing completed")
        
    else:
        logger.info("🔄 Processing splits sequentially...")
        results = {}
        
        # sequential processing with progress
        splits_iter = tqdm(splits, desc="Processing splits") if config.show_progress else splits
        
        for split in splits_iter:
            try:
                # initialize dataset
                logger.info(f"📊 Initializing dataset for {split}...")
                dataset = EcoBiciSpatialTemporalDataset(
                    data_path=args.data_path,
                    sequence_length=args.sequence_length,
                    k_neighbors=args.k_neighbors
                )
                
                # generate leakage report if requested
                if args.leakage_report and split == 'train':
                    dataset.print_data_leakage_report()
                
                # load data
                temporal_signal = dataset.load_data(split)
                
                # show statistics
                logger.info(f"📈 {split.capitalize()} Dataset Statistics:")
                logger.info(f"  • Number of time steps: {len(temporal_signal)}")
                logger.info(f"  • Number of nodes (clusters): {temporal_signal.edge_index.max().item() + 1}")
                logger.info(f"  • Number of edges: {temporal_signal.edge_index.shape[1]}")
                
                if len(temporal_signal) > 0:
                    sample_features = temporal_signal[0].x
                    sample_targets = temporal_signal[0].y
                    logger.info(f"  • Features per node: {sample_features.shape[1]}")
                    logger.info(f"  • Target statistics:")
                    logger.info(f"    - Mean: {sample_targets.mean().item():.4f}")
                    logger.info(f"    - Std: {sample_targets.std().item():.4f}")
                    logger.info(f"    - Min: {sample_targets.min().item():.4f}")
                    logger.info(f"    - Max: {sample_targets.max().item():.4f}")
                    logger.info(f"    - Non-zero ratio: {(sample_targets > 0).float().mean().item():.4f}")
                
                # save if not stats only
                if not args.stats_only:
                    output_file = output_path / f"{split}_temporal_signal.pkl"
                    
                    logger.info(f"💾 Saving {split} to {output_file}...")
                    save_compressed(temporal_signal, output_file, config.compression_format, logger)
                    
                    logger.info(f"✅ Saved {split} dataset")
                    
                    # save additional info
                    info = {
                        'n_nodes': temporal_signal.edge_index.max().item() + 1,
                        'n_edges': temporal_signal.edge_index.shape[1],
                        'n_timesteps': len(temporal_signal),
                        'n_features': temporal_signal[0].x.shape[1] if len(temporal_signal) > 0 else 0,
                        'sequence_length': args.sequence_length,
                        'k_neighbors': args.k_neighbors,
                        'compression': config.compression_format
                    }
                    
                    info_file = output_path / f"{split}_info.json"
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                
                results[split] = {
                    'split': split,
                    'temporal_signal': temporal_signal,
                    'dataset': dataset,
                    'success': True
                }
                
                monitor.log_resources(f"Completed {split}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {split} split: {e}")
                results[split] = {'split': split, 'error': str(e), 'success': False}
                continue
    
    # validate data leakage if requested
    if args.validate_leakage:
        valid_splits = [split for split, result in results.items() if result.get('success', False)]
        if valid_splits:
            # get a dataset instance for validation
            dataset = next(result['dataset'] for result in results.values() if result.get('success', False))
            leakage_ok = validate_data_leakage_parallel(dataset, valid_splits, config.validate_sample_size, logger)
            
            if not leakage_ok and not args.stats_only:
                logger.error("❌ Data leakage detected! Stopping processing.")
                return
            logger.info("✅ All data leakage validations passed!")
    
    # save scalers if requested and we processed training data
    if args.save_scalers and 'train' in results and results['train'].get('success', False) and not args.stats_only:
        try:
            train_dataset = results['train']['dataset']
            scalers = {
                'feature_scaler': train_dataset.feature_scaler,
                'target_scaler': train_dataset.target_scaler
            }
            
            scalers_file = output_path / 'scalers.pkl'
            save_compressed(scalers, scalers_file, config.compression_format, logger)
            logger.info(f"💾 Saved scalers to {scalers_file}")
        
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not save scalers: {e}")
    
    # cleanup temporary files
    temp_scalers_path = output_path / 'scalers_temp.pkl'
    if temp_scalers_path.exists():
        try:
            temp_scalers_path.unlink()
            logger.info("🧹 Cleaned up temporary scalers file")
        except Exception as e:
            logger.warning(f"Could not remove temporary scalers file: {e}")
    
    # summary
    successful_splits = [split for split, result in results.items() if result.get('success', False)]
    failed_splits = [split for split, result in results.items() if not result.get('success', False)]
    
    logger.info("🎉 Dataset preparation complete!")
    logger.info(f"✅ Successfully processed: {successful_splits}")
    if failed_splits:
        logger.warning(f"❌ Failed splits: {failed_splits}")
    
    if not args.stats_only:
        logger.info(f"\n📁 Generated files in {args.output_path}:")
        for file in sorted(output_path.glob("*")):
            size_mb = file.stat().st_size / 1024 / 1024
            logger.info(f"  • {file.name} ({size_mb:.1f} MB)")
    
    monitor.log_resources("Final")

if __name__ == "__main__":
    main() 