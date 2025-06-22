#!/usr/bin/env python3
"""
Optimized command-line script for training Gate models with enhanced performance features.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training.gate_trainer import OptimizedGateTrainer


def parse_args():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description="Optimized Gate model training for EcoBici demand prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # data paths
    parser.add_argument(
        '--train-path',
        type=str,
        default='data/clustered/train_cluster_features.parquet',
        help='Path to training data parquet file'
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default='data/clustered/val_cluster_features.parquet',
        help='Path to validation data parquet file'
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default='data/clustered/test_cluster_features.parquet',
        help='Path to test data parquet file'
    )
    
    # model configuration
    parser.add_argument(
        '--model-type',
        type=str,
        default='catboost',
        choices=['catboost', 'lightgbm', 'xgboost'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--hyperparameters',
        type=str,
        help='JSON string or path to JSON file with hyperparameters'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of iterations (overrides hyperparameters)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (overrides hyperparameters)'
    )
    parser.add_argument(
        '--depth',
        type=int,
        help='Tree depth (overrides hyperparameters)'
    )
    
    # feature selection
    parser.add_argument(
        '--feature-groups',
        type=str,
        nargs='+',
        help='Feature groups to include (e.g., temporal_base weather_base)'
    )
    
    # performance options
    parser.add_argument(
        '--memory-efficient',
        action='store_true',
        default=True,
        help='Enable memory efficient mode with monitoring'
    )
    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help='Disable early stopping'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarking with multiple models'
    )
    
    # output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/gate',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name for the saved model (auto-generated if not provided)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )
    parser.add_argument(
        '--plot-importance',
        action='store_true',
        help='Generate feature importance plots'
    )
    
    # experiment mode
    parser.add_argument(
        '--experiment-config',
        type=str,
        help='Path to JSON file with multiple experiment configurations'
    )
    
    # other options
    parser.add_argument(
        '--target-col',
        type=str,
        default='has_activity',
        help='Name of target column'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU training (disable GPU)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimize output (overrides verbose)'
    )
    
    return parser.parse_args()


def load_hyperparameters(hyperparameters_arg):
    """Load hyperparameters from JSON string or file with error handling."""
    if not hyperparameters_arg:
        return None
    
    # try to parse as JSON string first
    try:
        return json.loads(hyperparameters_arg)
    except json.JSONDecodeError:
        pass
    
    # try to load from file
    try:
        with open(hyperparameters_arg, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Hyperparameters file not found: {hyperparameters_arg}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in hyperparameters file: {hyperparameters_arg}")


def load_experiment_config(config_path):
    """Load experiment configuration from JSON file with validation."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # validate config structure
        if not isinstance(config, dict):
            raise ValueError("Experiment config must be a dictionary")
        
        for exp_name, exp_config in config.items():
            if 'model_type' not in exp_config:
                raise ValueError(f"Missing 'model_type' in experiment '{exp_name}'")
        
        return config
    except FileNotFoundError:
        raise ValueError(f"Experiment config file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in experiment config file: {config_path}")


def override_hyperparameters(hyperparameters, args):
    """Override hyperparameters with command line arguments."""
    if hyperparameters is None:
        hyperparameters = {}
    
    # override specific parameters
    if args.iterations is not None:
        if args.model_type == 'catboost':
            hyperparameters['iterations'] = args.iterations
        else:
            hyperparameters['n_estimators'] = args.iterations
    
    if args.learning_rate is not None:
        hyperparameters['learning_rate'] = args.learning_rate
    
    if args.depth is not None:
        if args.model_type == 'catboost':
            hyperparameters['depth'] = args.depth
        else:
            hyperparameters['max_depth'] = args.depth
    
    # force CPU if requested
    if args.force_cpu:
        print("🔧 Forcing CPU training (GPU disabled)")
        if args.model_type == 'catboost':
            hyperparameters['task_type'] = 'CPU'
            hyperparameters.pop('devices', None)
            hyperparameters.pop('gpu_ram_part', None)
        elif args.model_type == 'lightgbm':
            hyperparameters['device'] = 'cpu'
            hyperparameters.pop('gpu_use_dp', None)
            hyperparameters.pop('gpu_platform_id', None)
            hyperparameters.pop('gpu_device_id', None)
        elif args.model_type == 'xgboost':
            hyperparameters['tree_method'] = 'hist'  # CPU version
            hyperparameters.pop('gpu_id', None)
    
    return hyperparameters


def run_benchmark(trainer, verbose=True):
    """Run performance benchmark with multiple model types."""
    if verbose:
        print("\n=== RUNNING PERFORMANCE BENCHMARK ===")
    
    benchmark_configs = {
        'catboost_optimized': {
            'model_type': 'catboost',
            'hyperparameters': {
                'iterations': 500,
                'learning_rate': 0.1,
                'depth': 6,
                'early_stopping_rounds': 50
            }
        },
        'lightgbm_optimized': {
            'model_type': 'lightgbm',
            'hyperparameters': {
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': 6,
                'early_stopping_rounds': 50
            }
        },
        'xgboost_optimized': {
            'model_type': 'xgboost',
            'hyperparameters': {
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': 6,
                'early_stopping_rounds': 50
            }
        }
    }
    
    # run experiments
    results = trainer.run_experiment(benchmark_configs, save_models=True, verbose=verbose)
    
    if verbose:
        print("\n🏁 Benchmark completed!")
        print("Check experiment summary for detailed comparison.")
    
    return results


def validate_paths(train_path, val_path, test_path):
    """Validate that data paths exist."""
    for name, path in [('Training', train_path), ('Validation', val_path), ('Test', test_path)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} data file not found: {path}")


def main():
    """Enhanced main function with comprehensive error handling."""
    args = parse_args()
    
    # set verbosity
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("=== OPTIMIZED ECOBICI GATE MODEL TRAINING ===")
        print(f"Model type: {args.model_type}")
        print(f"Training data: {args.train_path}")
        print(f"Validation data: {args.val_path}")
        print(f"Test data: {args.test_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Memory efficient: {args.memory_efficient}")
        print(f"Early stopping: {not args.no_early_stopping}")
    
    try:
        # validate data paths
        validate_paths(args.train_path, args.val_path, args.test_path)
        
        # create optimized trainer
        trainer = OptimizedGateTrainer(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
            target_col=args.target_col,
            feature_groups=args.feature_groups,
            memory_efficient=args.memory_efficient,
            enable_early_stopping=not args.no_early_stopping
        )
        
        # prepare data
        if verbose:
            print("\n=== DATA PREPARATION ===")
        
        start_time = time.time()
        trainer.prepare_data(verbose=verbose)
        prep_time = time.time() - start_time
        
        if verbose:
            print(f"Data preparation completed in {prep_time:.2f} seconds")
        
        # benchmark mode
        if args.benchmark:
            results = run_benchmark(trainer, verbose=verbose)
            return
        
        # experiment mode
        if args.experiment_config:
            if verbose:
                print(f"\n=== RUNNING EXPERIMENTS ===")
                print(f"Config: {args.experiment_config}")
            
            experiment_config = load_experiment_config(args.experiment_config)
            results = trainer.run_experiment(
                experiment_config, 
                save_models=not args.no_save,
                verbose=verbose
            )
            
            if verbose:
                print("\n✅ All experiments completed!")
            return
        
        # single model training mode
        if verbose:
            print(f"\n=== SINGLE MODEL TRAINING ===")
        
        # load and override hyperparameters
        hyperparameters = load_hyperparameters(args.hyperparameters)
        hyperparameters = override_hyperparameters(hyperparameters, args)
        
        if hyperparameters and verbose:
            print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
        
        # train model
        start_time = time.time()
        trainer.train(
            model_type=args.model_type,
            hyperparameters=hyperparameters,
            verbose=verbose
        )
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\n⏱️  Total training time: {training_time:.2f} seconds")
        
        # get feature importance
        if verbose or args.plot_importance:
            trainer.get_feature_importance(plot=args.plot_importance)
        
        # save model
        if not args.no_save:
            model_path = trainer.save_model(args.model_name)
            if verbose:
                print(f"\n✅ Model saved successfully!")
        
        if verbose:
            print("\n🎉 Training completed successfully!")
    
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 