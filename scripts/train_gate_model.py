#!/usr/bin/env python3
"""
Command-line script for training Gate models.
"""
import argparse
import json
import sys
from pathlib import Path

# add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training.gate_trainer import GateTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Gate models for EcoBici demand prediction",
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
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def load_hyperparameters(hyperparameters_arg):
    """Load hyperparameters from JSON string or file."""
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
    """Load experiment configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Experiment config file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in experiment config file: {config_path}")


def override_hyperparameters(hyperparameters, args):
    """Override hyperparameters with command line arguments."""
    if hyperparameters is None:
        hyperparameters = {}
    
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
    
    return hyperparameters


def main():
    """Main function."""
    args = parse_args()
    
    print("=== EcoBici Gate Model Training ===")
    print(f"Model type: {args.model_type}")
    print(f"Training data: {args.train_path}")
    print(f"Validation data: {args.val_path}")
    print(f"Test data: {args.test_path}")
    print(f"Output directory: {args.output_dir}")
    
    # create trainer
    trainer = GateTrainer(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        feature_groups=args.feature_groups
    )
    
    # prepare data
    trainer.prepare_data()
    
    # experiment mode
    if args.experiment_config:
        print(f"\nRunning experiments from config: {args.experiment_config}")
        experiment_config = load_experiment_config(args.experiment_config)
        results = trainer.run_experiment(experiment_config, save_models=not args.no_save)
        return
    
    # single model training mode
    # load hyperparameters
    hyperparameters = load_hyperparameters(args.hyperparameters)
    hyperparameters = override_hyperparameters(hyperparameters, args)
    
    if hyperparameters:
        print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    
    # train model
    trainer.train(
        model_type=args.model_type,
        hyperparameters=hyperparameters
    )
    
    # get feature importance
    trainer.get_feature_importance()
    
    # save model
    if not args.no_save:
        trainer.save_model(args.model_name)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main() 