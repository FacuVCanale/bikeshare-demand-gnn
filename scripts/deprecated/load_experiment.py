#!/usr/bin/env python3
"""
Script to load and inspect saved GNN experiments.

This script demonstrates how to load trained models, architecture information,
and training history from experiment directories.
"""

import argparse
import json
from pathlib import Path
import sys

# add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_loader import (
    load_model_from_experiment,
    load_model_architecture,
    load_training_history,
    get_experiment_summary,
    recreate_model_from_architecture
)
from src.models.gnn_models import print_model_summary


def main():
    parser = argparse.ArgumentParser(description='Load and inspect GNN experiments')
    parser.add_argument('experiment_path', type=str,
                        help='Path to the experiment directory')
    parser.add_argument('--summary', action='store_true',
                        help='Show experiment summary')
    parser.add_argument('--architecture', action='store_true',
                        help='Show model architecture details')
    parser.add_argument('--history', action='store_true',
                        help='Show training history summary')
    parser.add_argument('--load-model', action='store_true',
                        help='Load the trained model')
    parser.add_argument('--model-file', type=str, default='final_model.pt',
                        help='Model file to load (default: final_model.pt)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to load model on')
    parser.add_argument('--all', action='store_true',
                        help='Show all available information')
    
    args = parser.parse_args()
    
    experiment_path = Path(args.experiment_path)
    
    if not experiment_path.exists():
        print(f"Error: Experiment path does not exist: {experiment_path}")
        return 1
    
    # if --all is specified, enable all options
    if args.all:
        args.summary = True
        args.architecture = True
        args.history = True
        args.load_model = True
    
    # if no specific options, show summary by default
    if not any([args.summary, args.architecture, args.history, args.load_model]):
        args.summary = True
    
    print(f"Inspecting experiment: {experiment_path}")
    print("=" * 80)
    
    # show experiment summary
    if args.summary:
        print("\n📊 EXPERIMENT SUMMARY")
        print("-" * 40)
        try:
            summary = get_experiment_summary(experiment_path)
            
            print(f"Experiment Name: {summary['experiment_name']}")
            print(f"Experiment Path: {summary['experiment_path']}")
            
            print("\nFiles Present:")
            for file_info in summary['files_present']:
                status = "✅" if file_info['exists'] else "❌"
                size_str = f"({file_info['size_mb']:.2f} MB)" if file_info['exists'] else ""
                print(f"  {status} {file_info['filename']} {size_str}")
            
            if summary['model_info']:
                print(f"\nModel Information:")
                model_info = summary['model_info']
                print(f"  Model Class: {model_info['model_class']}")
                print(f"  Total Parameters: {model_info['total_parameters']:,}")
                print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
                
                print(f"  Model Configuration:")
                for param, value in model_info['initialization_parameters'].items():
                    print(f"    {param}: {value}")
            
            if summary['final_metrics']:
                print(f"\nTraining Results:")
                metrics = summary['final_metrics']
                print(f"  Total Epochs: {metrics['total_epochs']}")
                print(f"  Final Train Loss: {metrics['final_train_loss']:.6f}")
                print(f"  Final Val Loss: {metrics['final_val_loss']:.6f}")
                print(f"  Best Val Loss: {metrics['best_val_loss']:.6f}")
        
        except Exception as e:
            print(f"Error loading experiment summary: {e}")
    
    # show detailed architecture
    if args.architecture:
        print("\n🏗️  MODEL ARCHITECTURE")
        print("-" * 40)
        try:
            architecture = load_model_architecture(experiment_path)
            
            print(f"Model Class: {architecture['model_class']}")
            print(f"Model Module: {architecture['model_module']}")
            print(f"Total Parameters: {architecture['total_parameters']:,}")
            print(f"Trainable Parameters: {architecture['trainable_parameters']:,}")
            
            print(f"\nInitialization Parameters:")
            for param, value in architecture['initialization_parameters'].items():
                print(f"  {param}: {value}")
            
            print(f"\nLayer Information:")
            for layer in architecture['layers'][:10]:  # show first 10 layers
                print(f"  {layer['name']}: {layer['type']} ({layer['parameters']:,} params)")
            
            if len(architecture['layers']) > 10:
                print(f"  ... and {len(architecture['layers']) - 10} more layers")
            
            print(f"\nModel Structure:")
            print(architecture['model_structure'])
        
        except Exception as e:
            print(f"Error loading architecture: {e}")
    
    # show training history summary
    if args.history:
        print("\n📈 TRAINING HISTORY")
        print("-" * 40)
        try:
            history = load_training_history(experiment_path)
            
            if history.get('train_loss'):
                train_losses = history['train_loss']
                val_losses = history.get('val_loss', [])
                
                print(f"Total Epochs: {len(train_losses)}")
                print(f"Initial Train Loss: {train_losses[0]:.6f}")
                print(f"Final Train Loss: {train_losses[-1]:.6f}")
                
                if val_losses:
                    print(f"Initial Val Loss: {val_losses[0]:.6f}")
                    print(f"Final Val Loss: {val_losses[-1]:.6f}")
                    print(f"Best Val Loss: {min(val_losses):.6f}")
                    print(f"Best Epoch: {val_losses.index(min(val_losses)) + 1}")
                
                # show training metrics for last epoch
                if history.get('train_metrics') and history['train_metrics']:
                    last_train_metrics = history['train_metrics'][-1]
                    print(f"\nFinal Training Metrics:")
                    for metric, value in last_train_metrics.items():
                        print(f"  Train {metric.upper()}: {value:.6f}")
                
                if history.get('val_metrics') and history['val_metrics']:
                    last_val_metrics = history['val_metrics'][-1]
                    print(f"\nFinal Validation Metrics:")
                    for metric, value in last_val_metrics.items():
                        print(f"  Val {metric.upper()}: {value:.6f}")
        
        except Exception as e:
            print(f"Error loading training history: {e}")
    
    # load and inspect model
    if args.load_model:
        print("\n🤖 LOADING MODEL")
        print("-" * 40)
        try:
            model = load_model_from_experiment(
                experiment_path,
                model_filename=args.model_file,
                device=args.device
            )
            
            print(f"✅ Successfully loaded model: {model.__class__.__name__}")
            print(f"Device: {next(model.parameters()).device}")
            print(f"Model in eval mode: {not model.training}")
            
            # get input shape for model summary
            try:
                architecture = load_model_architecture(experiment_path)
                num_features = architecture['initialization_parameters'].get('num_features')
                if num_features:
                    print(f"\nModel Summary:")
                    print_model_summary(model, (1000, num_features))  # example with 1000 nodes
            except:
                print("Could not load architecture for model summary")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    print("\n" + "=" * 80)
    return 0


if __name__ == '__main__':
    exit(main()) 