"""
Main Pipeline for Individual Station Bike Prediction

Complete pipeline that orchestrates:
1. Feature Engineering - Convert trip data to delta T format with features
2. Training - Train XGBoost and LightGBM models
3. Evaluation - Comprehensive metrics, plots, and feature importance

Based on the approach document for optimizing individual station predictions.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from feature_engineering import FeatureEngineer, load_and_process_data
from training import ModelTrainer, train_pipeline
from evaluation import ModelEvaluator, evaluate_models


class BikePrediictionPipeline:
    """
    Complete pipeline for bike station prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize the pipeline.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self.get_default_config()
        self.results = {}
        
    def get_default_config(self):
        """
        Get default configuration parameters.
        
        Returns:
            dict: Default configuration
        """
        return {
            # Data parameters
            'data_path': None,  # Must be provided
            'station_id': None,  # None for all stations, int for specific station
            'delta_t_minutes': 30,
            
            # Feature engineering
            'target_cols': ['arrivals', 'departures'],
            
            # Training parameters
            'model_types': ['xgboost', 'lightgbm'],
            'test_size': 0.2,
            'validation_size': 0.2,
            'scale_features': True,
            'use_time_split': True,
            'random_state': 42,
            
            # Output parameters
            'output_dir': 'results',
            'save_models': True,
            'save_plots': True,
            
            # Evaluation parameters
            'generate_plots': True,
            'feature_importance_top_n': 20,
        }
    
    def validate_config(self):
        """
        Validate configuration parameters.
        """
        if self.config['data_path'] is None:
            raise ValueError("data_path must be provided in configuration")
        
        if not os.path.exists(self.config['data_path']):
            raise FileNotFoundError(f"Data file not found: {self.config['data_path']}")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        print("Configuration validated successfully")
    
    def run_feature_engineering(self):
        """
        Run the feature engineering step.
        
        Returns:
            tuple: (processed_df, feature_engineer)
        """
        print(f"\n{'='*60}")
        print("STEP 1: FEATURE ENGINEERING")
        print(f"{'='*60}")
        
        # Load and process data
        processed_df, feature_engineer = load_and_process_data(
            data_path=self.config['data_path'],
            station_id=self.config['station_id'],
            delta_t_minutes=self.config['delta_t_minutes']
        )
        
        # Prepare for training
        X, y, metadata = feature_engineer.prepare_for_training(
            processed_df, 
            target_cols=self.config['target_cols']
        )
        
        # Store results
        self.results['feature_engineering'] = {
            'processed_df': processed_df,
            'feature_engineer': feature_engineer,
            'X': X,
            'y': y,
            'metadata': metadata,
            'feature_names': feature_engineer.get_feature_names()
        }
        
        # Save processed data
        if self.config['save_models']:
            data_path = os.path.join(self.config['output_dir'], 'processed_data.csv')
            processed_df.to_csv(data_path, index=False)
            print(f"Saved processed data: {data_path}")
        
        print(f"Feature engineering completed successfully")
        print(f"  Dataset shape: {X.shape}")
        print(f"  Features: {len(feature_engineer.get_feature_names())}")
        print(f"  Targets: {self.config['target_cols']}")
        
        return processed_df, feature_engineer
    
    def run_training(self):
        """
        Run the model training step.
        
        Returns:
            tuple: (trainer, training_results, best_models)
        """
        print(f"\n{'='*60}")
        print("STEP 2: MODEL TRAINING")
        print(f"{'='*60}")
        
        if 'feature_engineering' not in self.results:
            raise RuntimeError("Feature engineering must be run before training")
        
        fe_results = self.results['feature_engineering']
        X, y, metadata = fe_results['X'], fe_results['y'], fe_results['metadata']
        
        # Run training pipeline
        models_dir = os.path.join(self.config['output_dir'], 'models')
        
        trainer, training_results, best_models = train_pipeline(
            X=X,
            y=y,
            metadata=metadata,
            target_cols=self.config['target_cols'],
            model_types=self.config['model_types'],
            save_models=self.config['save_models'],
            save_dir=models_dir
        )
        
        # Store results
        self.results['training'] = {
            'trainer': trainer,
            'training_results': training_results,
            'best_models': best_models,
            'data_splits': trainer.prepare_data(X, y, metadata,
                                               test_size=self.config['test_size'],
                                               validation_size=self.config['validation_size'],
                                               scale_features=self.config['scale_features'],
                                               use_time_split=self.config['use_time_split'])
        }
        
        print(f"Training completed successfully")
        print(f"  Models trained: {len(training_results)}")
        print(f"  Best models identified: {len(best_models)}")
        
        return trainer, training_results, best_models
    
    def run_evaluation(self):
        """
        Run the model evaluation step.
        
        Returns:
            tuple: (evaluator, evaluation_results)
        """
        print(f"\n{'='*60}")
        print("STEP 3: MODEL EVALUATION")
        print(f"{'='*60}")
        
        if 'training' not in self.results:
            raise RuntimeError("Training must be run before evaluation")
        
        training_results = self.results['training']
        
        # Extract models from training results
        models_dict = {}
        for model_key, result in training_results['training_results'].items():
            models_dict[model_key] = result['model']
        
        # Run evaluation
        eval_dir = os.path.join(self.config['output_dir'], 'evaluation')
        
        evaluator, evaluation_results = evaluate_models(
            models_dict=models_dict,
            data_splits=training_results['data_splits'],
            target_cols=self.config['target_cols'],
            save_dir=eval_dir
        )
        
        # Store results
        self.results['evaluation'] = {
            'evaluator': evaluator,
            'evaluation_results': evaluation_results
        }
        
        print(f"Evaluation completed successfully")
        print(f"  Results saved to: {eval_dir}")
        
        return evaluator, evaluation_results
    
    def run_complete_pipeline(self):
        """
        Run the complete pipeline from start to finish.
        
        Returns:
            dict: Complete pipeline results
        """
        start_time = datetime.now()
        
        print(f"\n{'='*80}")
        print("INDIVIDUAL STATION BIKE PREDICTION PIPELINE")
        print(f"{'='*80}")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        
        try:
            # Validate configuration
            self.validate_config()
            
            # Step 1: Feature Engineering
            processed_df, feature_engineer = self.run_feature_engineering()
            
            # Step 2: Training
            trainer, training_results, best_models = self.run_training()
            
            # Step 3: Evaluation
            evaluator, evaluation_results = self.run_evaluation()
            
            # Pipeline completed successfully
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Total duration: {duration:.2f} seconds")
            print(f"Results saved to: {self.config['output_dir']}")
            
            # Final summary
            self._print_final_summary()
            
            return self.results
            
        except Exception as e:
            print(f"\n{'='*80}")
            print("PIPELINE FAILED")
            print(f"{'='*80}")
            print(f"Error: {str(e)}")
            raise e
    
    def _print_final_summary(self):
        """
        Print a final summary of pipeline results.
        """
        print(f"\n{'='*60}")
        print("FINAL PIPELINE SUMMARY")
        print(f"{'='*60}")
        
        if 'feature_engineering' in self.results:
            fe_results = self.results['feature_engineering']
            print(f"Data processed: {fe_results['X'].shape[0]} samples")
            print(f"Features created: {len(fe_results['feature_names'])}")
        
        if 'training' in self.results:
            training_results = self.results['training']['training_results']
            best_models = self.results['training']['best_models']
            print(f"Models trained: {len(training_results)}")
            print(f"Best models:")
            for target, model_info in best_models.items():
                print(f"  {target}: {model_info['model_key']} "
                      f"(RMSE: {model_info['metric_value']:.4f})")
        
        if 'evaluation' in self.results:
            eval_results = self.results['evaluation']['evaluation_results']
            print(f"Evaluation completed with comprehensive analysis")
            print(f"Results directory: {eval_results['save_dir']}")


def create_config_from_args():
    """
    Create configuration from command line arguments.
    
    Returns:
        dict: Configuration dictionary
    """
    parser = argparse.ArgumentParser(description='Individual Station Bike Prediction Pipeline')
    
    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the input data file (CSV or Parquet)')
    
    # Optional arguments
    parser.add_argument('--station_id', type=int, default=None,
                       help='Specific station ID to analyze (default: all stations)')
    parser.add_argument('--delta_t_minutes', type=int, default=30,
                       help='Time interval in minutes (default: 30)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--models', nargs='+', default=['xgboost', 'lightgbm'],
                       choices=['xgboost', 'lightgbm'],
                       help='Models to train (default: xgboost lightgbm)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--validation_size', type=float, default=0.2,
                       help='Validation set size (default: 0.2)')
    parser.add_argument('--no_scaling', action='store_true',
                       help='Disable feature scaling')
    parser.add_argument('--random_split', action='store_true',
                       help='Use random split instead of time-based split')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--no_fill_nulls', action='store_true',
                       help='Do not fill null values with zeros (keep nulls for model to handle)')
    parser.add_argument('--fill_strategy', type=str, default='zero',
                       choices=['zero', 'mean', 'median', 'forward_fill'],
                       help='Strategy for filling null values (default: zero)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'data_path': args.data_path,
        'station_id': args.station_id,
        'delta_t_minutes': args.delta_t_minutes,
        'target_cols': ['arrivals', 'departures'],
        'model_types': args.models,
        'test_size': args.test_size,
        'validation_size': args.validation_size,
        'scale_features': not args.no_scaling,
        'use_time_split': not args.random_split,
        'random_state': args.random_state,
        'output_dir': args.output_dir,
        'save_models': True,
        'save_plots': True,
        'generate_plots': True,
        'feature_importance_top_n': 20,
    }
    
    return config


def main():
    """
    Main function for running the pipeline from command line.
    """
    try:
        # Create configuration from command line arguments
        config = create_config_from_args()
        
        # Initialize and run pipeline
        pipeline = BikePrediictionPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        print("\nPipeline execution completed successfully!")
        
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage when run as script
    if len(sys.argv) == 1:
        # No command line arguments - show example usage
        print("Individual Station Bike Prediction Pipeline")
        print("=" * 50)
        print("\nExample usage:")
        print("python main.py --data_path data/trips_with_weather.parquet")
        print("python main.py --data_path data/trips_with_weather.parquet --station_id 123")
        print("python main.py --data_path data/trips_with_weather.parquet --output_dir my_results")
        print("\nFor full help: python main.py --help")
        
        # Show example programmatic usage
        print("\nExample programmatic usage:")
        print("""
from main import BikePrediictionPipeline

# Configure pipeline
config = {
    'data_path': 'data/trips_with_weather.parquet',  # Supports .parquet or .csv
    'station_id': None,  # All stations
    'output_dir': 'results'
}

# Run pipeline
pipeline = BikePrediictionPipeline(config)
results = pipeline.run_complete_pipeline()
""")
    else:
        # Run main function with command line arguments
        main() 