"""
Evaluation Module for Individual Station Bike Prediction

Comprehensive evaluation including:
- Metrics calculation and comparison
- Visualization plots  
- Feature importance analysis
- Model performance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive evaluation class for bike prediction models.
    """
    
    def __init__(self, models_dict, data_splits, target_cols=['arrivals', 'departures']):
        """
        Initialize the evaluator.
        
        Args:
            models_dict (dict): Dictionary of trained models
            data_splits (dict): Data splits from training
            target_cols (list): Target column names
        """
        self.models = models_dict
        self.data_splits = data_splits
        self.target_cols = target_cols
        self.predictions = {}
        self.metrics = {}
        
    def calculate_comprehensive_metrics(self):
        """
        Calculate comprehensive metrics for all models.
        
        Returns:
            pd.DataFrame: Metrics summary table
        """
        print("Calculating comprehensive metrics...")
        
        metrics_data = []
        
        for model_key, model in self.models.items():
            target = model_key.split('_')[-1]  # Extract target from model key
            model_type = model_key.replace(f'_{target}', '')
            
            for split in ['train', 'val', 'test']:
                X = self.data_splits[split]['X']
                y_true = self.data_splits[split]['y'][target]
                
                # Make predictions
                y_pred = model.predict((X,))
                
                # Store predictions
                pred_key = f"{model_key}_{split}"
                self.predictions[pred_key] = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'target': target,
                    'model_type': model_type,
                    'split': split
                }
                
                # Calculate metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # Additional metrics
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                max_error = np.max(np.abs(y_true - y_pred))
                median_ae = np.median(np.abs(y_true - y_pred))
                
                metrics_data.append({
                    'model': model_key,
                    'model_type': model_type,
                    'target': target,
                    'split': split,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'max_error': max_error,
                    'median_ae': median_ae
                })
        
        self.metrics_df = pd.DataFrame(metrics_data)
        print(f"Calculated metrics for {len(self.metrics_df)} model-split combinations")
        
        return self.metrics_df
    
    def print_metrics_summary(self):
        """
        Print a formatted summary of model performance.
        """
        if not hasattr(self, 'metrics_df'):
            self.calculate_comprehensive_metrics()
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Best models by target and metric
        for target in self.target_cols:
            print(f"\n--- {target.upper()} PREDICTIONS ---")
            target_metrics = self.metrics_df[
                (self.metrics_df['target'] == target) & 
                (self.metrics_df['split'] == 'test')
            ].copy()
            
            if len(target_metrics) > 0:
                # Sort by RMSE (best performance)
                target_metrics = target_metrics.sort_values('rmse')
                
                print("\nTop Models (by Test RMSE):")
                for _, row in target_metrics.head(3).iterrows():
                    print(f"  {row['model_type']:10} | RMSE: {row['rmse']:.4f} | "
                          f"MAE: {row['mae']:.4f} | R²: {row['r2']:.4f}")
                
                # Show validation vs test comparison for best model
                best_model = target_metrics.iloc[0]['model']
                val_metrics = self.metrics_df[
                    (self.metrics_df['model'] == best_model) & 
                    (self.metrics_df['split'] == 'val')
                ].iloc[0]
                test_metrics = target_metrics.iloc[0]
                
                print(f"\nBest Model ({best_model}) - Val vs Test:")
                print(f"  Validation | RMSE: {val_metrics['rmse']:.4f} | R²: {val_metrics['r2']:.4f}")
                print(f"  Test       | RMSE: {test_metrics['rmse']:.4f} | R²: {test_metrics['r2']:.4f}")
                
                overfitting = (test_metrics['rmse'] - val_metrics['rmse']) / val_metrics['rmse'] * 100
                print(f"  Overfitting: {overfitting:+.2f}%")
    
    def plot_predictions_vs_actual(self, save_dir='plots'):
        """
        Create scatter plots comparing predictions vs actual values.
        """
        if not hasattr(self, 'metrics_df'):
            self.calculate_comprehensive_metrics()
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating prediction vs actual plots...")
        
        for target in self.target_cols:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Predictions vs Actual - {target.title()}', fontsize=16)
            
            # Get models for this target
            target_models = [k for k in self.models.keys() if k.endswith(target)]
            
            for i, model_key in enumerate(target_models[:2]):  # Show top 2 models
                model_type = model_key.replace(f'_{target}', '')
                
                for j, split in enumerate(['val', 'test']):
                    ax = axes[i, j]
                    
                    pred_key = f"{model_key}_{split}"
                    if pred_key in self.predictions:
                        y_true = self.predictions[pred_key]['y_true']
                        y_pred = self.predictions[pred_key]['y_pred']
                        
                        # Create scatter plot
                        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                        
                        # Add perfect prediction line
                        min_val = min(y_true.min(), y_pred.min())
                        max_val = max(y_true.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                        
                        # Calculate R²
                        r2 = r2_score(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        
                        ax.set_xlabel('Actual')
                        ax.set_ylabel('Predicted')
                        ax.set_title(f'{model_type} - {split.title()} (R²={r2:.3f}, RMSE={rmse:.3f})')
                        ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'predictions_vs_actual_{target}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_path}")
    
    def plot_residuals_analysis(self, save_dir='plots'):
        """
        Create residual analysis plots.
        """
        if not hasattr(self, 'metrics_df'):
            self.calculate_comprehensive_metrics()
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating residual analysis plots...")
        
        for target in self.target_cols:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Residual Analysis - {target.title()}', fontsize=16)
            
            target_models = [k for k in self.models.keys() if k.endswith(target)]
            
            for i, model_key in enumerate(target_models[:2]):
                model_type = model_key.replace(f'_{target}', '')
                
                pred_key = f"{model_key}_test"
                if pred_key in self.predictions:
                    y_true = self.predictions[pred_key]['y_true']
                    y_pred = self.predictions[pred_key]['y_pred']
                    residuals = y_true - y_pred
                    
                    # Residuals vs Predicted
                    axes[i, 0].scatter(y_pred, residuals, alpha=0.6, s=20)
                    axes[i, 0].axhline(y=0, color='r', linestyle='--')
                    axes[i, 0].set_xlabel('Predicted Values')
                    axes[i, 0].set_ylabel('Residuals')
                    axes[i, 0].set_title(f'{model_type} - Residuals vs Predicted')
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Residuals histogram
                    axes[i, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
                    axes[i, 1].axvline(x=0, color='r', linestyle='--')
                    axes[i, 1].set_xlabel('Residuals')
                    axes[i, 1].set_ylabel('Frequency')
                    axes[i, 1].set_title(f'{model_type} - Residuals Distribution')
                    axes[i, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'residual_analysis_{target}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_path}")
    
    def plot_time_series_predictions(self, save_dir='plots', station_id=None, 
                                   sample_days=7):
        """
        Create time series plots showing predictions over time.
        """
        if not hasattr(self, 'metrics_df'):
            self.calculate_comprehensive_metrics()
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating time series prediction plots...")
        
        # Get test data with metadata
        test_metadata = self.data_splits['test']['metadata']
        
        for target in self.target_cols:
            target_models = [k for k in self.models.keys() if k.endswith(target)]
            best_model = target_models[0]  # Assume first is best
            
            pred_key = f"{best_model}_test"
            if pred_key in self.predictions:
                # Prepare data
                y_true = self.predictions[pred_key]['y_true']
                y_pred = self.predictions[pred_key]['y_pred']
                
                # Create DataFrame for plotting
                plot_df = pd.DataFrame({
                    'datetime': test_metadata['datetime'].values,
                    'station_id': test_metadata['station_id'].values,
                    'actual': y_true.values,
                    'predicted': y_pred
                })
                
                # Filter for specific station if provided
                if station_id is not None:
                    plot_df = plot_df[plot_df['station_id'] == station_id]
                else:
                    # Use the most active station
                    station_counts = plot_df.groupby('station_id').size()
                    station_id = station_counts.idxmax()
                    plot_df = plot_df[plot_df['station_id'] == station_id]
                
                # Sample days for clarity
                plot_df = plot_df.sort_values('datetime')
                plot_df = plot_df.head(sample_days * 48)  # 48 intervals per day
                
                # Create plot
                plt.figure(figsize=(15, 8))
                plt.plot(plot_df['datetime'], plot_df['actual'], 
                        label='Actual', linewidth=2, alpha=0.8)
                plt.plot(plot_df['datetime'], plot_df['predicted'], 
                        label='Predicted', linewidth=2, alpha=0.8)
                
                plt.xlabel('Time')
                plt.ylabel(f'{target.title()} Count')
                plt.title(f'Time Series Predictions - {target.title()} (Station {station_id})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plot_path = os.path.join(save_dir, f'time_series_{target}_station_{station_id}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved plot: {plot_path}")
    
    def analyze_feature_importance(self, top_n=20, save_dir='plots'):
        """
        Analyze and plot feature importance for all models.
        """
        print("Analyzing feature importance...")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        feature_importance_data = []
        
        for model_key, model in self.models.items():
            target = model_key.split('_')[-1]
            model_type = model_key.replace(f'_{target}', '')
            
            # Get feature importance
            importance = model.get_feature_importance()
            if importance is not None:
                feature_names = self.data_splits['train']['X'].columns
                
                for i, (feature, imp) in enumerate(zip(feature_names, importance)):
                    feature_importance_data.append({
                        'model': model_key,
                        'model_type': model_type,
                        'target': target,
                        'feature': feature,
                        'importance': imp,
                        'rank': i + 1
                    })
        
        if not feature_importance_data:
            print("No feature importance data available")
            return
        
        importance_df = pd.DataFrame(feature_importance_data)
        
        # Plot feature importance for each model
        for target in self.target_cols:
            target_models = [k for k in self.models.keys() if k.endswith(target)]
            
            fig, axes = plt.subplots(1, len(target_models), figsize=(8*len(target_models), 10))
            if len(target_models) == 1:
                axes = [axes]
            
            fig.suptitle(f'Feature Importance - {target.title()}', fontsize=16)
            
            for i, model_key in enumerate(target_models):
                model_type = model_key.replace(f'_{target}', '')
                model_importance = importance_df[importance_df['model'] == model_key]
                
                if len(model_importance) > 0:
                    # Get top features
                    top_features = model_importance.nlargest(top_n, 'importance')
                    
                    # Create horizontal bar plot
                    y_pos = range(len(top_features))
                    axes[i].barh(y_pos, top_features['importance'])
                    axes[i].set_yticks(y_pos)
                    axes[i].set_yticklabels(top_features['feature'], fontsize=8)
                    axes[i].set_xlabel('Importance')
                    axes[i].set_title(f'{model_type}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Invert y-axis to show most important at top
                    axes[i].invert_yaxis()
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, f'feature_importance_{target}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_path}")
        
        # Print top features summary
        print("\n" + "="*60)
        print("TOP FEATURE IMPORTANCE SUMMARY")
        print("="*60)
        
        for target in self.target_cols:
            print(f"\n--- {target.upper()} ---")
            target_importance = importance_df[importance_df['target'] == target]
            
            if len(target_importance) > 0:
                # Average importance across models
                avg_importance = (target_importance.groupby('feature')['importance']
                                .mean().sort_values(ascending=False))
                
                print(f"Top {min(10, len(avg_importance))} Features (averaged across models):")
                for i, (feature, importance) in enumerate(avg_importance.head(10).items(), 1):
                    print(f"  {i:2d}. {feature:30} | {importance:.4f}")
        
        return importance_df
    
    def create_comparison_plots(self, save_dir='plots'):
        """
        Create model comparison plots.
        """
        if not hasattr(self, 'metrics_df'):
            self.calculate_comprehensive_metrics()
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating model comparison plots...")
        
        # Model comparison by metric
        test_metrics = self.metrics_df[self.metrics_df['split'] == 'test']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics_to_plot = ['rmse', 'mae', 'r2', 'mape']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            
            # Create grouped bar plot
            pivot_data = test_metrics.pivot(index='target', columns='model_type', values=metric)
            pivot_data.plot(kind='bar', ax=ax)
            
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xlabel('Target')
            ax.set_ylabel(metric.upper())
            ax.legend(title='Model Type')
            ax.grid(True, alpha=0.3)
            
            if metric == 'r2':
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
    
    def generate_evaluation_report(self, save_dir='evaluation_results'):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_dir (str): Directory to save all results
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print(f"{'='*80}")
        
        # Calculate metrics
        metrics_df = self.calculate_comprehensive_metrics()
        
        # Save metrics to CSV
        metrics_path = os.path.join(save_dir, 'metrics_summary.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics summary: {metrics_path}")
        
        # Print summary
        self.print_metrics_summary()
        
        # Create all plots
        plots_dir = os.path.join(save_dir, 'plots')
        self.plot_predictions_vs_actual(plots_dir)
        self.plot_residuals_analysis(plots_dir)
        self.plot_time_series_predictions(plots_dir)
        self.create_comparison_plots(plots_dir)
        
        # Feature importance analysis
        importance_df = self.analyze_feature_importance(save_dir=plots_dir)
        if importance_df is not None:
            importance_path = os.path.join(save_dir, 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"Saved feature importance: {importance_path}")
        
        print(f"\n{'='*80}")
        print(f"EVALUATION REPORT COMPLETE - Results saved to: {save_dir}")
        print(f"{'='*80}")
        
        return {
            'metrics': metrics_df,
            'feature_importance': importance_df,
            'save_dir': save_dir
        }


def evaluate_models(models_dict, data_splits, target_cols=['arrivals', 'departures'], 
                   save_dir='evaluation_results'):
    """
    Convenience function for complete model evaluation.
    
    Args:
        models_dict (dict): Dictionary of trained models
        data_splits (dict): Data splits from training
        target_cols (list): Target column names
        save_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation results
    """
    evaluator = ModelEvaluator(models_dict, data_splits, target_cols)
    results = evaluator.generate_evaluation_report(save_dir)
    return evaluator, results


if __name__ == "__main__":
    print("Evaluation Module for Individual Station Bike Prediction")
    print("Use evaluate_models() to perform comprehensive evaluation") 