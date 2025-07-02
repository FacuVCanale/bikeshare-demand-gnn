#!/usr/bin/env python3
"""
Analyze K-means++ clustering to find optimal k based on silhouette score.

This script analyzes EcoBici station data to find the optimal number of clusters
using K-means++ initialization and silhouette score evaluation.

Features:
- Loads station data using the same preprocessing as generate_cluster_dataset.py
- Tests multiple k values with K-means++ initialization
- Calculates silhouette score, inertia, and cluster statistics
- Generates comprehensive visualizations saved to images/ directory
- Provides detailed logging throughout the analysis process
- Saves results in multiple formats (JSON, CSV, pickle)

Usage:
    python scripts/analyze_kmeans_optimal_k.py --data_dir data --output_dir data/clustering_analysis
    python scripts/analyze_kmeans_optimal_k.py --data_dir data --output_dir data/clustering_analysis --min_k 2 --max_k 150 --step_k 5
"""

import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime
import sys
import os
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# add src to python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.data_preparation import DataPreparator


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging configuration."""
    
    # create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"kmeans_analysis_{timestamp}.log"
    
    # configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - log file: {log_file}")
    
    return logger


def load_station_data(data_dir: str, logger: logging.Logger) -> pl.DataFrame:
    """Load station metadata from EcoBici dataset."""
    
    logger.info("Loading station data from EcoBici dataset...")
    
    # initialize data preparator
    data_prep = DataPreparator(data_dir=data_dir, log_level="INFO")
    
    # load datasets
    trips_weather, trips, users = data_prep.load_datasets()
    
    # filter low-activity stations (same as generate_cluster_dataset.py)
    logger.info("Filtering stations with less than 5000 trips...")
    trips_weather, trips, filtering_stats = data_prep.filter_low_activity_stations(
        trips_weather, trips, min_trips=5000
    )
    
    # log filtering statistics
    logger.info("Station filtering results:")
    logger.info(f"  -> Stations retained: {filtering_stats['active_stations']:,} / {filtering_stats['total_stations']:,}")
    
    # merge datasets
    logger.info("Merging datasets to extract station metadata...")
    merged_data = data_prep.merge_datasets(trips_weather, trips, users)
    
    # create temporal split to extract train data
    logger.info("Creating temporal split to use training data only...")
    train_data, val_data, test_data = data_prep.create_temporal_splits(
        merged_data, 
        train_end_date="2023-01-01",
        val_end_date="2023-07-01"
    )
    
    # extract station metadata from TRAIN data only
    logger.info("Extracting station metadata from TRAINING data only...")
    stations = data_prep.extract_station_metadata(train_data)
    
    logger.info(f"Station data loaded successfully:")
    logger.info(f"  -> Total stations: {stations.shape[0]}")
    
    return stations


def analyze_optimal_k(stations: pl.DataFrame, 
                     min_k: int, max_k: int, step_k: int,
                     random_state: int, logger: logging.Logger) -> Dict[str, Any]:
    """Analyze different k values for K-means++ clustering."""
    
    logger.info("="*80)
    logger.info("STARTING K-MEANS++ OPTIMAL K ANALYSIS")
    logger.info("="*80)
    logger.info(f"K range: {min_k} to {max_k} (step: {step_k})")
    logger.info(f"Total stations: {stations.shape[0]}")
    
    # prepare coordinates
    coords = stations.select(['lat', 'lon']).to_numpy()
    logger.info(f"Coordinate matrix shape: {coords.shape}")
    
    # standardize coordinates
    logger.info("Standardizing coordinates...")
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # generate k values to test
    k_values = list(range(min_k, max_k + 1, step_k))
    logger.info(f"Testing {len(k_values)} different k values")
    
    # initialize results storage
    results = {
        'k_values': k_values,
        'silhouette_scores': [],
        'inertias': [],
        'cluster_size_stats': [],
        'execution_times': [],
        'models': {},
        'coords_scaled': coords_scaled,
        'scaler': scaler,
        'original_coords': coords
    }
    
    # test each k value
    for i, k in enumerate(k_values):
        logger.info(f"Testing k={k} ({i+1}/{len(k_values)})...")
        
        start_time = datetime.now()
        
        try:
            # fit kmeans++
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = kmeans.fit_predict(coords_scaled)
            
            # calculate silhouette score
            if k > 1:
                sil_score = silhouette_score(coords_scaled, cluster_labels)
            else:
                sil_score = -1
            
            # get inertia
            inertia = kmeans.inertia_
            
            # calculate cluster size statistics
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            size_stats = {
                'min_size': counts.min(),
                'max_size': counts.max(),
                'mean_size': counts.mean(),
                'std_size': counts.std(),
                'empty_clusters': k - len(unique_labels)
            }
            
            # calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # store results
            results['silhouette_scores'].append(sil_score)
            results['inertias'].append(inertia)
            results['cluster_size_stats'].append(size_stats)
            results['execution_times'].append(execution_time)
            results['models'][k] = {
                'kmeans': kmeans,
                'labels': cluster_labels,
                'centroids': kmeans.cluster_centers_
            }
            
            # log results for this k
            logger.info(f"  -> Silhouette Score: {sil_score:.4f}")
            logger.info(f"  -> Inertia: {inertia:.2f}")
            logger.info(f"  -> Cluster sizes: min={size_stats['min_size']}, max={size_stats['max_size']}")
            logger.info(f"  -> Execution time: {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"  -> Error testing k={k}: {str(e)}")
            results['silhouette_scores'].append(-1)
            results['inertias'].append(float('inf'))
            results['cluster_size_stats'].append({'error': str(e)})
            results['execution_times'].append(0)
    
    # find optimal k based on silhouette score
    valid_scores = [(k, score) for k, score in zip(k_values, results['silhouette_scores']) if score > -1]
    
    if valid_scores:
        optimal_k, optimal_score = max(valid_scores, key=lambda x: x[1])
        results['optimal_k'] = optimal_k
        results['optimal_score'] = optimal_score
        
        logger.info("="*60)
        logger.info("OPTIMAL K ANALYSIS RESULTS")
        logger.info("="*60)
        logger.info(f"Optimal k: {optimal_k}")
        logger.info(f"Best silhouette score: {optimal_score:.4f}")
        
        # find top 5 k values
        top_5 = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Top 5 k values by silhouette score:")
        for i, (k, score) in enumerate(top_5, 1):
            inertia_idx = k_values.index(k)
            inertia = results['inertias'][inertia_idx]
            logger.info(f"  {i}. k={k}: silhouette={score:.4f}, inertia={inertia:.2f}")
            
    else:
        logger.warning("No valid silhouette scores found!")
        results['optimal_k'] = None
        results['optimal_score'] = None
    
    return results


def create_visualizations(results: Dict[str, Any], stations: pl.DataFrame, 
                         output_dir: Path, logger: logging.Logger):
    """Create comprehensive visualizations of the clustering analysis."""
    
    logger.info("="*60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*60)
    
    # create images directory
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving visualizations to: {images_dir}")
    
    # set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. silhouette score vs k plot
    logger.info("Creating silhouette score vs k plot...")
    
    plt.figure(figsize=(12, 8))
    
    # filter valid scores for plotting
    k_values = results['k_values']
    sil_scores = results['silhouette_scores']
    valid_indices = [i for i, score in enumerate(sil_scores) if score > -1]
    valid_k = [k_values[i] for i in valid_indices]
    valid_scores = [sil_scores[i] for i in valid_indices]
    
    plt.plot(valid_k, valid_scores, 'b-o', linewidth=2, markersize=6)
    
    # highlight optimal k
    if results['optimal_k']:
        optimal_idx = valid_k.index(results['optimal_k'])
        plt.plot(results['optimal_k'], valid_scores[optimal_idx], 'ro', markersize=10, 
                label=f'Optimal k={results["optimal_k"]} (score={results["optimal_score"]:.4f})')
        plt.axvline(x=results['optimal_k'], color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score vs Number of Clusters\nK-means++ Clustering Analysis', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # add text box with analysis info
    textstr = f'Stations: {len(stations)}\nK range: {min(k_values)}-{max(k_values)}\nValid scores: {len(valid_scores)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(images_dir / 'silhouette_score_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. elbow method plot (inertia vs k)
    logger.info("Creating elbow method plot...")
    
    plt.figure(figsize=(12, 8))
    
    # filter finite inertias
    inertias = results['inertias']
    valid_inertia_indices = [i for i, inertia in enumerate(inertias) if np.isfinite(inertia)]
    valid_k_inertia = [k_values[i] for i in valid_inertia_indices]
    valid_inertias = [inertias[i] for i in valid_inertia_indices]
    
    plt.plot(valid_k_inertia, valid_inertias, 'g-o', linewidth=2, markersize=6)
    
    # highlight optimal k
    if results['optimal_k'] and results['optimal_k'] in valid_k_inertia:
        optimal_inertia_idx = valid_k_inertia.index(results['optimal_k'])
        plt.plot(results['optimal_k'], valid_inertias[optimal_inertia_idx], 'ro', markersize=10,
                label=f'Optimal k={results["optimal_k"]}')
        plt.axvline(x=results['optimal_k'], color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
    plt.title('Elbow Method for Optimal k\nK-means++ Clustering Analysis', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(images_dir / 'elbow_method_inertia_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. optimal clustering visualization (if we have optimal k)
    if results['optimal_k'] and results['optimal_k'] in results['models']:
        logger.info(f"Creating optimal clustering visualization (k={results['optimal_k']})...")
        
        optimal_model = results['models'][results['optimal_k']]
        labels = optimal_model['labels']
        coords = results['original_coords']
        
        plt.figure(figsize=(15, 12))
        
        # create scatter plot with cluster colors
        scatter = plt.scatter(coords[:, 1], coords[:, 0], c=labels, cmap='tab20', 
                            s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # plot cluster centroids (transform back to original scale)
        centroids_original = results['scaler'].inverse_transform(optimal_model['centroids'])
        plt.scatter(centroids_original[:, 1], centroids_original[:, 0], 
                   c='red', marker='x', s=200, linewidth=3, label='Centroids')
        
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title(f'Optimal K-means++ Clustering (k={results["optimal_k"]})\n'
                 f'Silhouette Score: {results["optimal_score"]:.4f}', fontsize=14, pad=20)
        
        # add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster ID', fontsize=12)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(images_dir / f'optimal_clustering_k{results["optimal_k"]}_map.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. cluster size distribution for optimal k
        logger.info("Creating cluster size distribution plot...")
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(counts)), sorted(counts, reverse=True), 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.xlabel('Cluster Rank (by size)', fontsize=12)
        plt.ylabel('Number of Stations', fontsize=12)
        plt.title(f'Cluster Size Distribution (k={results["optimal_k"]})', fontsize=14, pad=20)
        
        # add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(images_dir / f'cluster_size_distribution_k{results["optimal_k"]}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"✓ All visualizations saved to: {images_dir}")


def save_results(results: Dict[str, Any], stations: pl.DataFrame, 
                output_dir: Path, logger: logging.Logger, args: argparse.Namespace):
    """Save comprehensive analysis results to files."""
    
    logger.info("="*60)
    logger.info("SAVING ANALYSIS RESULTS")
    logger.info("="*60)
    
    # create results summary
    summary = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'total_stations': len(stations),
            'k_range': f"{min(results['k_values'])}-{max(results['k_values'])}",
            'k_step': args.step_k,
            'total_k_tested': len(results['k_values']),
            'random_state': args.random_state,
        },
        'optimal_results': {
            'optimal_k': results['optimal_k'],
            'optimal_silhouette_score': results['optimal_score'],
        },
        'performance_summary': {
            'total_execution_time_seconds': sum(results['execution_times']),
            'total_execution_time_minutes': sum(results['execution_times']) / 60,
            'average_time_per_k': np.mean(results['execution_times']),
        }
    }
    
    # add top k results
    if results['optimal_k']:
        valid_scores = [(k, score) for k, score in zip(results['k_values'], results['silhouette_scores']) if score > -1]
        top_10 = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:10]
        
        summary['top_k_results'] = []
        for rank, (k, score) in enumerate(top_10, 1):
            k_idx = results['k_values'].index(k)
            inertia = results['inertias'][k_idx]
            exec_time = results['execution_times'][k_idx]
            
            summary['top_k_results'].append({
                'rank': rank,
                'k': k,
                'silhouette_score': score,
                'inertia': inertia,
                'execution_time_seconds': exec_time
            })
    
    # save summary as JSON
    summary_path = output_dir / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Analysis summary saved to: {summary_path}")
    
    # save k values and scores as CSV
    results_df_data = []
    for i, k in enumerate(results['k_values']):
        row = {
            'k': k,
            'silhouette_score': results['silhouette_scores'][i],
            'inertia': results['inertias'][i],
            'execution_time_seconds': results['execution_times'][i],
        }
        
        # add cluster size statistics if available
        if i < len(results['cluster_size_stats']) and isinstance(results['cluster_size_stats'][i], dict):
            stats = results['cluster_size_stats'][i]
            if 'error' not in stats:
                row.update({
                    'min_cluster_size': stats['min_size'],
                    'max_cluster_size': stats['max_size'],
                    'mean_cluster_size': stats['mean_size'],
                    'std_cluster_size': stats['std_size'],
                    'empty_clusters': stats['empty_clusters']
                })
        
        results_df_data.append(row)
    
    # convert to polars and save
    results_df = pl.DataFrame(results_df_data)
    csv_path = output_dir / 'k_analysis_results.csv'
    results_df.write_csv(csv_path)
    logger.info(f"Results CSV saved to: {csv_path}")
    
    # create final report
    report_lines = [
        "K-MEANS++ CLUSTERING ANALYSIS REPORT",
        "="*50,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Data directory: {args.data_dir}",
        f"Output directory: {args.output_dir}",
        "",
        "ANALYSIS CONFIGURATION:",
        f"  Total stations analyzed: {len(stations):,}",
        f"  K range tested: {min(results['k_values'])} to {max(results['k_values'])} (step: {args.step_k})",
        f"  Total k values tested: {len(results['k_values'])}",
        f"  Random state: {args.random_state}",
        "",
        "OPTIMAL K RESULTS:",
    ]
    
    if results['optimal_k']:
        report_lines.extend([
            f"  Best k: {results['optimal_k']}",
            f"  Best silhouette score: {results['optimal_score']:.4f}",
            "",
            "TOP 5 K VALUES BY SILHOUETTE SCORE:",
        ])
        
        valid_scores = [(k, score) for k, score in zip(results['k_values'], results['silhouette_scores']) if score > -1]
        top_5 = sorted(valid_scores, key=lambda x: x[1], reverse=True)[:5]
        for i, (k, score) in enumerate(top_5, 1):
            k_idx = results['k_values'].index(k)
            inertia = results['inertias'][k_idx]
            report_lines.append(f"  {i}. k={k}: silhouette={score:.4f}, inertia={inertia:.2f}")
    else:
        report_lines.append("  No valid results found!")
    
    report_lines.extend([
        "",
        "PERFORMANCE SUMMARY:",
        f"  Total execution time: {sum(results['execution_times']):.1f} seconds",
        f"  Average time per k: {np.mean(results['execution_times']):.2f} seconds",
        "",
        "OUTPUT FILES:",
        f"  Analysis summary: analysis_summary.json",
        f"  Results CSV: k_analysis_results.csv",
        f"  Visualizations: images/ directory",
        "",
        "✓ K-means++ clustering analysis completed successfully!"
    ])
    
    # save report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Analysis report saved to: {report_path}")
    
    logger.info("✓ All results saved successfully!")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze K-means++ clustering to find optimal k based on silhouette score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # data configuration
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root data directory containing EcoBici datasets')
    parser.add_argument('--output_dir', type=str, default='data/clustering_analysis',
                       help='Output directory for analysis results and visualizations')
    
    # k range configuration
    parser.add_argument('--min_k', type=int, default=2,
                       help='Minimum k value to test')
    parser.add_argument('--max_k', type=int, default=150,
                       help='Maximum k value to test')
    parser.add_argument('--step_k', type=int, default=5,
                       help='Step size between k values')
    
    # analysis configuration
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # setup logging
    logger = setup_logging(output_dir, args.log_level)
    
    logger.info("="*80)
    logger.info("K-MEANS++ OPTIMAL K ANALYSIS")
    logger.info("="*80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"K range: {args.min_k} to {args.max_k} (step: {args.step_k})")
    logger.info(f"Random state: {args.random_state}")
    logger.info("="*80)
    
    try:
        # step 1: load station data
        logger.info("STEP 1: LOADING STATION DATA")
        stations = load_station_data(args.data_dir, logger)
        
        # validate k range
        max_possible_k = len(stations) - 1
        actual_max_k = min(args.max_k, max_possible_k)
        
        if args.max_k > max_possible_k:
            logger.warning(f"Requested max_k ({args.max_k}) exceeds maximum possible k ({max_possible_k})")
            logger.warning(f"Adjusting max_k to {actual_max_k}")
        
        if args.min_k >= actual_max_k:
            logger.error(f"min_k ({args.min_k}) must be less than max_k ({actual_max_k})")
            sys.exit(1)
        
        # step 2: analyze optimal k
        logger.info(f"STEP 2: ANALYZING OPTIMAL K ({args.min_k} to {actual_max_k})")
        results = analyze_optimal_k(
            stations=stations,
            min_k=args.min_k,
            max_k=actual_max_k,
            step_k=args.step_k,
            random_state=args.random_state,
            logger=logger
        )
        
        # step 3: create visualizations
        logger.info("STEP 3: CREATING VISUALIZATIONS")
        create_visualizations(results, stations, output_dir, logger)
        
        # step 4: save results
        logger.info("STEP 4: SAVING RESULTS")
        save_results(results, stations, output_dir, logger, args)
        
        # final summary
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Total stations analyzed: {len(stations):,}")
        logger.info(f"K values tested: {len(results['k_values'])}")
        
        if results['optimal_k']:
            logger.info(f"Optimal k: {results['optimal_k']}")
            logger.info(f"Best silhouette score: {results['optimal_score']:.4f}")
        
        logger.info(f"Total execution time: {sum(results['execution_times']):.1f} seconds")
        logger.info(f"Visualizations saved to: {output_dir}/images/")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 