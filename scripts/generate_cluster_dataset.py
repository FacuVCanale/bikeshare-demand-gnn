#!/usr/bin/env python3
"""
Generate clustered dataset for EcoBici analysis.

This script:
1. Loads and merges all datasets (trips_with_weather, trips, users)
2. Creates temporal splits without data leakage  
3. Fits K-means clustering on training stations only
4. Generates cluster-level features with internal/external trip distinction
5. Handles ghost trips and user demographics
6. Saves train/validation/test datasets with comprehensive features

Usage:
    python scripts/generate_cluster_dataset.py --data_dir data --output_dir data/clustered
    
Features generated:
- Cluster metadata (centroids, station counts)
- Temporal features (hour, day, season, cyclical encodings)
- Weather features (temperature, humidity, precipitation, etc.)
- Internal trip features (departures/arrivals within cluster)
- External trip features (departures/arrivals across clusters)
- User demographics (gender, age) by trip type
- Bike model distributions by trip type
- Quality flags (user_matched, weather_available, etc.)
"""

import argparse
import polars as pl
from pathlib import Path
import logging
import pickle
from datetime import datetime
import sys
import os

# add src to python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering.data_preparation import DataPreparator
from clustering.station_clustering import StationClusterer
from clustering.cluster_features import ClusterFeatureGenerator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cluster_dataset_generation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(data: pl.DataFrame, filepath: str, description: str, logger):
    """Save checkpoint with logging."""
    logger.info(f"Saving {description} to {filepath}")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(filepath, compression="snappy")
    logger.info(f"  -> Saved {data.shape[0]:,} rows, {data.shape[1]} columns")


def load_checkpoint_if_exists(filepath: str, description: str, logger) -> pl.DataFrame:
    """Load checkpoint if it exists."""
    if Path(filepath).exists():
        logger.info(f"Loading {description} from checkpoint: {filepath}")
        data = pl.read_parquet(filepath)
        logger.info(f"  -> Loaded {data.shape[0]:,} rows, {data.shape[1]} columns")
        return data
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate clustered EcoBici dataset with internal/external trip features"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data",
        help="Root data directory (default: data)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/clustered",
        help="Output directory for clustered datasets (default: data/clustered)"
    )
    
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=93,
        help="Number of clusters for K-means (default: 93)"
    )
    
    parser.add_argument(
        "--dt_minutes", 
        type=int, 
        default=30,
        help="Time interval in minutes for aggregation (default: 30)"
    )
    
    parser.add_argument(
        "--train_end_date", 
        type=str, 
        default="2023-01-01",
        help="End date for training set (format: YYYY-MM-DD, default: 2023-01-01)"
    )
    
    parser.add_argument(
        "--val_end_date", 
        type=str, 
        default="2023-07-01",
        help="End date for validation set (format: YYYY-MM-DD, default: 2023-07-01)"
    )
    
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42,
        help="Random state for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--use_checkpoints", 
        action="store_true",
        help="Use checkpoints to resume processing"
    )
    
    parser.add_argument(
        "--clear_checkpoints", 
        action="store_true",
        help="Clear existing checkpoints before starting"
    )
    
    args = parser.parse_args()
    
    # setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 80)
    logger.info("ECOBICI CLUSTER DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    logger.info(f"Time interval: {args.dt_minutes} minutes") 
    logger.info(f"Train end date: {args.train_end_date}")
    logger.info(f"Validation end date: {args.val_end_date}")
    logger.info(f"Random state: {args.random_state}")
    logger.info("=" * 80)
    
    # create output directories
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    models_dir = output_dir / "models"
    
    for dir_path in [output_dir, checkpoints_dir, models_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # clear checkpoints if requested
    if args.clear_checkpoints:
        logger.info("Clearing existing checkpoints...")
        for checkpoint_file in checkpoints_dir.glob("*.parquet"):
            checkpoint_file.unlink()
        logger.info("  -> Checkpoints cleared")
    
    try:
        # initialize components
        data_prep = DataPreparator(data_dir=args.data_dir, log_level=args.log_level)
        clusterer = StationClusterer(
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            log_level=args.log_level
        )
        feature_gen = ClusterFeatureGenerator(
            dt_minutes=args.dt_minutes,
            log_level=args.log_level
        )
        
        # step 1: load and merge datasets
        logger.info("\n" + "="*60)
        logger.info("STEP 1: LOADING AND MERGING DATASETS")
        logger.info("="*60)
        
        merged_checkpoint = checkpoints_dir / "01_merged_datasets.parquet"
        merged_data = load_checkpoint_if_exists(merged_checkpoint, "merged datasets", logger) if args.use_checkpoints else None
        
        if merged_data is None:
            trips_weather, trips, users = data_prep.load_datasets()
            merged_data = data_prep.merge_datasets(trips_weather, trips, users)
            data_prep.validate_data_quality(merged_data)
            
            if args.use_checkpoints:
                save_checkpoint(merged_data, merged_checkpoint, "merged datasets", logger)
        
        # step 2: create temporal splits
        logger.info("\n" + "="*60)
        logger.info("STEP 2: CREATING TEMPORAL SPLITS")
        logger.info("="*60)
        
        splits_checkpoint = checkpoints_dir / "02_temporal_splits.pkl"
        
        if args.use_checkpoints and splits_checkpoint.exists():
            logger.info(f"Loading temporal splits from checkpoint: {splits_checkpoint}")
            with open(splits_checkpoint, 'rb') as f:
                train_data, val_data, test_data = pickle.load(f)
        else:
            train_data, val_data, test_data = data_prep.create_temporal_splits(
                merged_data, 
                train_end_date=args.train_end_date,
                val_end_date=args.val_end_date
            )
            
            if args.use_checkpoints:
                logger.info(f"Saving temporal splits to checkpoint: {splits_checkpoint}")
                with open(splits_checkpoint, 'wb') as f:
                    pickle.dump((train_data, val_data, test_data), f)
        
        # step 3: extract station metadata and fit clustering on train data only
        logger.info("\n" + "="*60)
        logger.info("STEP 3: STATION CLUSTERING (TRAIN DATA ONLY)")
        logger.info("="*60)
        
        clustering_model_path = models_dir / f"kmeans_k{args.n_clusters}_model.pkl"
        
        if args.use_checkpoints and clustering_model_path.exists():
            clusterer.load_model(clustering_model_path)
        else:
            # extract station metadata from TRAIN data only to prevent data leakage
            logger.info("Extracting station metadata from TRAINING data only...")
            train_stations = data_prep.extract_station_metadata(train_data)
            
            # fit clustering on train stations only
            clusterer.fit_clustering(train_stations)
            
            # save clustering model
            clusterer.save_model(clustering_model_path)
        
        # get clustering artifacts
        station_cluster_mapping = clusterer.get_station_cluster_mapping()
        cluster_centroids = clusterer.get_cluster_centroids()
        
        logger.info(f"Clustering summary:")
        logger.info(f"  -> {len(station_cluster_mapping)} stations mapped")
        logger.info(f"  -> {len(cluster_centroids)} clusters created")
        
        # step 4: generate cluster features for each split
        logger.info("\n" + "="*60)
        logger.info("STEP 4: GENERATING CLUSTER FEATURES")
        logger.info("="*60)
        
        datasets = {
            'train': train_data,
            'val': val_data, 
            'test': test_data
        }
        
        final_datasets = {}
        
        for split_name, split_data in datasets.items():
            logger.info(f"\nGenerating features for {split_name.upper()} split...")
            
            features_checkpoint = checkpoints_dir / f"04_{split_name}_cluster_features.parquet"
            
            if args.use_checkpoints:
                split_features = load_checkpoint_if_exists(features_checkpoint, f"{split_name} cluster features", logger)
            else:
                split_features = None
                
            if split_features is None:
                # generate cluster features for this split
                split_features = feature_gen.generate_cluster_features(
                    trips_df=split_data,
                    station_cluster_mapping=station_cluster_mapping,
                    cluster_centroids=cluster_centroids
                )
                
                if args.use_checkpoints:
                    save_checkpoint(split_features, features_checkpoint, f"{split_name} cluster features", logger)
            
            final_datasets[split_name] = split_features
        
        # step 5: save final datasets
        logger.info("\n" + "="*60)
        logger.info("STEP 5: SAVING FINAL DATASETS")
        logger.info("="*60)
        
        for split_name, split_data in final_datasets.items():
            output_path = output_dir / f"{split_name}_cluster_features.parquet"
            save_checkpoint(split_data, output_path, f"final {split_name} dataset", logger)
        
        # step 6: save metadata and summary
        logger.info("\n" + "="*60)
        logger.info("STEP 6: SAVING METADATA AND SUMMARY")
        logger.info("="*60)
        
        # save clustering metadata
        metadata = {
            'n_clusters': args.n_clusters,
            'dt_minutes': args.dt_minutes,
            'train_end_date': args.train_end_date,
            'val_end_date': args.val_end_date,
            'random_state': args.random_state,
            'total_stations': len(station_cluster_mapping),
            'cluster_centroids': cluster_centroids,
            'generation_timestamp': datetime.now().isoformat(),
            'train_shape': final_datasets['train'].shape,
            'val_shape': final_datasets['val'].shape,
            'test_shape': final_datasets['test'].shape,
            'feature_columns': final_datasets['train'].columns
        }
        
        metadata_path = output_dir / "dataset_metadata.json"
        import json
        import numpy as np
        
        def make_json_serializable(obj):
            """Convert non-serializable objects to JSON-serializable format."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'shape'):  # DataFrame shapes
                return list(obj)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                return list(obj)
            else:
                return obj
        
        with open(metadata_path, 'w') as f:
            serializable_metadata = make_json_serializable(metadata)
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        
        # create summary report
        summary_lines = [
            "ECOBICI CLUSTER DATASET GENERATION SUMMARY",
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data directory: {args.data_dir}",
            f"Output directory: {args.output_dir}",
            "",
            "CONFIGURATION:",
            f"  Number of clusters: {args.n_clusters}",
            f"  Time interval: {args.dt_minutes} minutes",
            f"  Train end date: {args.train_end_date}",
            f"  Validation end date: {args.val_end_date}",
            f"  Random state: {args.random_state}",
            "",
            "DATASET SHAPES:",
            f"  Train: {final_datasets['train'].shape}",
            f"  Validation: {final_datasets['val'].shape}",
            f"  Test: {final_datasets['test'].shape}",
            "",
            "CLUSTERING SUMMARY:",
            f"  Stations mapped: {len(station_cluster_mapping)}",
            f"  Clusters created: {len(cluster_centroids)}",
            "",
            "OUTPUT FILES:",
            f"  Train dataset: {output_dir}/train_cluster_features.parquet",
            f"  Validation dataset: {output_dir}/val_cluster_features.parquet", 
            f"  Test dataset: {output_dir}/test_cluster_features.parquet",
            f"  Clustering model: {clustering_model_path}",
            f"  Metadata: {metadata_path}",
            "",
            "FEATURES INCLUDED:",
            "  - Cluster metadata (centroids, station counts)",
            "  - Temporal features (hour, day, season, cyclical)",
            "  - Weather features (temperature, humidity, precipitation)",
            "  - Internal trip features (within cluster)",
            "  - External trip features (across clusters)",
            "  - User demographics by trip type",
            "  - Bike model distributions by trip type",
            "  - Quality flags (user_matched, weather_available)",
            "",
            "✓ Dataset generation completed successfully!"
        ]
        
        summary_path = output_dir / "generation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Saved summary report to {summary_path}")
        
        # final success message
        logger.info("\n" + "="*80)
        logger.info("✓ CLUSTER DATASET GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Train dataset: {final_datasets['train'].shape}")
        logger.info(f"Validation dataset: {final_datasets['val'].shape}")
        logger.info(f"Test dataset: {final_datasets['test'].shape}")
        logger.info(f"Total features: {final_datasets['train'].width}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 