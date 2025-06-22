#!/usr/bin/env python3
"""
Generate clustered or station-level dataset for EcoBici analysis.

This script:
1. Loads and merges all datasets (trips_with_weather, trips, users)
2. Creates temporal splits without data leakage  
3. Optionally fits K-means clustering on training stations (if --n_clusters specified)
4. Generates cluster-level OR station-level features with internal/external trip distinction
5. Handles ghost trips and user demographics
6. Saves train/validation/test datasets with comprehensive features

Usage:
    # With clustering (default):
    python scripts/generate_cluster_dataset.py --data_dir data --output_dir data/clustered --n_clusters 93
    
    # Without clustering (station-level features):
    python scripts/generate_cluster_dataset.py --data_dir data --output_dir data/station_level --n_clusters none
    
Features generated:
- Metadata (cluster centroids/station coordinates, counts)
- Temporal features (hour, day, season, cyclical encodings)
- Weather features (temperature, humidity, precipitation, etc.)
- Internal trip features (departures/arrivals within cluster/station)
- External trip features (departures/arrivals across clusters/stations)
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


def parse_n_clusters(value):
    """Parse n_clusters argument, allowing 'none' to disable clustering."""
    if value.lower() == 'none':
        return None
    return int(value)


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
        type=parse_n_clusters,
        default=93,
        help="Number of clusters for K-means (default: 93). Use 'none' to disable clustering and use individual stations"
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
    if args.n_clusters is None:
        logger.info("ECOBICI STATION-LEVEL DATASET GENERATION (NO CLUSTERING)")
    else:
        logger.info("ECOBICI CLUSTER DATASET GENERATION")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.n_clusters is None:
        logger.info(f"Clustering: DISABLED (using individual stations)")
    else:
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
        
        # only initialize clusterer if clustering is enabled
        clusterer = None
        if args.n_clusters is not None:
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
        
        # step 3: extract station metadata and fit clustering on train data only (if enabled)
        logger.info("\n" + "="*60)
        if args.n_clusters is not None:
            logger.info("STEP 3: STATION CLUSTERING (TRAIN DATA ONLY)")
        else:
            logger.info("STEP 3: STATION METADATA EXTRACTION (NO CLUSTERING)")
        logger.info("="*60)
        
        if args.n_clusters is not None:
            # clustering is enabled
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
        else:
            # no clustering - create station-level mapping
            logger.info("Extracting station metadata from TRAINING data only...")
            train_stations = data_prep.extract_station_metadata(train_data)
            
            # create identity mapping: each station is its own cluster
            station_cluster_mapping = {row[0]: row[0] for row in train_stations.iter_rows()}
            
            # create station centroids (each station is its own centroid)
            cluster_centroids = {}
            for row in train_stations.iter_rows():
                station_id, name, lat, lon = row
                cluster_centroids[station_id] = {
                    'lat': lat,
                    'lon': lon,
                    'station_count': 1,
                    'station_name': name
                }
            
            logger.info(f"Station-level processing summary:")
            logger.info(f"  -> {len(station_cluster_mapping)} stations extracted")
            logger.info(f"  -> Each station treated as individual cluster")
        
        # step 4: generate features for each split
        logger.info("\n" + "="*60)
        if args.n_clusters is not None:
            logger.info("STEP 4: GENERATING CLUSTER FEATURES")
        else:
            logger.info("STEP 4: GENERATING STATION-LEVEL FEATURES")
        logger.info("="*60)
        
        datasets = {
            'train': train_data,
            'val': val_data, 
            'test': test_data
        }
        
        final_datasets = {}
        
        for split_name, split_data in datasets.items():
            logger.info(f"\nGenerating features for {split_name.upper()} split...")
            
            # different checkpoint names for clustering vs station-level
            if args.n_clusters is not None:
                features_checkpoint = checkpoints_dir / f"04_{split_name}_cluster_features.parquet"
                feature_type = "cluster features"
            else:
                features_checkpoint = checkpoints_dir / f"04_{split_name}_station_features.parquet"
                feature_type = "station features"
            
            if args.use_checkpoints:
                split_features = load_checkpoint_if_exists(features_checkpoint, f"{split_name} {feature_type}", logger)
            else:
                split_features = None
                
            if split_features is None:
                # generate features for this split (cluster or station level)
                split_features = feature_gen.generate_cluster_features(
                    trips_df=split_data,
                    station_cluster_mapping=station_cluster_mapping,
                    cluster_centroids=cluster_centroids
                )
                
                if args.use_checkpoints:
                    save_checkpoint(split_features, features_checkpoint, f"{split_name} {feature_type}", logger)
            
            final_datasets[split_name] = split_features
        
        # step 5: save final datasets
        logger.info("\n" + "="*60)
        logger.info("STEP 5: SAVING FINAL DATASETS")
        logger.info("="*60)
        
        for split_name, split_data in final_datasets.items():
            if args.n_clusters is not None:
                output_path = output_dir / f"{split_name}_cluster_features.parquet"
            else:
                output_path = output_dir / f"{split_name}_station_features.parquet"
            save_checkpoint(split_data, output_path, f"final {split_name} dataset", logger)
        
        # step 6: save metadata and summary
        logger.info("\n" + "="*60)
        logger.info("STEP 6: SAVING METADATA AND SUMMARY")
        logger.info("="*60)
        
        # save metadata
        metadata = {
            'n_clusters': args.n_clusters,
            'clustering_enabled': args.n_clusters is not None,
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
        if args.n_clusters is not None:
            title = "ECOBICI CLUSTER DATASET GENERATION SUMMARY"
            config_line = f"  Number of clusters: {args.n_clusters}"
            summary_section = "CLUSTERING SUMMARY:"
            summary_details = [
                f"  Stations mapped: {len(station_cluster_mapping)}",
                f"  Clusters created: {len(cluster_centroids)}"
            ]
            output_files = [
                f"  Train dataset: {output_dir}/train_cluster_features.parquet",
                f"  Validation dataset: {output_dir}/val_cluster_features.parquet", 
                f"  Test dataset: {output_dir}/test_cluster_features.parquet",
                f"  Clustering model: {models_dir}/kmeans_k{args.n_clusters}_model.pkl",
                f"  Metadata: {metadata_path}"
            ]
            features_desc = [
                "  - Cluster metadata (centroids, station counts)",
                "  - Internal trip features (within cluster)",
                "  - External trip features (across clusters)"
            ]
        else:
            title = "ECOBICI STATION-LEVEL DATASET GENERATION SUMMARY"
            config_line = f"  Clustering: DISABLED (station-level features)"
            summary_section = "STATION PROCESSING SUMMARY:"
            summary_details = [
                f"  Stations processed: {len(station_cluster_mapping)}",
                f"  Each station treated individually"
            ]
            output_files = [
                f"  Train dataset: {output_dir}/train_station_features.parquet",
                f"  Validation dataset: {output_dir}/val_station_features.parquet", 
                f"  Test dataset: {output_dir}/test_station_features.parquet",
                f"  Metadata: {metadata_path}"
            ]
            features_desc = [
                "  - Station metadata (coordinates, names)",
                "  - Internal trip features (within station)",
                "  - External trip features (across stations)"
            ]
        
        summary_lines = [
            title,
            "=" * 50,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data directory: {args.data_dir}",
            f"Output directory: {args.output_dir}",
            "",
            "CONFIGURATION:",
            config_line,
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
            summary_section,
        ] + summary_details + [
            "",
            "OUTPUT FILES:",
        ] + output_files + [
            "",
            "FEATURES INCLUDED:",
        ] + features_desc + [
            "  - Temporal features (hour, day, season, cyclical)",
            "  - Weather features (temperature, humidity, precipitation)",
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
        if args.n_clusters is not None:
            logger.info("✓ CLUSTER DATASET GENERATION COMPLETED SUCCESSFULLY!")
        else:
            logger.info("✓ STATION-LEVEL DATASET GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Train dataset: {final_datasets['train'].shape}")
        logger.info(f"Validation dataset: {final_datasets['val'].shape}")
        logger.info(f"Test dataset: {final_datasets['test'].shape}")
        logger.info(f"Total features: {final_datasets['train'].width}")
        if args.n_clusters is not None:
            logger.info(f"Clustering mode: {args.n_clusters} clusters")
        else:
            logger.info(f"Station-level mode: {len(station_cluster_mapping)} individual stations")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 