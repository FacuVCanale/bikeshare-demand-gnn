#!/usr/bin/env python3
"""
Command-line script to prepare GNN dataset for EcoBici arrival prediction.
Usage: python scripts/prepare_gnn_dataset.py --data_path data --split train --output_path data/gnn
"""

import argparse
import sys
import os
from pathlib import Path
import pickle
import torch

# add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset.spatial_temporal_dataset import EcoBiciSpatialTemporalDataset

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GNN dataset for EcoBici arrival prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare training dataset
  python scripts/prepare_gnn_dataset.py --split train
  
  # Prepare all splits with custom parameters
  python scripts/prepare_gnn_dataset.py --split all --sequence_length 24 --k_neighbors 6
  
  # Just show dataset statistics
  python scripts/prepare_gnn_dataset.py --split train --stats_only
  
  # Validate data leakage and generate report
  python scripts/prepare_gnn_dataset.py --split train --validate_leakage --leakage_report
  
  # Full pipeline with all validations
  python scripts/prepare_gnn_dataset.py --split all --save_scalers --validate_leakage --verbose
        """
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data',
        help='Path to data directory (default: data)'
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        choices=['train', 'val', 'test', 'all'],
        default='train',
        help='Data split to prepare (default: train)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/gnn',
        help='Output directory for processed datasets (default: data/gnn)'
    )
    
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=12,
        help='Sequence length for temporal modeling (default: 12)'
    )
    
    parser.add_argument(
        '--k_neighbors',
        type=int,
        default=8,
        help='Number of spatial neighbors for graph construction (default: 8)'
    )
    
    parser.add_argument(
        '--stats_only',
        action='store_true',
        help='Only show dataset statistics without saving'
    )
    
    parser.add_argument(
        '--save_scalers',
        action='store_true',
        help='Save feature and target scalers for later use'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--validate_leakage',
        action='store_true',
        help='Validate that there is no data leakage in lag features'
    )
    
    parser.add_argument(
        '--leakage_report',
        action='store_true',
        help='Generate comprehensive data leakage prevention report'
    )
    
    args = parser.parse_args()
    
    # create output directory
    output_path = Path(args.output_path)
    if not args.stats_only:
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🚴 EcoBici GNN Dataset Preparation")
    print(f"{'='*50}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"K-neighbors: {args.k_neighbors}")
    print(f"Output path: {args.output_path}")
    print()
    
    # initialize dataset
    if args.verbose:
        print("📊 Initializing dataset...")
    
    dataset = EcoBiciSpatialTemporalDataset(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        k_neighbors=args.k_neighbors
    )
    
    # generate leakage report if requested
    if args.leakage_report:
        dataset.print_data_leakage_report()
    
    # validate data leakage if requested
    if args.validate_leakage:
        print("\n🔍 Validating data leakage...")
        for split in ['train', 'val', 'test']:
            file_path = Path(args.data_path) / "clustered" / f"{split}_cluster_features.parquet"
            if file_path.exists():
                try:
                    dataset.validate_no_data_leakage(split, sample_size=500)
                except Exception as e:
                    print(f"❌ Validation failed for {split}: {e}")
                    if not args.stats_only:
                        return  # stop processing if leakage detected
        print("✅ All data leakage validations passed!")
    
    # determine splits to process
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    # process each split
    for split in splits:
        print(f"🔄 Processing {split} split...")
        
        try:
            # load data
            temporal_signal = dataset.load_data(split)
            
            # show statistics
            print(f"\n📈 {split.capitalize()} Dataset Statistics:")
            print(f"  • Number of time steps: {len(temporal_signal)}")
            print(f"  • Number of nodes (clusters): {temporal_signal.edge_index.max().item() + 1}")
            print(f"  • Number of edges: {temporal_signal.edge_index.shape[1]}")
            
            if len(temporal_signal) > 0:
                sample_features = temporal_signal[0].x
                sample_targets = temporal_signal[0].y
                print(f"  • Features per node: {sample_features.shape[1]}")
                print(f"  • Target statistics:")
                print(f"    - Mean: {sample_targets.mean().item():.4f}")
                print(f"    - Std: {sample_targets.std().item():.4f}")
                print(f"    - Min: {sample_targets.min().item():.4f}")
                print(f"    - Max: {sample_targets.max().item():.4f}")
                print(f"    - Non-zero ratio: {(sample_targets > 0).float().mean().item():.4f}")
            
            # save if not stats only
            if not args.stats_only:
                output_file = output_path / f"{split}_temporal_signal.pkl"
                
                if args.verbose:
                    print(f"💾 Saving to {output_file}...")
                
                with open(output_file, 'wb') as f:
                    pickle.dump(temporal_signal, f)
                
                print(f"✅ Saved {split} dataset to {output_file}")
                
                # save additional info
                info = {
                    'n_nodes': temporal_signal.edge_index.max().item() + 1,
                    'n_edges': temporal_signal.edge_index.shape[1],
                    'n_timesteps': len(temporal_signal),
                    'n_features': temporal_signal[0].x.shape[1] if len(temporal_signal) > 0 else 0,
                    'sequence_length': args.sequence_length,
                    'k_neighbors': args.k_neighbors
                }
                
                info_file = output_path / f"{split}_info.pkl"
                with open(info_file, 'wb') as f:
                    pickle.dump(info, f)
        
        except Exception as e:
            print(f"❌ Error processing {split} split: {e}")
            continue
    
    # save scalers if requested and we processed training data
    if args.save_scalers and ('train' in splits) and not args.stats_only:
        try:
            scalers = {
                'feature_scaler': dataset.feature_scaler,
                'target_scaler': dataset.target_scaler
            }
            
            scalers_file = output_path / 'scalers.pkl'
            with open(scalers_file, 'wb') as f:
                pickle.dump(scalers, f)
            
            print(f"💾 Saved scalers to {scalers_file}")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not save scalers: {e}")
    
    print(f"\n🎉 Dataset preparation complete!")
    
    if not args.stats_only:
        print(f"\n📁 Generated files in {args.output_path}:")
        for file in output_path.glob("*"):
            print(f"  • {file.name}")

if __name__ == "__main__":
    main() 