#!/usr/bin/env python3
"""
Test script to verify reproducibility of GNN training.

This script runs the same experiment twice with identical parameters
and verifies that the results are exactly the same.
"""

import subprocess
import sys
import json
from pathlib import Path


def run_experiment(experiment_name: str, seed: int = 42) -> dict:
    """Run a single GNN experiment and return the results"""
    cmd = [
        sys.executable, "scripts/train_gnn.py",
        "--model", "transformer",
        "--hidden_dim", "64",  # smaller for faster testing
        "--num_layers", "2",
        "--epochs", "10",  # fewer epochs for testing
        "--seed", str(seed),
        "--experiment_name", experiment_name,
        "--save_results"
    ]
    
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Experiment {experiment_name} completed successfully")
        
        # try to load results
        results_file = Path(f"experiments/gnn/{experiment_name}/training_history.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Results file not found at {results_file}")
            return {}
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {experiment_name}:")
        print(f"Exit code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return {}


def compare_results(results1: dict, results2: dict, tolerance: float = 1e-10) -> bool:
    """Compare two result dictionaries for approximate equality"""
    
    if not results1 or not results2:
        print("One or both result sets are empty")
        return False
    
    # compare final losses
    try:
        final_loss1 = results1['val_loss'][-1] if results1.get('val_loss') else None
        final_loss2 = results2['val_loss'][-1] if results2.get('val_loss') else None
        
        if final_loss1 is None or final_loss2 is None:
            print("Could not extract final validation losses")
            return False
        
        diff = abs(final_loss1 - final_loss2)
        print(f"Final validation loss 1: {final_loss1:.10f}")
        print(f"Final validation loss 2: {final_loss2:.10f}")
        print(f"Difference: {diff:.2e}")
        
        if diff < tolerance:
            print(f"✅ Results are reproducible (difference < {tolerance})")
            return True
        else:
            print(f"❌ Results are NOT reproducible (difference >= {tolerance})")
            return False
            
    except Exception as e:
        print(f"Error comparing results: {e}")
        return False


def main():
    """Run reproducibility test"""
    print("="*60)
    print("GNN REPRODUCIBILITY TEST")
    print("="*60)
    
    seed = 12345
    exp1_name = "repro_test_1"
    exp2_name = "repro_test_2"
    
    # run first experiment
    print(f"\n--- Running first experiment (seed={seed}) ---")
    results1 = run_experiment(exp1_name, seed)
    
    # run second experiment with same seed
    print(f"\n--- Running second experiment (seed={seed}) ---")
    results2 = run_experiment(exp2_name, seed)
    
    # compare results
    print(f"\n--- Comparing results ---")
    is_reproducible = compare_results(results1, results2)
    
    print(f"\n--- Summary ---")
    if is_reproducible:
        print("🎉 SUCCESS: Training is reproducible!")
        print("The same seed produces identical results.")
    else:
        print("💥 FAILURE: Training is NOT reproducible!")
        print("The same seed produces different results.")
        print("Check the seeding configuration.")
    
    return 0 if is_reproducible else 1


if __name__ == "__main__":
    exit(main()) 