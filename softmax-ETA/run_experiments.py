"""
Example script showing how to run different GBDT experiments.
This demonstrates various ways to use the train_gbdt.py script.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a command and print the result."""
    print(f"\n🚀 Running: {command}")
    print("=" * 80)
    
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
    else:
        print("✅ Command completed successfully")
    
    return result.returncode == 0


def main():
    """Run a series of GBDT experiments."""
    
    print("🧪 GBDT EXPERIMENTS RUNNER")
    print("=" * 80)
    print("This script will run several GBDT experiments with different configurations.")
    print("Make sure you have run feat-eng.py first to generate the data splits.")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not Path("train_gbdt.py").exists():
        print("❌ train_gbdt.py not found. Make sure you're in the softmax-ETA directory.")
        return
    
    # Experiment configurations
    experiments = [
        {
            "name": "lgb_default",
            "description": "LightGBM with default parameters",
            "command": "python train_gbdt.py --model lgb --experiment-name lgb_default"
        },
        {
            "name": "xgb_default", 
            "description": "XGBoost with default parameters",
            "command": "python train_gbdt.py --model xgb --experiment-name xgb_default"
        },
        {
            "name": "lgb_tuned",
            "description": "LightGBM with custom hyperparameters",
            "command": "python train_gbdt.py --model lgb --experiment-name lgb_tuned --eta-params 'num_leaves=50,learning_rate=0.05,n_estimators=200' --dest-params 'num_leaves=50,learning_rate=0.05,n_estimators=200'"
        },
        {
            "name": "xgb_tuned",
            "description": "XGBoost with custom hyperparameters", 
            "command": "python train_gbdt.py --model xgb --experiment-name xgb_tuned --eta-params 'max_depth=8,learning_rate=0.05,n_estimators=200' --dest-params 'max_depth=8,learning_rate=0.05,n_estimators=200'"
        },
        {
            "name": "lgb_fast",
            "description": "LightGBM with fast training parameters",
            "command": "python train_gbdt.py --model lgb --experiment-name lgb_fast --eta-params 'num_leaves=20,learning_rate=0.2,n_estimators=50' --dest-params 'num_leaves=20,learning_rate=0.2,n_estimators=50'"
        }
    ]
    
    successful_experiments = []
    failed_experiments = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n📊 EXPERIMENT {i}/{len(experiments)}: {exp['name'].upper()}")
        print(f"Description: {exp['description']}")
        print("-" * 60)
        
        success = run_command(exp['command'])
        
        if success:
            successful_experiments.append(exp['name'])
        else:
            failed_experiments.append(exp['name'])
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"✅ Successful experiments ({len(successful_experiments)}):")
    for exp_name in successful_experiments:
        print(f"   - {exp_name}")
    
    if failed_experiments:
        print(f"\n❌ Failed experiments ({len(failed_experiments)}):")
        for exp_name in failed_experiments:
            print(f"   - {exp_name}")
    
    print(f"\n📁 Results can be found in the 'results/' directory")
    print(f"📊 To compare results, check the CSV files in results/")
    

if __name__ == "__main__":
    main() 