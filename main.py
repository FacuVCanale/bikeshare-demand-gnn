import pandas as pd
import numpy as np
from src.dataset.tabular_dataset import TabularDataset
from src.models.lgbm import LightGBMModel
from src.training.main import train_process

def main():
    """
    Main function to run the training pipeline with LightGBM.
    """
    # 1. Create a sample DataFrame
    print("Creating sample data...")
    data = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'feature3': np.random.rand(1000),
        'target': np.random.rand(1000) * 100
    })
    target_cols = ['target']

    # 2. Instantiate TabularDataset
    print("Creating dataset...")
    dataset = TabularDataset(data, target_columns=target_cols)

    # 3. Instantiate LightGBMModel
    print("Creating model...")
    lgbm = LightGBMModel(model_type='regressor')

    # 4. Define hyperparameters
    hps = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42
    }

    # 5. Call train_process
    print("Starting training process...")
    trained_model = train_process(model=lgbm, dataset=dataset, hps=hps)

    # Example of saving the model
    print("Saving trained model...")
    trained_model.save('models/lgbm_model.joblib')
    
    # Example of loading and predicting
    print("Loading model and making a prediction...")
    loaded_model = LightGBMModel()
    loaded_model.load('models/lgbm_model.joblib')
    
    sample_data = (dataset.test[0].head(1),) # tuple with one dataframe
    prediction = loaded_model.predict(sample_data)
    print(f"Prediction on a sample from test set: {prediction}")


if __name__ == '__main__':
    main() 