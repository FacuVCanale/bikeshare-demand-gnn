import logging
from src.models.base_model import BaseModel
from src.dataset.base_dataset import BaseDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_process(model: BaseModel, dataset: BaseDataset, hps: dict):
    """
    Main training process wrapper.

    :param model: An instance of a class that inherits from BaseModel.
    :param dataset: An instance of a class that inherits from BaseDataset.
    :param hps: A dictionary of hyperparameters for the model.
    """
    logging.info("Starting training process...")

    logging.info("Preparing dataset for the model...")
    dataset.prepare_for(model)
    
    logging.info("Setting hyperparameters...")
    model.set_hps(hps)
    
    logging.info("Starting model training...")
    model.train(dataset.train, dataset.valid)
    
    logging.info("Evaluating model on validation set...")
    metrics = model.get_metrics(dataset.valid)
    logging.info(f"Validation metrics: {metrics}")

    logging.info("Evaluating model on test set...")
    test_metrics = model.get_metrics(dataset.test)
    logging.info(f"Test metrics: {test_metrics}")

    logging.info("Training process finished.")
    return model 