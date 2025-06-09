from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for models.
    """
    def __init__(self):
        self.model = None
        self.hps = None

    @abstractmethod
    def prepare_dataset(self, dataset):
        """
        Prepare the dataset using double dispatch.
        """
        pass

    @abstractmethod
    def set_hps(self, hps: dict):
        """
        Set hyperparameters for the model.
        """
        pass

    @abstractmethod
    def train(self, train_data, valid_data=None):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, data):
        """
        Make predictions with the trained model.
        """
        pass
    
    @abstractmethod
    def get_metrics(self, data) -> dict:
        """
        Calculate and return metrics on the given data.
        """
        pass

    def save(self, path):
        """
        Save the model to a file.
        """
        raise NotImplementedError

    def load(self, path):
        """
        Load a model from a file.
        """
        raise NotImplementedError 