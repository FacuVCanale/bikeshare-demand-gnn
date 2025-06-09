from abc import ABC, abstractmethod

class BaseDataset(ABC):
    """
    Abstract base class for datasets.
    """
    def __init__(self):
        self.train = None
        self.valid = None
        self.test = None

    @abstractmethod
    def prepare_for(self, model):
        """
        Prepare the dataset for a specific model.
        This could involve transformations, feature engineering, etc.
        """
        pass 