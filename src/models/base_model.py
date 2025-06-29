from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This class defines the common interface that all models must implement,
    ensuring consistency across different model types (GBDT, CatBoost, etc.).
    """
    
    def __init__(self):
        """Initialize the base model."""
        self.model = None
        self.model_type = None
        self.hps = {}
    
    def prepare_dataset(self, dataset):
        """
        Prepare the dataset for this specific model type.
        
        Uses double dispatch pattern - the dataset will call the appropriate
        preparation method based on the model type.
        
        Args:
            dataset: Dataset object that implements preparation methods
            
        Returns:
            Prepared data suitable for this model type
        """
        # Default implementation - subclasses can override if needed
        if hasattr(dataset, 'prepare_for_model'):
            return dataset.prepare_for_model(self)
        else:
            raise NotImplementedError(
                f"Dataset {type(dataset)} does not implement preparation for {type(self)}"
            )
    
    @abstractmethod
    def train(self, train_data: Tuple[Any, Any], valid_data: Optional[Tuple[Any, Any]] = None):
        """
        Train the model on the provided training data.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            valid_data: Optional tuple of (X_valid, y_valid) for validation
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any):
        """
        Make predictions on the provided data.
        
        Args:
            data: Input data for prediction. Can be a single array or tuple (X, y)
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def get_metrics(self, data: Tuple[Any, Any]) -> Dict[str, float]:
        """
        Calculate and return performance metrics on the provided data.
        
        Args:
            data: Tuple of (X, y_true) for evaluation
            
        Returns:
            Dictionary containing metric names and values
        """
        pass
    
    @abstractmethod
    def set_hps(self, hps: Dict[str, Any]):
        """
        Set hyperparameters for the model.
        
        Args:
            hps: Dictionary of hyperparameter names and values
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """
        Save the trained model to a file.
        
        Args:
            path: File path where the model should be saved
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load a trained model from a file.
        
        Args:
            path: File path from which to load the model
            
        Returns:
            The loaded model object
        """
        pass
    
    def predict_proba(self, data: Any):
        """
        Make probability predictions (for classification models).
        
        Args:
            data: Input data for prediction
            
        Returns:
            Probability predictions
            
        Raises:
            ValueError: If the model is not a classifier
        """
        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifiers")
        
        # Default implementation - subclasses should override
        raise NotImplementedError("predict_proba not implemented for this model type")
    
    def get_feature_importance(self) -> Optional[Any]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Feature importance array/values if available, None otherwise
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the current hyperparameters of the model.
        
        Returns:
            Dictionary of current hyperparameters
        """
        return self.hps.copy()
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_type={self.model_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return (f"{self.__class__.__name__}("
                f"model_type={self.model_type}, "
                f"hyperparameters={len(self.hps)} params)") 