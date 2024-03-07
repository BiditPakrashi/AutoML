from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class BaseFeatureSelection(ABC):

    name = 'base'

    @abstractmethod
    def fit(self, X, y):
        raise NotImplemented
    
    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplemented
    

class DummyFeatureSelection(BaseFeatureSelection):

    name = 'Dummy feature selection'

    def __init__(self) -> None:
        self.features = None

    def fit(self, X, y):
        # something happens to choose the right columns
        self.features = list(range(X.shape[1]))
        return X[:, self.features].copy()
    
    def transform(self, X, y=None):
        return X[:, self.features].copy()
    

class BorutaFeatureSelection(BaseFeatureSelection):
    name = 'Boruta feature selection'

    def __init__(self) -> None:
        self.features = None

    def fit(self, X, y):
        # Convert X and y to numpy arrays if they are not already
        X = np.array(X)
        y = np.array(y)

        # Create a random forest classifier
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        # Create an instance of the BorutaPy feature selector
        from boruta import BorutaPy
        selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)

        # Perform feature selection
        selector.fit(X, y)

        # Get the selected features
        self.features = selector.support_

        return X[:, self.features].copy()

    def transform(self, X, y=None):
        return X[:, self.features].copy()
