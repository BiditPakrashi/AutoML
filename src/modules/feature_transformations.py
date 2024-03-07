from abc import ABC, abstractmethod
import pandas as pd
from sklearn.decomposition import PCA

class BaseFeatureTransformation(ABC):

    @abstractmethod
    def fit(self, X, y):
        raise NotImplemented
    
    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplemented


class DummyFeatureTransformation(BaseFeatureTransformation):

    name = 'Dummy feature transformation'

    def __init__(self) -> None:
        self.factor = 1

    def fit(self, X, y):

        self.factor = 2
        return X * self.factor
    
    def transform(self, X, y=None):
        return X * self.factor


class CleanNANFeatureTransformation(BaseFeatureTransformation):
    name = 'Clear Features where 80 percent or more missing Values'

    def __init__(self):
        self.factor = 80

    def fit(self, X, y=None):
        # Calculate the threshold for removing columns
        threshold = len(X) * (self.factor / 100)
        # Get the columns with more NaN values than the threshold
        cols_to_remove = X.columns[X.isnull().sum() > threshold]
        print(cols_to_remove)
        # Remove the columns from the DataFrame
        X.drop(cols_to_remove, axis=1, inplace=True)

        return X

    def transform(self, X, y=None):
        # Remove the same columns as in the fit method
        threshold = len(X) * (self.factor / 100)
        cols_to_remove = X.columns[X.isnull().sum() > threshold]
        print(cols_to_remove)
        X.drop(cols_to_remove, axis=1, inplace=True)

        return X


class PCAFeatureTransformation(BaseFeatureTransformation):
    name = 'PCA Feature Transformation'

    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X, y=None):
        transformed_X = self.pca.transform(X)
        return transformed_X

