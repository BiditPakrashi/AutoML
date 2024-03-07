from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class DataConnector:

    TRAIN_DATA_PATH = "/Users/bpakra200/AiEdge/AutoML/data/train.csv"

    def put_data(self, testdata, prediction, file_name="/Users/bpakra200/AiEdge/AutoML/data/Prediction/predictions.csv"):
        df_predictions = pd.DataFrame({"id": testdata["id"], "prediction": prediction})
        df_predictions.to_csv(file_name, index=False)

    def get_data(self, path=None, target_label=None):
        if path is None:
            path = self.TRAIN_DATA_PATH

        df_train = pd.read_csv(path)
        if target_label is not None:
            # Putting feature variable to X
            X = df_train.drop([target_label], axis=1)

            # Putting response variable to y
            y = df_train[target_label]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7, test_size=0.3, random_state=27
            )
            return X_train, y_train, X_test, y_test
        return df_train
