from skopt import BayesSearchCV
import numpy as np

class XGBoostTuner:

    SEARCH_SPACE = {
        "eta": (0.0, 1.0, "uniform"),
        "gamma": (0.0, 50, "uniform"),
        "max_depth": (1, 15),
        "min_child_weight": (0.0, 10.0, "uniform"),
        "max_delta_step": (0.0, 10.0, "uniform"),
        "subsample": (0.0, 1.0, "uniform"),
        "lambda": (0.0, 5.0, "uniform"),
        "alpha": (0.0, 5.0, "uniform"),
    }

    name = "Hyperparameter search"

    def __init__(self, model) -> None:

        self.model = model
        self.optimizer = BayesSearchCV(
            model,
            self.SEARCH_SPACE,
            n_iter=32,
            cv=3,
        )
        self.best_params = None

    def fit(self, X, y):
        self.optimizer.fit(X, y)
        self.best_params = self.optimizer.best_params_

        return self.optimizer.best_estimator_


class LightGBMTuner:

    SEARCH_SPACE = {
        "boosting_type": ["gbdt", "dart"],
        "num_leaves": (10, 100),
        "learning_rate": (0.01, 0.1, "log-uniform"),
        "subsample": (0.5, 1.0, "uniform"),
        "colsample_bytree": (0.5, 1.0, "uniform"),
        "reg_alpha": (0.0, 1.0, "uniform"),
        "reg_lambda": (0.0, 1.0, "uniform"),
        "bagging_fraction": (0.5, 1.0, "uniform"),
    }

    name = "Hyperparameter search for LightGBM"

    def __init__(self, model) -> None:

        self.model = model
        self.optimizer = BayesSearchCV(
            model,
            self.SEARCH_SPACE,
            n_iter=50,
            scoring="accuracy",
            cv=5,
            random_state=42,
        )
        self.best_params = None
        self.best_score = None

    def get_categorical_features(self, df):
        categorical_features = df.select_dtypes(exclude=["int", "float"]).columns.tolist()
        return categorical_features

    def change_categorical_type(self, df, categorical_features):
        df[categorical_features] = df[categorical_features].astype("category")
        return df

    def fit(self, X_train, y_train, X_test, y_test):
        # categorical_features may not required here
        categorical_features = self.get_categorical_features(X_train)
        X_train = self.change_categorical_type(X_train, categorical_features)
        X_test = self.change_categorical_type(X_test, categorical_features)
        np.int = int
        self.optimizer.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=categorical_features
        )
        self.best_params = self.optimizer.best_params_
        self.best_score = self.optimizer.best_score_
        return self.optimizer.best_estimator_
