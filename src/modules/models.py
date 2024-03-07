from lightgbm import LGBMClassifier


class LGBModel:
    def __init__(self):
        self.model = LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            feature_fraction=0.9,
            bagging_freq=5,
            verbose=1,
            early_stopping_rounds=2,
            n_jobs=8,
        )

    def get_categorical_features(self, df):
        categorical_features = df.select_dtypes(exclude=["int", "float"]).columns.tolist()
        return categorical_features

    def change_categorical_type(self, df, categorical_features):
        df[categorical_features] = df[categorical_features].astype("category")
        return df

    def fit(self, X_train, y_train, X_test, y_test):
        categorical_features = self.get_categorical_features(X_train)
        X_train = self.change_categorical_type(X_train, categorical_features)
        X_test = self.change_categorical_type(X_test, categorical_features)
        return self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            categorical_feature=categorical_features,
        )

    def predict(self, X, y=None):
        categorical_features = self.get_categorical_features(X)
        X = self.change_categorical_type(X, categorical_features)
        return self.model.predict(X)
