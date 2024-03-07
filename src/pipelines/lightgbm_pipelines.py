from sklearn.metrics import roc_auc_score
from modules.hyperparameter_search import LightGBMTuner
from modules.feature_selection import DummyFeatureSelection, BorutaFeatureSelection
from modules.feature_transformations import CleanNANFeatureTransformation, PCAFeatureTransformation
from modules.models import LGBModel
import mlflow
import numpy as np


score_function_dictionary = {"auc": roc_auc_score}
lgbmodel = LGBModel()


class Pipeline(mlflow.pyfunc.PythonModel):

    def __init__(
        self,
        pipeline_name="lgb_pipeline",
        score_name="auc",
        steps=[
            CleanNANFeatureTransformation(),
            # DummyFeatureSelection()
            # PCAFeatureTransformation(60)
            # BorutaFeatureSelection()
        ],
        tuner=LightGBMTuner,
        source_model=lgbmodel,
    ):

        self.pipeline_name = pipeline_name
        self.score_name = score_name
        self.steps = steps
        self.tuner = tuner(source_model.model)
        self.model = source_model
        self.score = 0
        self.accepted_steps = []

    def fit(self, X_train, y_train, X_test, y_test):

        current_score = self.score
        for step in self.steps:
            X_train, X_test = self.fit_step(X_train, y_train, X_test, y_test, step)
            self.log_step_performance(step, X_train, y_train, X_test, y_test)
            if self.score > current_score:
                self.accepted_steps.append(step)
                current_score = self.score

        self.model = self.tune_model(X_train, y_train, X_test, y_test)
        self.log_step_performance(self.tuner, X_train, y_train, X_test, y_test)

    def predict(self, context, X):
        for step in self.accepted_steps:
            X = step.transform(X)

        return self.model.predict(X)

    def log_step_performance(self, step, X_train, y_train, X_text, y_test):
        with mlflow.start_run(run_name=step.name, nested=True):
            self.score = self.calculate_performance(X_train, y_train, X_text, y_test)
            mlflow.log_metric(self.score_name, self.score)

    def calculate_performance(self, X_train, y_train, X_test, y_test):

        np.int = int
        self.model.fit(
           X_train, y_train, X_test, y_test)
        y_scores = self.model.predict(X_test)
        score = score_function_dictionary[self.score_name](y_test, y_scores)
        return score

    def tune_model(self, X_train, y_train, X_test, y_test):
        model = self.tuner.fit(X_train, y_train, X_test, y_test)
        lgbmodel.model = model
        return lgbmodel

    def fit_step(self, X_train, y_train, X_test, y_test, step):

        X_train = step.fit(X_train, y_train)
        X_test = step.transform(X_test, y_test)

        return X_train, X_test
