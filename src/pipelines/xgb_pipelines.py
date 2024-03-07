from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from modules.hyperparameter_search import XGBoostTuner
from modules.feature_selection import DummyFeatureSelection
from modules.feature_transformations import DummyFeatureTransformation
import mlflow


score_function_dictionary = {
    'auc': roc_auc_score
}

xgb_clf = XGBClassifier(
    n_estimators=100,
    objective='binary:logistic', 
    n_jobs=-1
)

class Pipeline(mlflow.pyfunc.PythonModel):

    def __init__(
            self, 
            pipeline_name='xgb_pipeline', 
            score_name='auc', 
            steps=[
                DummyFeatureTransformation(),
                DummyFeatureSelection()
            ], 
            tuner=XGBoostTuner, 
            source_model=xgb_clf
        ):
        
        self.pipeline_name = pipeline_name
        self.score_name = score_name
        self.steps = steps
        self.tuner = tuner(source_model)
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

        self.model = self.tune_model(X_train, y_train)
        self.log_step_performance(self.tuner, X_train, y_train, X_test, y_test)

    def predict(self, context, X):
        for step in self.accepted_steps:
            X = step.transform(X) 

        return self.model.predict_proba(X)[:, 1]
    
    def log_step_performance(
            self, 
            step, 
            X_train, 
            y_train, 
            X_text, 
            y_test
        ): 
        with mlflow.start_run(run_name=step.name, nested=True):
            self.score = self.calculate_performance(X_train, y_train, X_text, y_test) 
            mlflow.log_metric(self.score_name, self.score)

    def calculate_performance(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        y_scores = self.model.predict_proba(X_test)[:, 1]
        score = score_function_dictionary[self.score_name](y_test, y_scores)
        return score

    def tune_model(self, X_train, y_train): 
        model = self.tuner.fit(X_train, y_train)
        return model

    def fit_step(self, X_train, y_train, X_test, y_test, step):

        X_train = step.fit(X_train, y_train)
        X_test = step.transform(X_test, y_test)

        return X_train, X_test