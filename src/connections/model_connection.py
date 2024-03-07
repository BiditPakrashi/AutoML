import mlflow

LOGGED_MODEL = 'runs:/579cf889ebb746b698ccf47e6e94fc6f/lgb_pipeline'


class ModelConnector:

    def log_model(self, pipeline, X_train, y_train, X_test, y_test):
        with mlflow.start_run(run_name=pipeline.pipeline_name):
            pipeline.fit(X_train, y_train, X_test, y_test)
            mlflow.pyfunc.log_model(pipeline.pipeline_name, python_model=pipeline)

    def load_model(self):
        return mlflow.pyfunc.load_model(LOGGED_MODEL)