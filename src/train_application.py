from connections.data_connection import DataConnector
from connections.model_connection import ModelConnector
from pipelines.lightgbm_pipelines import Pipeline


def run():

    data_connector = DataConnector()
    model_connector = ModelConnector()
    pipeline = Pipeline()

    X_train, y_train, X_test, y_test = data_connector.get_data(target_label='default_ind')
    model_connector.log_model(pipeline, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    run()