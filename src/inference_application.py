from connections.data_connection import DataConnector
from connections.model_connection import ModelConnector


def run():
    data_connector = DataConnector()
    model_connector = ModelConnector()

    X_test = data_connector.get_data(
        path="/Users/bpakra200/AiEdge/AutoML/data/test.csv"
    )
    pipeline = model_connector.load_model()
    predictions = pipeline.predict(X_test)
    print(predictions)
    data_connector.put_data(X_test, predictions)

if __name__ == "__main__":
    run()
