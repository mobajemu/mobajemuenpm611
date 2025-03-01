import json
from ml_code.data_load import load_data
from ml_code.pre_processing import preprocess_data
from ml_code.models import ModelFactory
from ml_code.train import train_and_evaluate
from ml_code.metrics import print_metrics
from test.unit_test import TestDataLoader

# open and load json file
with open('config.json') as config_file:
    config = json.load(config_file)

# load data
data = load_data("data/data.csv")
print(data.head())

# preprocess data
X_train, X_test, y_train, y_test = preprocess_data(data)

#get model
model = ModelFactory.get_model(config['model_type'])

# train and evaluate model
accuracy, cm, y_test, y_prob = train_and_evaluate(model, X_train, X_test, y_train, y_test)

# print metrics
print_metrics(accuracy, cm, y_test, y_prob)