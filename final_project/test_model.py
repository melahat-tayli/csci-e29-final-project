from .train import Train
from .preprocess_data import PreProcessing
from luigi import Task, Parameter
import os
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from functools import wraps

SHARED_RELATIVE_PATH = "data"

registered_models_and_scores = {}


def register(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        model_name, testing_score = func(*args, **kwargs)
        registered_models_and_scores[model_name + '_testing_score'] = testing_score
        return model_name, testing_score
    return wrapped

@register
def test_model_performance(model_name, loaded_model, x_test, y_test):
    acc_score = loaded_model.score(x_test, y_test)
    return model_name, acc_score


class TestModel(Task):
    data = Parameter(default="heart.csv")
    source_train = Parameter(default="train")
    model = Parameter(default=RandomForestClassifier)
    source_test = Parameter(default="test")

    def requires(self):
        return {'model_param': Train(self.data, self.source_train, self.model), 'test_data': PreProcessing(self.data, self.source_test)}

    def run(self):
        loaded_model = pickle.load(open(self.input()['model_param'].path, 'rb'))
        df_test = pd.read_csv(self.input()['test_data'].path)
        x_test = df_test.drop(columns=['target'])
        y_test = df_test['target']
        model_name = os.path.splitext(os.path.basename(self.input()['model_param'].path))[0]
        test_model_performance(model_name, loaded_model, x_test, y_test)
        self.show_registered()

    def show_registered(self):
        print('Registered_models_and_training_scores:', registered_models_and_scores)


