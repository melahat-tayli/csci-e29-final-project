from .preprocess_data import PreProcessing
from luigi import Parameter, Task, format, LocalTarget
from functools import wraps
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import pickle

SHARED_RELATIVE_PATH = "data"

registered_models_and_scores = {}


def register(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        clf, model_name, training_score = func(*args, **kwargs)
        registered_models_and_scores[model_name + '_training_score'] = training_score
        return clf, model_name, training_score
    return wrapped

@register
def fit_model(model, x_train, y_train):
    clf = model()
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    return clf, model.__name__, accuracy_score(y_train, y_train_pred)


class Train(Task):
    data = Parameter(default="heart.csv")
    train_or_test = Parameter("train")
    model = Parameter(default=RandomForestClassifier)

    def requires(self):
        return PreProcessing(self.data, self.train_or_test)

    def output(self):
        # returns the trained model
        path = os.path.join(os.path.abspath(SHARED_RELATIVE_PATH), self.model.__name__ + '_parameters' + '.pkl')
        return LocalTarget(path, format=format.Nop)

    def run(self):
        df = pd.read_csv(self.input().open('r'))
        x_train = df.drop(columns=['target'])
        y_train = df['target']
        clf, model_name, acc_score = fit_model(self.model, x_train, y_train)
        with open(self.output().path, 'wb') as f:
            pickle.dump(clf, f)
        self.show_registered()

    def show_registered(self):
        print('Registered_models_and_training_scores:', registered_models_and_scores)







