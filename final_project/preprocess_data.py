from .load_data import TrainTestSplit
from luigi import Parameter, Task, LocalTarget
import os
import pandas as pd
from .preprocess_son import preprocess

SHARED_RELATIVE_PATH = "data"


class PreProcessing(Task):
    data = Parameter(default="heart.csv")
    train_or_test = Parameter(default="train")

    def requires(self):
        return TrainTestSplit(self.data, self.train_or_test)

    def output(self):
        path = os.path.join(os.path.abspath(SHARED_RELATIVE_PATH), 'preprocessed_' + self.train_or_test + '.csv')
        return LocalTarget(path)

    def run(self):
        df = pd.read_csv(self.input().path)
        df_preprocessed = preprocess(df, self.train_or_test)
        df_preprocessed.to_csv(self.output().path, index=False)

