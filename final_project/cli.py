from luigi import build
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .load_data import DownloadRawData
from .test_model import TestModel


def main():
    build(
        [
            DownloadRawData(),
            TestModel(model=RandomForestClassifier, source_test="test"),
            TestModel(model=LogisticRegression, source_test="test"),
        ],
        local_scheduler=True,
    )
