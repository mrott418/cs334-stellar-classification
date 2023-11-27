import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from dt import DT
from knn import KNN
from nn import NN
from rf import RF
from util import get_metrics_from_prediction
from xgb import XGB


class Ensemble(object):
    def __init__(self):
        self.ensemble = None
        self.estimators = None
        self.knn, self.nn, self.dt, self.rf, self.xgb = KNN(), NN(), DT(), RF(), XGB()

    def train(self, xTrain, yTrain):
        self.knn.train(xTrain, yTrain)
        self.nn.train(xTrain, yTrain)
        self.dt.train(xTrain, yTrain)
        self.rf.train(xTrain, yTrain)
        self.xgb.train(xTrain, yTrain)
        self.estimators = [
            ('knn', self.knn.get_model()),
            ('nn', self.nn.get_model()),
            ('dt', self.dt.get_model()),
            ('rf', self.rf.get_model()),
            ('xgb', self.xgb.get_model()),
        ]
        self.ensemble = VotingClassifier(self.estimators)
        self.ensemble.fit(xTrain, yTrain)

    def predict(self, xTest):
        return self.ensemble.predict(xTest)


def main():
    # load dataset
    xTrain, yTrain, xTest, yTest = pd.read_csv("../data/xTrain.csv"), np.ravel(
        pd.read_csv("../data/yTrain.csv")), pd.read_csv(
        "../data/xTest.csv"), np.ravel(pd.read_csv("../data/yTest.csv"))

    le = LabelEncoder()
    yTrainEncoded = le.fit_transform(yTrain)

    ensemble = Ensemble()
    ensemble.train(xTrain, yTrainEncoded)

    pred = le.inverse_transform(ensemble.predict(xTest))
    metrics = get_metrics_from_prediction(yTest, pred, "Ensemble")
    print(metrics)
    # Accuracy 0.97825


if __name__ == "__main__":
    main()
