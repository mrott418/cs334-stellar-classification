import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

from dt import DT
from knn import KNN
from nn import NN
from rf import RF
from util import get_metrics_from_prediction, generate_roc, generate_multi_model_roc
from xgb import XGB


class Ensemble(object):
    def __init__(self):
        self.ensemble = None
        self.estimators = None
        self.knn, self.nn, self.dt, self.rf, self.xgb = KNN(), NN(), DT(), RF(), XGB()

    def train(self, xTrain, yTrain):
        self.estimators = [
            ('knn', self.knn.get_model()),
            ('nn', self.nn.get_model()),
            ('dt', self.dt.get_model()),
            ('rf', self.rf.get_model()),
            ('xgb', self.xgb.get_model()),
        ]
        self.ensemble = VotingClassifier(self.estimators, voting='soft', weights=[1, 2, 3, 5, 5])
        self.ensemble.fit(xTrain, yTrain)

    def predict(self, xTest):
        return self.ensemble.predict(xTest)

    def get_model(self):
        return self.ensemble

    def get_models(self):
        return [self.knn, self.nn, self.dt, self.rf, self.xgb, self]


def main():
    # load dataset
    xTrain, yTrain, xTest, yTest = pd.read_csv("../data/xTrain.csv"), \
        np.ravel(pd.read_csv("../data/yTrain.csv")), \
        pd.read_csv("../data/xTest.csv"), \
        np.ravel(pd.read_csv("../data/yTest.csv"))

    le = LabelEncoder()
    yTrainEncoded = le.fit_transform(yTrain)

    ensemble = Ensemble()
    ensemble.train(xTrain, yTrainEncoded)

    pred = le.inverse_transform(ensemble.predict(xTest))

    metrics = get_metrics_from_prediction(yTest, pred, "Ensemble")
    print(metrics)
    # Accuracy 0.9784
    # Micro F-1: 0.9784
    # Macro F-1: 0.9747

    # generates ROC graphs
    # generate_roc(ensemble, xTrain, yTrain, xTest, yTest)
    # plt.xlim([0, 0.2])
    # plt.ylim([0.8, 1.0])
    # plt.title("Ensemble ROC")
    # plt.show()

    # generates ROC graphs for each type
    generate_multi_model_roc(ensemble.get_models(), xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()
