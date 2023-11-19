import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from util import tune_hyperparams, get_metrics_from_prediction

# Hyperparameters to tune for XGB
# Tune only one parameter at a time
grid_params = {
    "max_depth": range(1, 11),
    "min_child_weight": range(1, 6),
    "gamma": [i / 20.0 for i in range(0, 11)],
}


class XGB(object):
    def __init__(self):
        # Tuned hyperparams: {'max_depth': 10, 'min_child_weight': 5, 'gamma': 0.5}
        self.model = XGBClassifier(random_state=334,
                                   max_depth=10,
                                   min_child_weight=5,
                                   gamma=0.5,
                                   learning_rate=0.1)

    def tune(self, xTrain, yTrain):
        params = tune_hyperparams(xTrain, yTrain, XGBClassifier(random_state=334), grid_params)
        print(params)
        return params

    def train(self, xTrain, yTrain, tune=False):
        if tune:
            self.tune(xTrain, yTrain)
        print(self.model.fit(xTrain, yTrain))

    def predict(self, xTest):
        return self.model.predict(xTest)


def main():
    # load dataset
    xTrain, yTrain, xTest, yTest = pd.read_csv("../data/xTrain.csv"), np.ravel(pd.read_csv("../data/yTrain.csv")), pd.read_csv(
        "../data/xTest.csv"), np.ravel(pd.read_csv("../data/yTest.csv"))

    le = LabelEncoder()
    yTrainEncoded = le.fit_transform(yTrain)
    model = XGB()
    model.train(xTrain, yTrainEncoded)

    pred = le.inverse_transform(model.predict(xTest))
    metrics = get_metrics_from_prediction(yTest, pred, "XGBClassifier")
    print(metrics)
    # Accuracy 0.97815


if __name__ == "__main__":
    main()
