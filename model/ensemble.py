import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from util import tune_hyperparams, get_metrics

# Hyperparameters to tune for DT
grid_params = {
    "max_depth": range(1, 11),
    "min_samples_leaf": range(1, 21, 2),
    "n_estimators": [i * 10 for i in range(5, 16)],
    "criterion": ["gini", "entropy"],
}


class RF(object):
    def __init__(self):
        # Tuned hyperparams: {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 11, 'n_estimators': 60}
        self.model = RandomForestClassifier(random_state=334,
                                            max_depth=10,
                                            min_samples_leaf=11,
                                            criterion="gini",
                                            n_estimators=60,
                                            n_jobs=-1)

    def tune(self, xTrain, yTrain):
        params = tune_hyperparams(xTrain, yTrain, RandomForestClassifier(random_state=334), grid_params)
        print(params)
        return params

    def train(self, xTrain, yTrain, tune=False):
        if tune:
            self.tune(xTrain, yTrain)
        print(self.model.fit(xTrain, yTrain))

    def predict(self, xTest):
        return self.model.predict(xTest)

    def metrics(self, xTest, yTest):
        return get_metrics(xTest, yTest, self.model)


def main():
    # load dataset
    xTrain, yTrain, xTest, yTest = pd.read_csv("../data/xTrain.csv"), np.ravel(pd.read_csv("../data/yTrain.csv")), pd.read_csv(
        "../data/xTest.csv"), np.ravel(pd.read_csv("../data/yTest.csv"))

    model = RF()
    model.train(xTrain, yTrain)

    metrics = model.metrics(xTest, yTest)
    print(metrics)
    # Accuracy 0.9773


if __name__ == "__main__":
    main()
