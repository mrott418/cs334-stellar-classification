import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from util import tune_hyperparams, get_metrics, generate_roc

# Hyperparameters to tune for KNN
grid_params = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}


class KNN(object):
    def __init__(self):
        # Tuned hyperparams: {'metric': 'manhattan', 'n_neighbors': 6, 'weights': 'distance'}
        self.model = KNeighborsClassifier(n_neighbors=6, weights='distance', metric='manhattan')

    def tune(self, xTrain, yTrain):
        params = tune_hyperparams(xTrain, yTrain, KNeighborsClassifier(), grid_params)
        self.model = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                                          weights=params["weights"],
                                          metric=params["metric"])
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

    def get_model(self):
        return self.model


def main():
    # load dataset
    xTrain, yTrain, xTest, yTest = pd.read_csv("../data/xTrain.csv"), \
        np.ravel(pd.read_csv("../data/yTrain.csv")), \
        pd.read_csv("../data/xTest.csv"), \
        np.ravel(pd.read_csv("../data/yTest.csv"))

    model = KNN()
    model.train(xTrain, yTrain)

    metrics = model.metrics(xTest, yTest)
    print(metrics)
    # Accuracy 0.9541
    # Micro F-1: 0.9541
    # Macro F-1: 0.9489

    # generates ROC graphs
    # generate_roc(model, xTrain, yTrain, xTest, yTest)
    # plt.xlim([0, 0.2])
    # plt.ylim([0.8, 1.0])
    # plt.title("KNN ROC")
    # plt.show()


if __name__ == "__main__":
    main()
