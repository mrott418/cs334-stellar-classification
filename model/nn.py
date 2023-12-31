import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier

from util import tune_hyperparams, get_metrics, generate_roc

# Hyperparameters to tune for NN
grid_params = {
    "hidden_layer_sizes": [(100), (20, 10), (15, 10, 5), (10, 5)],
    "activation": ["logistic", "relu", "tanh"]
}


class NN(object):
    def __init__(self):
        # Tuned hyperparams: {'hidden_layer_sizes': (10, 5), 'activation': 'tanh'}
        self.model = MLPClassifier(hidden_layer_sizes=[10, 5],
                                   random_state=334,
                                   activation="tanh",
                                   max_iter=500)

    def tune(self, xTrain, yTrain):
        params = tune_hyperparams(xTrain, yTrain, MLPClassifier(random_state=334, max_iter=500), grid_params)
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

    model = NN()
    model.train(xTrain, yTrain)

    metrics = model.metrics(xTest, yTest)
    print(metrics)
    # Accuracy 0.9715
    # Micro F-1: 0.9715
    # Macro F-1: 0.9674

    # generates ROC graphs
    generate_roc(model, xTrain, yTrain, xTest, yTest)
    plt.xlim([0, 0.2])
    plt.ylim([0.8, 1.0])
    plt.title("Neural Network ROC")
    plt.show()


if __name__ == "__main__":
    main()
