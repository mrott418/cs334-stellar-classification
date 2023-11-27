import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from util import tune_hyperparams, get_metrics, generate_roc

# Hyperparameters to tune for DT
grid_params = {
    "criterion": ["gini", "entropy"],
    "max_depth": range(3, 11),
    "min_samples_leaf": range(1, 31, 2),
}


class DT(object):
    def __init__(self):
        # Tuned hyperparams: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 19}
        self.model = DecisionTreeClassifier(random_state=334,
                                            criterion="entropy",
                                            max_depth=10,
                                            min_samples_leaf=19)

    def tune(self, xTrain, yTrain):
        params = tune_hyperparams(xTrain, yTrain, DecisionTreeClassifier(random_state=334), grid_params)
        self.model = DecisionTreeClassifier(random_state=334,
                                            criterion=params["criterion"],
                                            max_depth=params["max_depth"],
                                            min_samples_leaf=params["min_samples_leaf"])
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
    xTrain, yTrain, xTest, yTest = pd.read_csv("../data/xTrain.csv"), np.ravel(pd.read_csv("../data/yTrain.csv")), pd.read_csv(
        "../data/xTest.csv"), np.ravel(pd.read_csv("../data/yTest.csv"))

    model = DT()

    # generates ROC graphs
    generate_roc(model, xTrain, yTrain, xTest, yTest)
    plt.xlim([0, 0.2])
    plt.ylim([0.8, 1.0])
    plt.title("Decision Tree ROC")
    plt.show()
    #metrics = model.metrics(xTest, yTest)
    #print(metrics)
    # Accuracy 0.9755


if __name__ == "__main__":
    main()
