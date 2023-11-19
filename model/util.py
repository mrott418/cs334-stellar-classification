from matplotlib import pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV


def tune_hyperparams(x, y, model, params):
    # Exhaustive grid search and cross validation
    cv = GridSearchCV(model, params, cv=5, n_jobs=-1)
    res = cv.fit(x, y)
    return res.best_params_


def get_metrics(x, y, model):
    # Return accuracy and confusion matrix along with plot of the matrix
    yHat = model.predict(x)
    fig, ax = plt.subplots()
    ax.set(title=model.__class__.__name__ + " Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y, yHat, ax=ax)
    plt.show()
    return {
        'accuracy': accuracy_score(y, yHat),
        'confusion_matrix': confusion_matrix(y, yHat)
    }


def get_metrics_from_prediction(y, yHat, model):
    # Return accuracy and confusion matrix along with plot of the matrix
    fig, ax = plt.subplots()
    ax.set(title=model + " Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y, yHat, ax=ax)
    plt.show()
    return {
        'accuracy': accuracy_score(y, yHat),
        'confusion_matrix': confusion_matrix(y, yHat)
    }
