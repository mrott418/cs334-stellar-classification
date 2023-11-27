from matplotlib import pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import VotingClassifier


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
    ConfusionMatrixDisplay.from_predictions(y, yHat, ax=ax,
                                            im_kw={'vmin': 0, 'vmax': 400},
                                            text_kw={'c': 'w', 'bbox': dict(facecolor='black', alpha=0.3)})
    plt.show()
    return {
        'accuracy': accuracy_score(y, yHat),
        'confusion_matrix': confusion_matrix(y, yHat)
    }


def get_metrics_from_prediction(y, yHat, model):
    # Return accuracy and confusion matrix along with plot of the matrix
    fig, ax = plt.subplots()
    ax.set(title=model + " Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y, yHat, ax=ax,
                                            im_kw={'vmin': 0, 'vmax': 400},
                                            text_kw={'c': 'w', 'bbox': dict(facecolor='black', alpha=0.3)})
    plt.show()
    return {
        'accuracy': accuracy_score(y, yHat),
        'confusion_matrix': confusion_matrix(y, yHat)
    }

def generate_roc(model, xTrain, yTrain, xTest, yTest):
    classes = ["QSO", "GALAXY", "STAR"]
    graphs = []
    for cls in classes:
        newTrain = []
        newTest = []

        # converts labels to binary
        for index in range(0, len(yTrain)):
            if yTrain[index] == cls:
                newTrain.append(1)
            else:
                newTrain.append(0)

        for index in range(0, len(yTest)):
            if yTest[index] == cls:
                newTest.append(1)
            else:
                newTest.append(0)

        model.train(xTrain, newTrain)
        graphs.append(RocCurveDisplay.from_estimator(model.model, xTest, newTest))

    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--')

    for index in range(0, len(graphs)):
        xdat = graphs[index].line_.get_xdata()
        ydat = graphs[index].line_.get_ydata()
        lbl = classes[index] + " AUC = " + f'{graphs[index].roc_auc:.4f}'
        plt.plot(xdat, ydat, label=lbl)

    plt.legend()

def generate_roc_ensemble(model, xTrain, yTrain, xTest, yTest):
    classes = ["QSO", "GALAXY", "STAR"]
    graphs = []
    for cls in classes:
        newTrain = []
        newTest = []

        # converts labels to binary
        for index in range(0, len(yTrain)):
            if yTrain[index] == cls:
                newTrain.append(1)
            else:
                newTrain.append(0)

        for index in range(0, len(yTest)):
            if yTest[index] == cls:
                newTest.append(1)
            else:
                newTest.append(0)

        model.train(xTrain, newTrain)
        pred = model.predict(xTest)
        graphs.append(RocCurveDisplay.from_predictions(newTest, pred, drop_intermediate=False))

    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--')

    for index in range(0, len(graphs)):
        xdat = graphs[index].line_.get_xdata()
        ydat = graphs[index].line_.get_ydata()
        lbl = classes[index] + " AUC = " + f'{graphs[index].roc_auc:.4f}'
        plt.plot(xdat, ydat, label=lbl)

    plt.legend()