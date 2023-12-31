from matplotlib import pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import VotingClassifier


def tune_hyperparams(x, y, model, params):
    # Exhaustive grid search and cross validation
    cv = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring='f1_macro')
    res = cv.fit(x, y)
    return res.best_params_


def plot_graph(y, yHat, ax):
    # Return metrics along with confusion matrix plot
    ConfusionMatrixDisplay.from_predictions(y, yHat, ax=ax,
                                            im_kw={'vmin': 0, 'vmax': 400},
                                            text_kw={'c': 'w', 'bbox': dict(facecolor='black', alpha=0.3)})
    plt.show()
    return {
        'accuracy': accuracy_score(y, yHat),
        'micro f-1': f1_score(y, yHat, average='micro'),
        'macro f-1': f1_score(y, yHat, average='macro'),
        'confusion_matrix': confusion_matrix(y, yHat)
    }


def get_metrics(x, y, model):
    yHat = model.predict(x)
    fig, ax = plt.subplots()
    ax.set(title=model.__class__.__name__ + " Confusion Matrix")
    return plot_graph(y, yHat, ax)


def get_metrics_from_prediction(y, yHat, model):
    fig, ax = plt.subplots()
    ax.set(title=model + " Confusion Matrix")
    return plot_graph(y, yHat, ax)


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
        graphs.append(RocCurveDisplay.from_estimator(
            model.get_model(), xTest, newTest))

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


def generate_multi_model_roc(models, xTrain, yTrain, xTest, yTest):
    classes = ["QSO", "GALAXY", "STAR"]
    for cls in classes:
        graphs = []
        for model in models:
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
            graphs.append(RocCurveDisplay.from_estimator(
                model.get_model(), xTest, newTest))

        plt.figure()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], 'k--')

        for index in range(0, len(graphs)):
            xdat = graphs[index].line_.get_xdata()
            ydat = graphs[index].line_.get_ydata()
            lbl = models[index].__class__.__name__ + " AUC = " + f'{graphs[index].roc_auc:.4f}'
            plt.plot(xdat, ydat, label=lbl)

        plt.legend()
        plt.xlim([0, 0.2])
        plt.ylim([0.8, 1.0])
        plt.title(cls + " ROC")
        plt.show()
