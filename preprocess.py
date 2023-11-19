import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def split_data(df):
    X = df[["u", "g", "r", "i", "z", "redshift"]]
    y = df["class"]
    return train_test_split(X, y, random_state=334, test_size=0.2)


def mean_normalize(df):
    # mean normalization
    return (df - df.mean()) / df.std()


def minmax_normalize(df):
    # mean normalization
    return (df - df.min()) / (df.max() - df.min())


def main():
    # load whole dataset
    dataset = pd.read_csv("./data/star_classification.csv")

    # split dataset
    xTrain, xTest, yTrain, yTest = split_data(dataset)

    fig, ax = plt.subplots()
    ax.set(title="Pearson Correlation of Training Data")
    sns.heatmap(xTrain.corr(), annot=True, ax=ax)
    plt.show()

    # save file
    xTrain.to_csv("./data/xTrain.csv", index=False)
    xTest.to_csv("./data/xTest.csv", index=False)
    yTrain.to_csv("./data/yTrain.csv", index=False)
    yTest.to_csv("./data/yTest.csv", index=False)

    # DON'T NEED THESE

    # mean normalization
    # mean_normalize(xTrain).to_csv("xTrainMean.csv", index=False)
    # mean_normalize(xTest).to_csv("xTestMean.csv", index=False)

    # min-max normalization
    # minmax_normalize(xTrain).to_csv("xTrainMinMax.csv", index=False)
    # minmax_normalize(xTest).to_csv("xTestMinMax.csv", index=False)


if __name__ == "__main__":
    main()
