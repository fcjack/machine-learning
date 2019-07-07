import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


if __name__ == '__main__':
    df = pd.read_csv("binario_iris.csv")
    y = np.array(df.iloc[:, -1])
    x = np.array(df.iloc[:, :4])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)

    # Fit The plot
    model = svm.SVC()
    model.fit(x_train, y_train.ravel())

    # Calculate Test Prediction
    y_pred = model.predict(x_test)
    print(model.score(x_test, y_test.ravel()))

    # Calculate accuracy
    print(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(cm, index=[i for i in np.unique(y)],
                         columns=[i for i in np.unique(y)])
    plt.figure(figsize=(5, 5))
    sn.heatmap(df_cm, annot=True)

    # Build and plot graphs for the first two features
    # inside the dataset
    y = np.array(df.iloc[:, -1])
    x = np.array(df.iloc[:, :2])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)

    # Create the meshgrid and values for test dataset
    X0, X1 = x_test[:, 0], x_test[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    fig2 = plt.figure(figsize=(5, 5))
    ax = plt.subplot()

    # title for the plot
    title = 'SVC with polynomial (degree 3) kernel'

    model.fit(x_train, y_train)
    polyPredicted = model.predict(x_test)
    print("Accuracy on clf: %s", accuracy_score(y_test, polyPredicted))

    plot_contours(model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_title(title)

    plt.show()
