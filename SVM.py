import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def load_dataset(filename):
    df = pd.read_csv(filename)
    train_set = df.sample(frac=0.80, random_state=0)
    test_set = df.drop(train_set.index)
    return train_set, test_set


if __name__ == '__main__':
    # import some data to play with
    train, test = load_dataset("binario_iris.csv")
    train_target = train.iloc[:, -1]  # Target decision, real value for training dataset
    test_target = test.iloc[:, -1]  # target decision, real value for test dataset

    # Using pandas to get the two first parameters as features to analyze
    train_features = train.iloc[:, :2]
    test_features = test.iloc[:, :2]

    svc = svm.SVC(kernel='linear', C=1.0)
    linearSVC = svm.LinearSVC(C=1.0, max_iter=10000)
    rbfSvc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
    polySvc = svm.SVC(kernel='poly', degree=3, gamma='auto', C=1.0)
    models = [svc, linearSVC, rbfSvc, polySvc]
    models = (clf.fit(train_features, train_target) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Create the meshgrid and values for test dataset
    X0, X1 = test_features.iloc[:, 0].values, test_features.iloc[:, 1].values
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        polyPredicted = clf.predict(test_features)
        print("Accuracy on clf: %s", accuracy_score(test_target, polyPredicted))

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=test_target, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

plt.show()
