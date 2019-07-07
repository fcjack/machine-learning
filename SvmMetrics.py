import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
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


df = pd.read_csv("binario_iris.csv")
y = np.array(df.iloc[:, -1])
x = np.array(df.iloc[:, :4])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, stratify=y)

# Fit The plot
model = svm.LinearSVC()
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

plt.show()
