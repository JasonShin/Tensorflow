import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from matplotlib import pyplot as plt


def main():
    columns = "age sex bmi map tc ldl hdl tch ltg glu".split()
    diabetes = datasets.load_diabetes()
    print(diabetes)
    print(columns)
    df = pd.DataFrame(diabetes.data, columns=columns)
    y = diabetes.target

    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(predictions)

    # the linear model
    print 'score', model.score(X_test, y_test)

    # KFold split example
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([1, 2, 3, 4])
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)
    print kf
    KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # leave one out cross validation
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    for train_index, test_index in loo.split(X):
        print('TRAIN:', train_index, 'TEST:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
