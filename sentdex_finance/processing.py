import numpy as np
import pandas as pd 
import _pickle as pickle 
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import style

style.use('seaborn-darkgrid')

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days+1):
        col = '{}_{}d'.format(ticker, i)
        df[col] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.04
    for col in cols:
        if col > requirement:
            return 1 # buy
        elif col < -requirement:
            return -1 # sell
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, *[df['{}_{}d'.format(ticker, i)] for i in range(1, 8)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread: ', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change() 
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    Y = vals = df['{}_target'.format(ticker)].values

    return X, Y, df

def do_ml(ticker):
    X, Y, df = extract_featuresets(ticker)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.25)

    # clf = neighbors.KNeighborsClassifier()
    clf = svm.LinearSVC()
    # clf = VotingClassifier([('lsvc', svm.LinearSVC()), 
    #                         ('knn', neighbors.KNeighborsClassifier()), 
    #                         ('rfor', RandomForestClassifier())])
    print("Start fit")
    clf.fit(X_train, Y_train)
    confidence = clf.score(X_test, Y_test)
    print('Accuracy', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread: ', Counter(predictions))
    return confidence


if __name__ == "__main__":
    do_ml('ABT')

