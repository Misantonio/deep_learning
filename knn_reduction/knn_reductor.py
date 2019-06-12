import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

style.use('seaborn')


digits = load_digits()
X = digits.data
Y = digits.target
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

clf = TruncatedSVD(n_components=2)
X_2d = clf.fit_transform(X_train)

plt.scatter(X_2d[:,0], X_2d[:,1], c=Y_train, s=50, cmap=plt.cm.Paired)
plt.colorbar()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2 Components PC')
plt.show()


def compute_test(x_test, y_test, clf, cv):
    KFolds = KFold(x_test.shape[0], n_folds=cv)
    scores = []

    for i, j in KFolds:
        test_set = x_test[j]
        test_labels = y_test[j]
        scores.append(accuracy_score(test_labels, clf.predict(test_set)))
    return sum(scores)/len(scores)

k = np.arange(20)+1
parameters = {'n_neighbors':k}
knearest = KNeighborsClassifier()
clf = GridSearchCV(knearest, parameters, cv=30)

accuracy = []
params = []

for d in range(1, 11):
    svd = TruncatedSVD(n_components=d)

    if d < 64:
        X_fit_train = svd.fit_transform(X_train)
        X_fit_test = svd.transform(X_test)
    else:
        X_nl = X_train
        X_nl1 = X_test
    
    clf.fit(X_fit_train, Y_train)

    accuracy.append(compute_test(X_fit_test, Y_test, clf, 30))
    params.append(clf.best_params_['n_neighbors'])

print(accuracy)
print(params)