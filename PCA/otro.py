from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

style.use('ggplot')

def plot_2D():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1 - {0}%'.format(per_var[0]), fontsize = 15)
    ax.set_ylabel('PC2- {0}%'.format(per_var[1]), fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    for target, color in zip(targets,colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'PC1'], 
                   final_df.loc[indicesToKeep, 'PC2'], 
                   c = color,
                   s = 50)
    ax.legend(targets)

def plot_3D():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC1- {0}%'.format(per_var[0]), fontsize = 15)
    ax.set_ylabel('PC2- {0}%'.format(per_var[1]), fontsize = 15)
    ax.set_zlabel('PC3- {0}%'.format(per_var[2]), fontsize = 15)
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'PC1'], 
                   final_df.loc[indicesToKeep, 'PC2'], 
                   final_df.loc[indicesToKeep, 'PC3'],
                   c = color,
                   s = 50)
    ax.legend(targets)


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal length','sepal width','petal length','petal width','target']

df = pd.read_csv(url, names=columns)
X = df.loc[:, columns[:-1]].values
Y = df.loc[:, columns[-1]].values

scaled_X = StandardScaler().fit_transform(X)
pca = PCA()
pca_data = pca.fit_transform(scaled_X)
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]
pca_df = pd.DataFrame(pca_data, columns=labels)

num_components = 0
per = 0
for i in per_var:
    per += i
    num_components += 1
    if per > 97:
        break

# Scree plot
plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.show()

final_df = pd.concat([pca_df[labels[:num_components]], df['target']], axis=1)

# Distribution plot
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
if num_components == 2:
    plot_2D()
elif num_components == 3:
    plot_3D()
plt.show()


