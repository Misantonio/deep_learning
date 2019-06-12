import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
import time

style.use('ggplot')

def plot_2D():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('2 component reduction', fontsize = 20)

    for target, color in zip(targets,colors):
        indicesToKeep = df['target'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1'], 
                    df.loc[indicesToKeep, 'PC2'], 
                    c = color,
                    s = 50,
                    alpha=0.4)
    ax.legend(targets)
    plt.show()

def plot_3D():
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_zlabel('PC3', fontsize = 15)
    for target, color in zip(targets, colors):
        indicesToKeep = df['target'] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1'], 
                   df.loc[indicesToKeep, 'PC2'], 
                   df.loc[indicesToKeep, 'PC3'],
                   c = color,
                   s = 50)
    ax.legend(targets)
    plt.show()

# LOAD MNIST DATA
mnist = fetch_mldata("MNIST original")
X = mnist.data / 255.0
Y = mnist.target

feat_cols = ['pixel' +str(i) for i in range(X.shape[1])]

df1 = pd.DataFrame(X, columns=feat_cols)
df1['target'] = Y
df1['target'] = df1['target'].apply(lambda i: str(i))
df1 = df1.sample(frac=1).reset_index(drop=True)

# PCA MODEL
n_samples = 15000

pca = PCA(n_components=50)
pca_result = pca.fit_transform(df1.loc[range(n_samples),feat_cols].values)
per_var = np.round(pca.explained_variance_ratio_*100)
labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]

# SCREE PLOT
plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_components = 0
per = 0
for i in per_var:
    per += i
    pca_components += 1
    if per > 75:
        break

print('\nPCA Components: {}\n'.format(pca_components))

# t-SNE MODEL
n_components = 2

time_start = time.time()
tsne = TSNE(n_components=n_components, verbose=1, perplexity=100, n_iter=300, learning_rate=1000)
tsne_results = tsne.fit_transform(pca_result[:,:pca_components])

print('t-SNE done! Elapsed time: {} seconds'.format(time.time() - time_start))

labels = ['PC'+str(x) for x in range(1, n_components+1)]
df = pd.DataFrame(data=tsne_results, columns=labels)
df['target'] = df1['target'].iloc[:n_samples]

targets = [str(float(i)) for i in range(0, 10)]
colors = ['#003366', '#71F79F', '#3DD6D0', '#cc9900', '#CE8147', '#cc99ff', '#90A9B7', '#49254F', '#FF8000', '#FFFF40']

if n_components == 2:
    plot_2D()
elif n_components == 2:
    plot_3D()