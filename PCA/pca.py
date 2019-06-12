import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def generate_data():
    genes = ['gene'+str(i) for i in range(1, 101)]

    wt = ['wt'+str(i) for i in range(1, 6)]
    ko = ['ko'+str(i) for i in range(1, 6)]
    data = pd.DataFrame(columns=[*wt, *ko], index=genes)
    for gene in data.index:
        data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 15), size=5)
        data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 15), size=5)

    return data

def scale_data(data):
    scaled_data = preprocessing.scale(data.T) # mean = 0, std_dev = 1
    return scaled_data

if __name__ == '__main__':
    df = generate_data()
    scaled_data = scale_data(df)
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data) # Data reduction

    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    pca_df = pd.DataFrame(pca_data, columns=labels)

    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('My PCA graph')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))

    for sample in pca_df.index:
        plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
    
    plt.show()