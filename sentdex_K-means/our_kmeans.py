import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np 
import random

style.use('ggplot')

X = np.array([[1,1.1],
              [1.1, 1.2],
              [0.9, 1.1],
              [1, .96], 
              [-0.3, 0.1],
              [-0.1, 0],
              [-0.2, 0],
              [0.5, 0.5],
              [0.42, 0.52],
              [0.49, 0.48], 
              [1.50, 1.5],
              [1.54, 1.49],
              [1.49, 1.49]])

colors = 10*["b", "g", "c", "k", "y", "r"]

class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.ss = 0
    
    def fit(self, data):
        self.centroids = {}
        # Initialize arbitrary centroids
        np.random.shuffle(data)
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}
             # Initialize empty classifications dictionary
            for i in range(self.k):
                self.classifications[i] = []

            # Compute distances for each point and classify it
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)

            # Compute new centroids based on the average of each classification
            for classification in self.classifications:
                self.centroids[classification] = np.mean(self.classifications[classification], axis=0)
            
            # Check the differnce between centroids
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
            
            if optimized:
                sums = []
                for i in range(len(self.centroids)):
                    res = self.classifications[i] - self.centroids[i]
                    sums.append(np.sum(res**2))
                self.ss = sum(sums)
                break


    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
elbow = 0

if elbow:
    vars = []
    for k in range(1, 6):
        clf = KMeans(k=k)
        clf.fit(X)
        vars.append(clf.ss)
    plt.scatter(range(1, 6), vars)
else:
    clf = KMeans(k=4)
    clf.fit(X)

    unknowns = np.array([[1,3],
                        [3, 8],
                        [6, 10], 
                        [1,0]])

    for i in range(len(clf.centroids)):
        plt.scatter(clf.centroids[i][0], clf.centroids[i][1], marker='o', color='k', s=150, linewidths=5)
        color = colors[i]
        for featurset in clf.classifications[i]:
            plt.scatter(featurset[0], featurset[1], marker='x', color=color, s=150, linewidths=5) 
    
    plt.title(clf.ss)

    for unknown in unknowns:
        classification = clf.predict(unknown)
        plt.scatter(unknown[0], unknown[1], color=colors[classification], marker='*', s=150) 

plt.show()