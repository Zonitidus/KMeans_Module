import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

data1 = pd.read_fwf('a1.txt', header = None)
#plt.scatter(data1[0].values, data1[1].values)

#plt.show()



#df = data1.T
normalized_df = (data1-data1.mean())/data1.std()

print(data1)
X = normalized_df.to_numpy()

##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors = 10*["g","r","c","b","k"]


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:

                min_distance = self.min_distance(data, featureset) 
                self.classifications[min_distance[1]].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
                    

            if optimized:
                break




    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum(np.square(x-y)))

    def min_distance(self, data, datapoint):

        min_distance = [0,0]

        for i in range(self.k):
            
            distance = self.euclidean_distance(self.centroids[i], datapoint)

            if min_distance[0] == 0:
                min_distance[0] = distance
                min_distance[1] = i
            else:
                
                if min_distance[0] > distance:
                    min_distance[0] = distance
                    min_distance[1] = i

        return min_distance

clf = K_Means(k = 6)
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()