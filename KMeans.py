import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_fwf('a1.txt', header = None)
plt.scatter(data[0].values, data[1].values)

plt.show()


class KMeans():
  
  def __init__(self, k, max_iter=300):
    self.k = k
    self.max_iter = max_iter
  
  def fit(self, data):
    self.centroids = {}
    
    for centroid in range(self.k):
      self.centroids[centroid] = data[centroid]


    for iterations in range(self.max_iter):

      self.clasified_data={}

      for cluster in range(self.k):
        self.clasified_data[cluster] = []


      for data_points in data:
          
        min_distance = [0,0]

        for i in range(self.k):
            
            distance = self.euclidean_distance(self.centroids[i], data[i])

            if min_distance[0] == 0:
                min_distance[0] = distance
                min_distance[1] = i
            else:
                
                if min_distance[0] > distance:
                    min_distance[0] = distance
                    min_distance[1] = i

            




  def euclidean_distance(self, x, y):
    return np.sqrt(np.sum(np.square(x-y)))


