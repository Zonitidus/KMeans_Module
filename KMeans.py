import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data1 = pd.read_fwf('a1.txt', header = None)
plt.scatter(data1[0].values, data1[1].values)

plt.show()



df = data1.T

"""
x = df.values # is a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
"""

normalized_df = (df-df.mean())/df.std()


class KMeans():
  
  def __init__(self, k, max_iter=300):
    self.k = k
    self.max_iter = max_iter
    self.centroids = {}
    
    self.clasified_data = {}
  


  def fit(self, data):
    self.centroids = {}
    
    for centroid in range(self.k):

      self.centroids[centroid] = data[centroid]


    for iterations in range(self.max_iter):

      self.clasified_data = {}

      for cluster in range(self.k):
        self.clasified_data[cluster] = []

      for data_points in data:
        min_distance = self.min_distance(data)
        self.clasified_data[min_distance[1]].append(data_points)

    
      old_centroids = dict(self.centroids)

      for key in self.clasified_data:
        
        print(type(key), key)

        points = self.clasified_data[key]
        self.centroids[key] = np.average(self.clasified_data[key], axis = 0)
        
"""
      optimized = True

      for c in self.centroids:
        original_centroid = prev_centroids[c]
        current_centroid = self.centroids[c]
        if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > 0.001:
          print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
          optimized = False

      if optimized:
        break

    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

"""

      
        

  def euclidean_distance(self, x, y):
    return np.sqrt(np.sum(np.square(x-y)))

  def min_distance(self, data):

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

    return min_distance



kmeans = KMeans(15)
kmeans.fit(normalized_df)


classified_data = kmeans.clasified_data

print(classified_data)