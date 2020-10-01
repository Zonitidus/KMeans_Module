import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data1 = pd.read_fwf('a1.txt', header = None)
#plt.scatter(data1[0].values, data1[1].values)

#plt.show()



df = data1.T

"""
x = df.values # is a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
"""

normalized_df = (df-df.mean())/df.std()


class KMeans():
  

  """
    Description: Inicializa un nuevo objeto de  
    param: k - número de clusters
    param: max_iter - número máximo de iteraciones
    pre: k esté definida
    post: se crea un nuevo objeto de tipo Kmeans
  """
  def __init__(self, k, max_iter=300):
    self.k = k 
    self.max_iter = max_iter
    self.centroids = {} #Diccionario que almacena los puntos del dataset que serán usados como centroides.
    
    self.clasified_data = {} #Diccionario que almacena las listas de puntos que pertenecen a cada centroide
  


  """
    Description: Entrena el modelo definiendo los centroides y modificándolos según la posición de los puntos asociados a dichos centroides.
    param: data - Dataset a analizar
    pre: k debe estar definido; data debe
    post: Se definen los clusters luego de max_iter iteraciones 
  """
  def fit(self, data):
    
    self.centroids = {}

    #Se definen como centroides los primeros k elementos del dataset
    for centroid in range(self.k):

      self.centroids[centroid] = data[centroid] 

    #Se repite el proceso de clustering un número específico de veces (max_iter)
    for iterations in range(self.max_iter):

      self.clasified_data = {}

      #Se inicializa una lista vacía para cada cluster en el diccionario de datos clasificados
      for cluster in range(self.k):
        self.clasified_data[cluster] = []

      #Relaciona los datapoints con su cluster más cercano
      for data_points in data:
        min_distance = self.min_distance(data) 
        self.clasified_data[min_distance[1]].append(data_points) # Agrega el datapoint al diccionario que clasifica los datos en los diferentes clusters

      #Se guarda una copia de los centroides anteriores
      prev_centroids = dict(self.centroids)

      #Se redefinen los clusters con el promedio de los puntos que pertenecen a cada agrupación
      for key in self.clasified_data:
        
        #print(type(key), key)

        points = self.clasified_data[key]
        self.centroids[key] = np.average(self.clasified_data[key], axis = 0)
        
      #Partimos del supuesto que los clusters son óptimos
      optimized = True

      #Comparamos los clusters anteriores con los que acabamos da calcular. 
      # Si no se encuentran en el rango de tolerancia, continuamos con las iteraciones (max_iter)
      for c in self.centroids:

        original_centroid = prev_centroids[c]
        current_centroid = self.centroids[c]
        if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > 0.001:
          #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
          optimized = False

      if optimized:
        break




  """
    Description: 
    param:
    pre:
    post:
  """
  def predict(self,data):
      distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
      classification = distances.index(min(distances))
      return classification

  """
    Description:
    param:
    pre:
    post:
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





























X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])




km = KMeans(2)
km.fit(X)

print((km.clasified_data[0]))

#plt.scatter(km.clasified_data[0][0], km.clasified_data[0][1])
#plt.scatter(km.clasified_data[1][0], km.clasified_data[1][1])

#plt.plot(km.clasified_data[0], "g")
#plt.plot(km.clasified_data[1], "r")
        
#plt.show()