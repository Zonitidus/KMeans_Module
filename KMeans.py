import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.widgets import Button
style.use('ggplot')
import numpy as np



data1 = pd.read_fwf('a1.txt', header = None)
#plt.scatter(data1[0].values, data1[1].values)

#Normalización residual, los parámetros de población son desconocidos
normalized_df = ((data1-data1.mean())/data1.std()).to_numpy() 



class KMeans():
  
  def __init__(self, k, max_iter=300):
        
    self.k = k 
    self.max_iter = max_iter
    self.centroids = {} #Diccionario que almacena los puntos del dataset que serán usados como centroides.
    self.clasified_data = {} #Diccionario que almacena las listas de puntos que pertenecen a cada centroide

    self.update_scatter_bttn = 0
    self.data = 0

  """
    Description: Define los centroides y los modifica según el promedio de los puntos asociados a ellos.
    param: data - np.array que define el dataset a analizar
    pre: k debe estar definido; data debe
    post: Se definen los clusters luego de max_iter iteraciones 
  """
  def fit(self, data):
    
    self.data = data
    self.centroids = {}

    #Se definen como centroides los primeros k elementos del dataset
    for centroid in range(self.k):
      self.centroids[centroid] = data[centroid] 

    #Se repite el proceso de clustering un número específico de veces (max_iter)

    self.step_by_step()
    plt.show()
    
  
  def step_by_step(self):
    
    self.clasified_data = {}

    #Se inicializa una lista vacía para cada cluster en el diccionario de datos clasificados
    for cluster in range(self.k):
      self.clasified_data[cluster] = []

    #Relaciona los datapoints con su cluster más cercano
    for data_point in self.data:
        
      min_distance = self.min_distance(self.data, data_point) 
      self.clasified_data[min_distance[1]].append(data_point) # Agrega el datapoint al diccionario que clasifica los datos en los diferentes clusters
        
      #Se guarda una copia de los centroides anteriores
      prev_centroids = dict(self.centroids)

      #Se redefinen los clusters con el promedio de los puntos que pertenecen a cada agrupación
      for key in self.clasified_data:
        
        #print(type(key), key)
        self.centroids[key] = np.average(self.clasified_data[key], axis = 0)
        
      #Partimos del supuesto que los clusters son óptimos
      optimized = True

      #Comparamos los clusters anteriores con los que acabamos da calcular. 
      # Si se mueven más del rango de toleracia, continuamos con las iteraciones (max_iter)
      for c in self.centroids:

        original_centroid = prev_centroids[c]
        current_centroid = self.centroids[c]
        if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > 0.001:
          #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
          optimized = False

      if optimized:
        break

    self.scatter()


  


  """
    Description: Distancia euclidiana entre puntos
    param: x - vector inicial
    param: y - vector final
    pre: Ambos puntos 'x' y 'y' están definidos
    post: Se calcula la distancia entre ambos puntos.
  """
  def euclidean_distance(self, x, y):
    #Distancia euclidiana
    return np.sqrt(np.sum(np.square(x-y)))

  def min_distance(self, data, datapoint):
    # Arreglo que guarda el cluster más cercano a un punto[1] y la distancia entre ellos[0]
    min_distance = [0,0]

    #Se recorren los clusters
    for i in range(self.k):
        
        #se obtiene la distancia entre el datapoint y el cluster i
        distance = self.euclidean_distance(self.centroids[i], datapoint)
        
        #Debe darse un valor la primera vez que se ejecuta
        if min_distance[0] == 0:
            min_distance[0] = distance
            min_distance[1] = i
        else:
            
            #Si la distancia hallada es menor, se actualiza el arreglo
            if min_distance[0] > distance:
              min_distance[0] = distance
              min_distance[1] = i

    return min_distance

  def init_widgets(self):
    
    axbttn1 = plt.axes([0.1, 0.1, 0.1, 0.1]) 

    self.update_scatter_bttn = Button( ax = axbttn1, 
                                    label = 'Update', 
                                    color = 'gray', 
                                    hovercolor = 'green')
        
    self.update_scatter_bttn.on_clicked(self.step_by_step)

  def scatter(self):
    for centroid in self.centroids:
      print(self.centroids[centroid][0], " --- ", self.centroids[centroid][1])
      plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],marker="x", color="k", s=150, linewidths=5)

    colors = 10*["g","r","c","b","k"]

    for classification in self.clasified_data:
      color = colors[classification]
      for featureset in self.clasified_data[classification]:
        plt.scatter(featureset[0], featureset[1], color=color, s=80, linewidths=2)



kmeans = KMeans(k = 6)
kmeans.fit(normalized_df)


















