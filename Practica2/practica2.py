import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from math import floor


#Activar para relizar pruebas con menos Datos
prueba = True



accidentes = pd.read_csv('accidentes_2013.csv')

if(prueba):
    subconjunto_accidentes = accidentes_2013_orig.sample(len(accidentes_2013_orig)//10)

#seleccionar solo accidentes entre las 6 y las 12 de la mañana
#subset = accidentes.loc[(accidentes['HORA']>=6) & (accidentes['HORA']<=12)]

#seleccionar accidentes no mortales
#subset = accidentes.loc[accidentes['TOT_MUERTOS']==0]

#seleccionar variables de interés para clustering
#usadas = ['HORA', 'DIASEMANA', 'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X = subset[usadas]
#seleccionar accidentes de tipo 'colisión de vehículos'
subset1 = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]
caso_estudio1 = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X1 = subset1[caso_estudio1]
subset2 = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Atropello")]
caso_estudio2 = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X2 = subset2[caso_estudio2]
subset3 = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Salida de la vía por la derecha")]
caso_estudio3 = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X3 = subset3[caso_estudio3]
subset4 = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Salida de la vía por la izquierda")]
caso_estudio4 = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
X4 = subset4[caso_estudio4]


for x in (X1,X2,X3,X4):
    X_normal = preprocessing.normalize(X, norm='l2')

clustering_algorithm = (
    ('k-means',k_means)
)

for name,algotithm in clustering_algorithm:
    print('{:19s}'.format(name),end='')
    t = t.time.time()
    cluster_predict = algorithms.fit_predict(X_normal)
    k = len(set(cluster_predict))
    print(': k: {3.0f}, '.format(k),end='')
    print("{:6.2f}")
