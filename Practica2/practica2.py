import time

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing
from math import floor


#Activar para relizar pruebas con menos Datos
prueba = True



accidentes = pd.read_csv('accidentes_2013.csv')

if(prueba):
    accidentes = accidentes.sample(len(accidentes)//20)

#seleccionar solo accidentes entre las 6 y las 12 de la mañana
#subset = accidentes.loc[(accidentes['HORA']>=6) & (accidentes['HORA']<=12)]

#seleccionar accidentes no mortales
#subset = accidentes.loc[accidentes['TOT_MUERTOS']==0]

#seleccionar variables de interés para clustering
#usadas = ['HORA', 'DIASEMANA', 'TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
# usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
# X = subset[usadas]
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

casos_de_estudio = [X1,X2,X3,X4]

k_means = cluster.KMeans(init='k-means++',n_clusters=4, n_init=5)
mbkm = cluster.MiniBatchKMeans(n_clusters=4)
ms = cluster.MeanShift()
spectral = cluster.SpectralClustering(n_clusters=4)
affinity_propagation = cluster.AffinityPropagation()
dbscan = cluster.DBSCAN(eps=0.1)
birch = cluster.Birch(n_clusters=4,threshold=0.1)
ward = cluster.AgglomerativeClustering(n_clusters=100, linkage='ward')

clustering_algorithm = (
    ('k-means',k_means),
    ('MiniBatkMeans',mbkm),
    ('MeanShift',ms),
    ('DBSCAN',dbscan),
    ('Birch',birch),
    ('SpectralClustering',spectral),
    ('Ward',ward)
)

for i,X in enumerate(casos_de_estudio):
    print("\nCaso de estudio "+str(i+1)+":\n")
    X_normal = preprocessing.normalize(X, norm='l2')

    for name,algorithm in clustering_algorithm:
        print('{:19s}'.format(name),end='')
        t = time.time()
        cluster_predict = algorithm.fit_predict(X_normal)
        tiempo = time.time() - t
        k = len(set(cluster_predict))
        print(": k: {:3.0f}, ".format(k),end='')
        print("{:6.2f} segundos, ".format(tiempo),end='')
        if(k>1) and (name is not 'Ward'):
            metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)

            metric_SH = metrics.silhouette_score(X_normal, cluster_predict,metric = 'euclidean',sample_size=floor(0.1*len(X1)),random_state=123456)
        else:
            metric_CH = 0
            metric_SH = 0
        print("CH Index: {:8.9f}, ".format(metric_CH),end='')
        print("SC: {:.5f}".format(metric_SH))
