# LEiDA(Cabral 2017. Sci Rep.)-PART2: K-means and centroids for each brain states
# Author: zhangjiaqi(Smile.Z), CASIA, Brainnetome
import numpy as np 
from scipy.signal import hilbert
from scipy.spatial.distance import cosine
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import pandas as pd
from sklearn.decomposition import PCA
import os
from validclust import ValidClust


# get Best K
# V1: (nsubjects * T) * N
# M: each k run M times for average score

def Decide_K(V1):
    X = []
    for i in range(V1.shape[0]):
        X.append(V1[i])
    vclust = ValidClust(k=list(range(3, 11)), methods = ['kmeans'])
    cvi_vals = vclust.fit_predict(X)
    cvi_vals.to_csv('DecideK/cluster.csv')
    vclust.plot()
    plt.savefig('DecideK/cluster.png')
    
    
if __name__ == '__main__':
    path_mdd = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/MDD/'
    path_hc = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/HC/'
    mdd_file = os.listdir(path_mdd)
    hc_file = os.listdir(path_hc)
    
    V1 = np.zeros((20*230, 246))
    
    i = 0
    for file in hc_file:
        path = path_hc+file
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    i = 0
    for file in mdd_file:
        path = path_mdd+file
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    Decide_K(V1)
