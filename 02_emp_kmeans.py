# LEiDA(Cabral 2017. Sci Rep.)-PART2: K-means and centroids for each brain states
# author: zhangjiaqi(Smile.Z), CASIA, Brainnetome
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
from mpl_toolkits.mplot3d import Axes3D


# get Best K
# V1: (nsubjects * T) * N
# M: each k run M times for average score

def Decide_K(V1):
    X = []
    for i in range(V1.shape[0]):
        X.append(V1[i])
    vclust = ValidClust(k=list(range(2, 21)), methods = ['kmeans'])
    cvi_vals = vclust.fit_predict(X)
    cvi_vals.to_csv('DecideK/cluster.csv')
    vclust.plot()
    plt.savefig('DecideK/cluster.png')


# get centroids for each brain states and sort by probability
# V1: (nsubjects * T) * N
# k: best cluster number

def EMP_BrainStates(V1, k):
    X = []
    for i in range(V1.shape[0]):
        X.append(V1[i])
    km = KMeans(n_clusters=k)
    km.fit(X)

    count = pd.Series(km.labels_).value_counts()
    center = pd.DataFrame(km.cluster_centers_, dtype=np.float)
    r= pd.concat([count, center], axis=1)
    np.savetxt(str(k)+'/centroids_'+str(k)+'_count.txt', np.array(count), delimiter=' ')
    r.to_csv(str(k)+'/centroids_'+str(k)+'_cluster.csv')
    data = r.values[np.argsort(-r.values[:, 0])]
    centroids = data[:, 1:]
    
    plt.clf()
    vec = PCA(n_components=2).fit_transform(X)
    df2 = pd.DataFrame(vec)
    df2['labels'] = km.labels_
    visual_vec = k*[0]
    for m in range(k):
        visual_vec[m] = df2[df2['labels'] == m]
        plt.scatter(visual_vec[m][0], visual_vec[m][1], s=5)
    plt.savefig(str(k)+'/kmeans_visualize_2d_'+str(k)+'_cluster.png')
    plt.clf()
    
    fig = plt.figure()
    ax = Axes3D(fig)
    vec = PCA(n_components=3).fit_transform(X)
    df3 = pd.DataFrame(vec)
    df3['labels'] = km.labels_
    visual_vec = k*[0]
    for m in range(k):
        visual_vec[m] = df3[df3['labels'] == m]
        ax.scatter(visual_vec[m][0], visual_vec[m][1], visual_vec[m][2],s=5)
    plt.savefig(str(k)+'/kmeans_visualize_3d_'+str(k)+'_cluster.png')
    

    return centroids
    
    
if __name__ == '__main__':
    path_mdd = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/MDD/'
    path_hc = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/HC/'
    mdd_file = os.listdir(path_mdd)
    hc_file = os.listdir(path_hc)
    
    V1 = np.zeros((40*230, 246))
    
    i = 0
    for file in hc_file:
        path = path_hc+file
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    for file in mdd_file:
        path = path_mdd+file
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    Decide_K(V1)

    for k in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(k))
        center = EMP_BrainStates(V1, k)
        np.savetxt(str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')