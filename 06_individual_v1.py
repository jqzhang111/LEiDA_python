# LEiDA(Cabral 2017. Sci Rep.)-PART6: Individual V1
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
import itertools
from scipy import stats
from scipy.stats import ttest_ind
sns.set()



# get centroids for each brain states and sort by probability
# V1: (nsubjects * T) * N
# k: best cluster number

def EMP_BrainStates(V1, k, path):
    X = []
    for i in range(V1.shape[0]):
        X.append(V1[i])
    km = KMeans(n_clusters=k)
    km.fit(X)
    count = pd.Series(km.labels_).value_counts()
    center = pd.DataFrame(km.cluster_centers_, dtype=np.float)
    r= pd.concat([count, center], axis=1)
    np.savetxt('V1/'+path+'/'+str(k)+'/centroids_'+str(k)+'_count.txt', np.array(count), delimiter=' ')
    r.to_csv('V1/'+path+'/'+str(k)+'/centroids_'+str(k)+'_cluster.csv')
    data = r.values[np.argsort(-r.values[:, 0])]
    centroids = data[:, 1:] 
    return centroids





if __name__ == '__main__':
    path_mdd = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/MDD/'
    path_hc = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/HC/'


    # MDD1 Group
    f = open('mdd1.txt', 'r')
    m1 = f.readlines()

    V1 = np.zeros((11*230, 246))
    j = 0
    for i in range(len(m1)):
        path = path_mdd+m1[i].replace('\n','')+'.txt'
        vec = np.loadtxt(path)
        for k in range(vec.shape[0]):
            V1[j, :] = vec[k]
            j = j+1
    
    for k in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/MDD1/'+str(k))
        center = EMP_BrainStates(V1, k, 'MDD1')
        np.savetxt('V1/MDD1/'+str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')

    # MDD2 Group
    f = open('mdd2.txt', 'r')
    m2 = f.readlines()

    V1 = np.zeros((9*230, 246))
    j = 0
    for i in range(len(m2)):
        path = path_mdd+m2[i].replace('\n','')+'.txt'
        vec = np.loadtxt(path)
        for k in range(vec.shape[0]):
            V1[j, :] = vec[k]
            j = j+1
    
    for k in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/MDD2/'+str(k))
        center = EMP_BrainStates(V1, k, 'MDD2')
        np.savetxt('V1/MDD2/'+str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')

    # HC Group
    f = open('hc.txt', 'r')
    hc = f.readlines()

    V1 = np.zeros((20*230, 246))
    j = 0
    for i in range(len(hc)):
        path = path_hc+hc[i].replace('\n','')+'.txt'
        vec = np.loadtxt(path)
        for k in range(vec.shape[0]):
            V1[j, :] = vec[k]
            j = j+1
    
    for k in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/HC/'+str(k))
        center = EMP_BrainStates(V1, k, 'HC')
        np.savetxt('V1/HC/'+str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')


    # per subject
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject')
    for i in range(len(m1)):
        V1 = np.zeros((230, 246))
        path = path_mdd+m1[i].replace('\n','')+'.txt'
        V1 = np.loadtxt(path)
    
        for k in range(2, 21):
            os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+m1[i].replace('\n','')+'/'+str(k))
            center = EMP_BrainStates(V1, k, 'subject/'+m1[i].replace('\n',''))
            np.savetxt('V1/subject/'+m1[i].replace('\n','')+'/'+str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')
    

    for i in range(len(m2)):
        V1 = np.zeros((230, 246))
        path = path_mdd+m2[i].replace('\n','')+'.txt'
        V1 = np.loadtxt(path)
    
        for k in range(2, 21):
            os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+m2[i].replace('\n','')+'/'+str(k))
            center = EMP_BrainStates(V1, k, 'subject/'+m2[i].replace('\n',''))
            np.savetxt('V1/subject/'+m2[i].replace('\n','')+'/'+str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')


    for i in range(len(hc)):
        V1 = np.zeros((230, 246))
        path = path_hc+hc[i].replace('\n','')+'.txt'
        V1 = np.loadtxt(path)
    
        for k in range(2, 21):
            os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+hc[i].replace('\n','')+'/'+str(k))
            center = EMP_BrainStates(V1, k, 'subject/'+hc[i].replace('\n',''))
            np.savetxt('V1/subject/'+hc[i].replace('\n','')+'/'+str(k)+'/centroids_'+str(k)+'_cluster.txt', center, delimiter=' ')
