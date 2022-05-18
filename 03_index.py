# LEiDA(Cabral 2017. Sci Rep.)-PART3: Index for each brain state
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


# Yeo7 Correlation with cluster
def Yeo7Corr(K):
    centers = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt')
    yeo7 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/00_Assign2Yeo7/output/DICE_Yeo-7_&_Brainnetome_res-1x1x1.txt')
    yeo7 = np.delete(yeo7, 0, axis=0)
    yeo7 = np.delete(yeo7, 0, axis=1)
    corr = np.zeros((K, 7))
    p_value = np.zeros((K, 7))
    for i in range(K):
        for j in range(7):
            centers[i, :][centers[i, :]<0] = 0
            corr[i][j] = stats.pearsonr(centers[i, :], yeo7[j, :])[0]
            p_value[i][j] = stats.pearsonr(centers[i, :], yeo7[j, :])[1]
    return corr, p_value


# Yeo17 Correlation with cluster
def Yeo17Corr(K):
    centers = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt')
    yeo17 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/00_Assign2Yeo7/output/DICE_Yeo-17_&_Brainnetome_res-1x1x1.txt')
    yeo17 = np.delete(yeo17, 0, axis=0)
    yeo17 = np.delete(yeo17, 0, axis=1)
    corr = np.zeros((K, 17))
    p_value = np.zeros((K, 17))
    for i in range(K):
        for j in range(17):
            centers[i, :][centers[i, :]<0] = 0
            corr[i][j] = stats.pearsonr(centers[i, :], yeo17[j, :])[0]
            p_value[i][j] = stats.pearsonr(centers[i, :], yeo17[j, :])[1]
    return corr, p_value


# Community for cluster
# K: number of cluster
def Community(K):
    f = open('brainnetome_subregions.txt', 'r')
    subregions = f.readlines()
    centers = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt')
    community_pname = {}
    community_pno = {}
    community_nname = {}
    community_nno = {}
    for i in range(K):
        pname = []
        pno = []
        nname = []
        nno = []
        for j in range(centers[i].shape[0]):
            if centers[i][j] >0:
                pname.append(subregions[j])
                pno.append(j)
            else:
                nname.append(subregions[j])
                nno.append(j)
        community_pname[i] = pname 
        community_pno[i] = pno
        community_nname[i] = nname 
        community_nno[i] = nno
    return community_pname, community_pno, community_nname, community_nno

# Sign for each subject
# V1: ntp * nregions 230*246
# K: number of cluster
def Sign(V1, K):
    cluster = np.zeros((V1.shape[0]))
    centers = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt')
    for i in range(V1.shape[0]):
        dis = []
        for j in range(K):
            dis.append(np.linalg.norm(V1[i]-centers[j]))
        cluster[i] = dis.index(min(dis))
    return cluster


# Fractional Occupancy for each subject
# V1: ntp * nregions 230*246
# cluster: sign for which cluster
# K: number of cluster
def FO(V1, cluster,K):
    fo = np.zeros((K))
    cluster = list(cluster)
    for i in range(K):
        fo[i] = cluster.count(i)/V1.shape[0]
    return fo


# Dwell Time for each subject
def DT(cluster, K):
    cluster = list(map(int, cluster))
    cnt = np.zeros((K))
    sl = np.zeros((K))
    dt = np.zeros((K))
    for key, group in itertools.groupby(cluster):
        cnt[key] += 1
        sl[key] += len(list(group))
    for i in range(K):
        dt[i] = 2*sl[i]/cnt[i]
    return dt


# Markov Chain Transition Probabilities

def transition_matrix(transitions, K):
    n = 1+ max(transitions) #number of states

    M = np.zeros((K, K))
    N = np.zeros((K, K))

    for (i,j) in zip(transitions,transitions[1:]):
        M[int(i)][int(j)] += 1

    #now convert to probabilities:
    for i in range(M.shape[0]):
        s = np.sum(M[i])
        if s>0:
            N[i, :] = M[i, :]/s
    return N


if __name__ == "__main__":
    for i in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i))
        corr, p_value = Yeo7Corr(i)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/yeo7corr.txt', corr, delimiter=' ')
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/yeo7pvalue.txt', p_value, delimiter=' ')

        corr, p_value = Yeo17Corr(i)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/yeo17corr.txt', corr, delimiter=' ')
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/yeo17pvalue.txt', p_value, delimiter=' ')

        community_pname, community_pno, community_nname, community_nno = Community(i)
        for j in range(i):
            if community_pname[j] != {}:
                f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/cluster_'+str(j)+'_positive_region_name.txt', 'a+')
                for name in community_pname[j]:
                    f.writelines(name)
                f.close()
                np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/cluster_'+str(j)+'_positive_region_no.txt', np.array(list(map(int, community_pno[j]))), delimiter=' ')
            if community_nname[j] != {}:
                f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/cluster_'+str(j)+'_negative_region_name.txt', 'a+')
                for name in community_nname[j]:
                    f.writelines(name)
                f.close()
                np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(i)+'/cluster_'+str(j)+'_negative_region_no.txt', np.array(list(map(int, community_nno[j]))), delimiter=' ')

        
    mdd_path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/MDD/'
    hc_path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/HC/'
    mdd_file = os.listdir(mdd_path)
    hc_file = os.listdir(hc_path)

    for sub in mdd_file:
        print(sub[:7]+' starting...')
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:7])
        V1 = np.loadtxt(mdd_path+sub)
        for K in range(2, 21):
            os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:7]+'/'+str(K))
            cluster = Sign(V1, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:7]+'/'+str(K)+'/V1_cluster.txt', np.array(cluster), delimiter=' ')
            fo = FO(V1, cluster, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:7]+'/'+str(K)+'/FO.txt', np.array(fo), delimiter=' ')
            dt = DT(cluster, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:7]+'/'+str(K)+'/DT.txt', np.array(dt), delimiter=' ')
            markov_matrix = transition_matrix(cluster, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:7]+'/'+str(K)+'/Markov_Matrix.txt', np.array(markov_matrix), delimiter=' ')
        print(sub[:7]+' finished.')



    for sub in hc_file:
        print(sub[:10]+' starting...')
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:10])
        V1 = np.loadtxt(hc_path+sub)
        for K in range(2, 21):
            os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:10]+'/'+str(K))
            cluster = Sign(V1, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:10]+'/'+str(K)+'/V1_cluster.txt', np.array(cluster), delimiter=' ')
            fo = FO(V1, cluster, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:10]+'/'+str(K)+'/FO.txt', np.array(fo), delimiter=' ')
            dt = DT(cluster, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:10]+'/'+str(K)+'/DT.txt', np.array(dt), delimiter=' ')
            markov_matrix = transition_matrix(cluster, K)
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+sub[:10]+'/'+str(K)+'/Markov_Matrix.txt', np.array(markov_matrix), delimiter=' ')
        print(sub[:10]+' finished.')


    print("MDD Group1 Starting...")
    V1 = np.zeros((11*230, 246))
    i = 0
    mdd1 = []
    f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/mdd1.txt', 'r')
    mdd1 = f.readlines()
    for file in mdd1:
        path = mdd_path+file.replace('\n', '')+'.txt'
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup1/'+str(K))
        cluster = Sign(V1, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup1/'+str(K)+'/V1_cluster.txt', np.array(cluster), delimiter=' ')
        fo = FO(V1, cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup1/'+str(K)+'/FO.txt', np.array(fo), delimiter=' ')
        dt = DT(cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup1/'+str(K)+'/DT.txt', np.array(dt), delimiter=' ')
        markov_matrix = transition_matrix(cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup1/'+str(K)+'/Markov_Matrix.txt', np.array(markov_matrix), delimiter=' ')

    print("MDD Group1 finished.")


    print("MDD Group2 Starting...")
    V1 = np.zeros((9*230, 246))
    i = 0
    mdd2 = []
    f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/mdd2.txt', 'r')
    mdd2 = f.readlines()
    for file in mdd2:
        path = mdd_path+file.replace('\n', '')+'.txt'
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup2/'+str(K))
        cluster = Sign(V1, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup2/'+str(K)+'/V1_cluster.txt', np.array(cluster), delimiter=' ')
        fo = FO(V1, cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup2/'+str(K)+'/FO.txt', np.array(fo), delimiter=' ')
        dt = DT(cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup2/'+str(K)+'/DT.txt', np.array(dt), delimiter=' ')
        markov_matrix = transition_matrix(cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/MDDGroup2/'+str(K)+'/Markov_Matrix.txt', np.array(markov_matrix), delimiter=' ')

    print("MDD Group2 finished.")


    print("HC Group Starting...")
    V1 = np.zeros((20*230, 246))
    i = 0
    for file in hc_file:
        path = hc_path+file
        vec = np.loadtxt(path)
        for j in range(vec.shape[0]):
            V1[i, :] = vec[j]
            i = i+1

    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/HCGroup/'+str(K))
        cluster = Sign(V1, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/HCGroup/'+str(K)+'/V1_cluster.txt', np.array(cluster), delimiter=' ')
        fo = FO(V1, cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/HCGroup/'+str(K)+'/FO.txt', np.array(fo), delimiter=' ')
        dt = DT(cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/HCGroup/'+str(K)+'/DT.txt', np.array(dt), delimiter=' ')
        markov_matrix = transition_matrix(cluster, K)
        np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/HCGroup/'+str(K)+'/Markov_Matrix.txt', np.array(markov_matrix), delimiter=' ')

    print("HC Group finished.")


