# LEiDA(Cabral 2017. Sci Rep.)-PART4: permutations test for empircal data
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
from mlxtend.evaluate import permutation_test


# K: number of cluster
# s1: name list of group1
# s2: name list of group2
def Permutation_Test(K, s1, s2, path):
    alpha = 0.05/K
    # permutation test for dwell time
    print('Test for dwell time, K='+str(K)+'.')
    dt1 = np.zeros((len(s1), K))
    dt2 = np.zeros((len(s2), K))
    for i in range(len(s1)):
        DT = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s1[i]).replace('\n', '')+'/'+str(K)+'/DT.txt')
        dt1[i, :] = (DT).T
    for i in range(len(s2)):
        DT = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s2[i]).replace('\n', '')+'/'+str(K)+'/DT.txt')
        dt2[i, :] = (DT).T

    p = []
    for j in range(K):
        p_value = permutation_test(dt1[:, j], dt2[:, j], num_rounds=10000)
        if p_value<alpha:
            print("State "+str(j+1)+" is obviously different.")
        p.append(p_value)
    np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+path+'/DT/'+str(K)+'/Dwell_Time_Test.txt', np.array(p), delimiter=' ')
    
    # permutation test for fractional occupancy
    print('Test for fractional occupancy, K='+str(K)+'.')
    
    

    fo1 = np.zeros((len(s1), K))
    fo2 = np.zeros((len(s2), K))
    for i in range(len(s1)):
        FO = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s1[i]).replace('\n', '')+'/'+str(K)+'/FO.txt')
        fo1[i, :] = (FO).T
    for i in range(len(s2)):
        FO = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s2[i]).replace('\n', '')+'/'+str(K)+'/FO.txt')
        fo2[i, :] = (FO).T

    p = []
    for j in range(K):
        p_value = permutation_test(fo1[:, j], fo2[:, j], num_rounds=10000)
        if p_value<alpha:
            print("State "+str(j+1)+" is obviously different.")
        p.append(p_value)
    np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+path+'/FO/'+str(K)+'/Fractional_Occupancy_Test.txt', np.array(p), delimiter=' ')

    # permutation test for cluster condition
    print('Test for cluster condition, K='+str(K)+'.')
    

    cluster1 = np.zeros((len(s1), K))
    cluster2 = np.zeros((len(s2), K))
    for i in range(len(s1)):
        Cluster = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s1[i]).replace('\n', '')+'/'+str(K)+'/V1_cluster.txt')
        Cluster = list(map(int, Cluster))
        Clu = []
        for m in range(K):
            Clu.append(Cluster.count(m))
        cluster1[i, :] = np.array(Clu)
    for i in range(len(s2)):
        Cluster = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s2[i]).replace('\n', '')+'/'+str(K)+'/V1_cluster.txt')
        Cluster = list(map(int, Cluster))
        Clu = []
        for m in range(K):
            Clu.append(Cluster.count(m))
        cluster2[i, :] = np.array(Clu)

    p = []
    for j in range(K):
        p_value = permutation_test(cluster1[:, j], cluster2[:, j], num_rounds=10000)
        if p_value<alpha:
            print("State "+str(j+1)+" is obviously different.")
        p.append(p_value)
    np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+path+'/Cluster/'+str(K)+'/Cluster_Condition_Test.txt', np.array(p), delimiter=' ')


    # permutation test for markov transition probability
    print('Test for markov transition probability, K='+str(K)+'.')
    

    ma1 = np.zeros((len(s1), K, K))
    ma2 = np.zeros((len(s2), K, K))
    for i in range(len(s1)):
        MA = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s1[i]).replace('\n', '')+'/'+str(K)+'/Markov_Matrix.txt')
        for m in range(K):
            for n in range(K):
                ma1[i][m][n] = MA[m][n]
    for i in range(len(s2)):
        MA = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+str(s2[i]).replace('\n', '')+'/'+str(K)+'/Markov_Matrix.txt')
        for m in range(K):
            for n in range(K):
                ma2[i][m][n] = MA[m][n]

    p = np.zeros((K, K))
    for m in range(K):
        for n in range(K): 
            p_value = permutation_test(ma1[:, m, n], ma2[:, m, n], num_rounds=10000)
            if p_value<alpha:
                print("Transition Probability from State "+str(m+1)+" to State "+str(n+1) +" is obviously different.")
            p[m][n] = p_value
    np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+path+'/Markov/'+str(K)+'/Transition_Probability_Test.txt', np.array(p), delimiter=' ')



if __name__ == '__main__':
    mdd1 = []
    mdd2 = []
    hc = []
    f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/mdd1.txt', 'r')
    mdd1 = f.readlines()
    f.close()
    f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/mdd2.txt', 'r')
    mdd2 = f.readlines()
    f.close()
    f = open('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/hc.txt', 'r')
    hc = f.readlines()
    f.close()
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/DT')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/DT')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/DT')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/Markov')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/Markov')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/Markov')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/Cluster')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/Cluster')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/Cluster')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/FO')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/FO')
    os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/FO')
    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/DT/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/FO/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/Cluster/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_MDD2/Markov/'+str(K))

        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/DT/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/FO/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/Cluster/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/Markov/'+str(K))

        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/DT/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/FO/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/Cluster/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/Markov/'+str(K))

        print("MDD Group1 VS MDD Group2 starting...")
        Permutation_Test(K, mdd1, mdd2,'MDD1_VS_MDD2')
        print("MDD Group1 VS MDD Group2 finished.")

        print("MDD Group1 VS HC Group starting...")
        Permutation_Test(K, mdd1, hc, 'MDD1_VS_HC')
        print("MDD Group1 VS HC Group finished.")

        print("MDD Group2 VS HC Group starting...")
        Permutation_Test(K, mdd2, hc, 'MDD2_VS_HC')
        print("MDD Group2 VS HC Group finished.")