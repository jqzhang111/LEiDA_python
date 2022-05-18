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
from scipy.stats import ttest_ind


# K: number of cluster
# s1: name list of group1
# s2: name list of group2
def Permutation_Test(K, s1, s2, path):
    alpha = 0.05
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
        d1 = list(dt1[:, j])
        d2 = list(dt2[:, j])
        newd1 = [x for x in d1 if x != 'nan']
        newd2 = [x for x in d2 if x != 'nan']
        res = ttest_ind(newd1, newd2, permutations=5000)
        if res.pvalue<alpha:
            print("State "+str(j+1)+" is obviously different.")
        p.append(res.pvalue)
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
        f1 = list(fo1[:, j])
        f2 = list(fo2[:, j])
        newf1 = [x for x in f1 if x != 'nan']
        newf2 = [x for x in f2 if x != 'nan']
        res = ttest_ind(newf1, newf2, permutations=5000)
        if res.pvalue<alpha:
            print("State "+str(j+1)+" is obviously different.")
        p.append(res.pvalue)
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
        c1 = list(cluster1[:, j])
        c2 = list(cluster2[:, j])
        newc1 = [x for x in c1 if x != 'nan']
        newc2 = [x for x in c2 if x != 'nan']
        res = ttest_ind(newc1, newc2, permutations=5000)
        if res.pvalue<alpha:
            print("State "+str(j+1)+" is obviously different.")
        p.append(res.pvalue)
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
            m1 = list(ma1[:, m, n])
            m2 = list(ma2[:, m, n])
            newm1 = [x for x in m1 if x != 'nan']
            newm2 = [x for x in m2 if x != 'nan']
            res = ttest_ind(newm1, newm2, permutations=5000)
            if res.pvalue<alpha:
                print("Transition Probability from State "+str(m+1)+" to State "+str(n+1) +" is obviously different.")
            p[m][n] = res.pvalue
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

        print("MDD Group1 VS MDD Group2 starting...")
        Permutation_Test(K, mdd1, mdd2,'MDD1_VS_MDD2')
        print("MDD Group1 VS MDD Group2 finished.")

    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/DT/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/FO/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/Cluster/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD1_VS_HC/Markov/'+str(K))

        print("MDD Group1 VS HC Group starting...")
        Permutation_Test(K, mdd1, hc, 'MDD1_VS_HC')
        print("MDD Group1 VS HC Group finished.")
    
    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/DT/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/FO/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/Cluster/'+str(K))
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/MDD2_VS_HC/Markov/'+str(K))

        print("MDD Group2 VS HC Group starting...")
        Permutation_Test(K, mdd2, hc, 'MDD2_VS_HC')
        print("MDD Group2 VS HC Group finished.")


