# LEiDA(Cabral 2017. Sci Rep.)-PART6: fit for sFC
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


# p: weights for each V1
# V1: leading eigenvector
# sFC: static FC(average FC)
def Fit(p, V1, sFC):   
    FC = np.zeros((246, 246))
    for i in range(len(p)):
        FC += p[i] * np.dot(np.matrix(V1[i]).T, np.matrix(V1[i]))
    inds = np.triu_indices(246, 1)
    S = FC[inds]
    F = sFC[inds]
    return stats.pearsonr(S, F)[0], stats.pearsonr(S, F)[1]


if __name__ =='__main__':
    f = open('mdd1.txt', 'r')
    mdd1 = f.readlines()
    f.close()
    f = open('mdd2.txt', 'r')
    mdd2 = f.readlines()
    f.close()
    f = open('hc.txt', 'r')
    hc = f.readlines()
    f.close()

    for i in range(len(mdd1)):
        sFC = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/00_FMRI/xinxiang/FC/'+mdd1[i].replace('\n', '')+'.txt')
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+mdd1[i].replace('\n', ''))
        for K in range(2, 21):
            fit = []
            pvalue = []
            # use (mdd1+mdd2+hc) V1 to fit subject FC
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt'
            V1 = np.loadtxt(path)
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+mdd1[i].replace('\n', '')+'/'+str(K)+'/FO.txt'
            p = list(np.loadtxt(path))
            f, p = Fit(p, V1, sFC)
            fit.append(f)
            pvalue.append(p)

            # use subject V1 to fit subject FC
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+mdd1[i].replace('\n', '')+'/'+str(K)+'/centroids_'+str(K)+'_cluster.txt'
            V1 = np.loadtxt(path)
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+mdd1[i].replace('\n', '')+'/'+str(K)+'/centroids_'+str(K)+'_count.txt'
            count = np.loadtxt(path)
            p = []
            for j in range(K):
                p.append(count[j]/np.sum(count))
            f, p = Fit(p, V1, sFC)
            fit.append(f)
            pvalue.append(p)

            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+mdd1[i].replace('\n', '')+'/'+str(K)+'_cluster_fit.txt', fit, delimiter=' ')
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+mdd1[i].replace('\n', '')+'/'+str(K)+'_cluster_p.txt', pvalue, delimiter=' ')


    for i in range(len(mdd2)):
        sFC = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/00_FMRI/xinxiang/FC/'+mdd2[i].replace('\n', '')+'.txt')
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+mdd2[i].replace('\n', ''))
        for K in range(2, 21):
            fit = []
            pvalue = []
            # use (mdd1+mdd2+hc) V1 to fit subject FC
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt'
            V1 = np.loadtxt(path)
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+mdd2[i].replace('\n', '')+'/'+str(K)+'/FO.txt'
            p = list(np.loadtxt(path))
            f, p = Fit(p, V1, sFC)
            fit.append(f)
            pvalue.append(p)

            # use subject V1 to fit subject FC
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+mdd2[i].replace('\n', '')+'/'+str(K)+'/centroids_'+str(K)+'_cluster.txt'
            V1 = np.loadtxt(path)
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+mdd2[i].replace('\n', '')+'/'+str(K)+'/centroids_'+str(K)+'_count.txt'
            count = np.loadtxt(path)
            p = []
            for j in range(K):
                p.append(count[j]/np.sum(count))
            f, p = Fit(p, V1, sFC)
            fit.append(f)
            pvalue.append(p)

            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+mdd2[i].replace('\n', '')+'/'+str(K)+'_cluster_fit.txt', fit, delimiter=' ')
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+mdd2[i].replace('\n', '')+'/'+str(K)+'_cluster_p.txt', pvalue, delimiter=' ')


    
    for i in range(len(hc)):
        sFC = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/00_FMRI/xinxiang/FC/'+hc[i].replace('\n', '')+'.txt')
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+hc[i].replace('\n', ''))
        for K in range(2, 21):
            fit = []
            pvalue = []
            # use (mdd1+mdd2+hc) V1 to fit subject FC
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step2_emp_kmeans/'+str(K)+'/centroids_'+str(K)+'_cluster.txt'
            V1 = np.loadtxt(path)
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/subject/'+hc[i].replace('\n', '')+'/'+str(K)+'/FO.txt'
            p = list(np.loadtxt(path))
            f, p = Fit(p, V1, sFC)
            fit.append(f)
            pvalue.append(p)

            # use subject V1 to fit subject FC
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+hc[i].replace('\n', '')+'/'+str(K)+'/centroids_'+str(K)+'_cluster.txt'
            V1 = np.loadtxt(path)
            path = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/V1/subject/'+hc[i].replace('\n', '')+'/'+str(K)+'/centroids_'+str(K)+'_count.txt'
            count = np.loadtxt(path)
            p = []
            for j in range(K):
                p.append(count[j]/np.sum(count))
            f, p = Fit(p, V1, sFC)
            fit.append(f)
            pvalue.append(p)

            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+hc[i].replace('\n', '')+'/'+str(K)+'_cluster_fit.txt', fit, delimiter=' ')
            np.savetxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step6_individual_analysis/Fit/'+hc[i].replace('\n', '')+'/'+str(K)+'_cluster_p.txt', pvalue, delimiter=' ')


    
   

    








   

