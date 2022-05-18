# LEiDA(Cabral 2017. Sci Rep.)-PART5: Visualization of result
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

def Func_Network(K):
    yeo7 = ['Visual', 'Somatomotor', 'Dorsal', 'Ventral', 'Limbic', 'Frontoparietal', 'Default']
    yeo17 = ['Visual_A', 'Visual_B', 'Somatomotor_A', 'Somatomotor_B', 'Temporal Parietal',
    'Frontoparietal', 'Dorsal Attention_B', 'Salience+Ventral Attention_A', 'Salience+Ventral Attention_B', 
    'Control_A', 'Control_B', 'Control_C', 'Default_A', 'Default_B', 'Default_C',
    'Limbic_A', 'Limbic_B']

    corr7 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(K)+'/yeo7corr.txt')
    corr17 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(K)+'/yeo17corr.txt')
    
    p7 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(K)+'/yeo7pvalue.txt')
    p17 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/cluster/'+str(K)+'/yeo17pvalue.txt')

    x = list(range(1, K+1, 1))
    xx = list(range(0, K+1, 1))
    plt.figure(figsize=(25, 18))
    plt.xticks(np.arange(len(xx)), xx)
    plt.axhline(y=0)
    wid = 1.0/8
    for i in range(7):
        plt.bar(x, corr7[:, i], width=wid, label=yeo7[i])
        for j in range(K):
            if p7[:, i][j]<0.001 and corr7[:, i][j]>0:
                plt.text(x[j], corr7[:, i][j], "*", horizontalalignment='center', verticalalignment= 'bottom')
            if p7[:, i][j]<0.001 and corr7[:, i][j]<0:
                plt.text(x[j], corr7[:, i][j], "*", horizontalalignment='center', verticalalignment= 'top')
        x = [a+wid for a in x]
    plt.xlabel('Brain States obtained with LEiDA for k='+str(K))
    plt.ylabel("Pearson's correlation with Yeo7")
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=8)
    plt.savefig('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/func_network/'+str(K)+'/yeo7_cluster.png')
    plt.clf()

    x = list(range(1, K+1, 1))
    plt.figure(figsize=(25, 5))
    plt.xticks(np.arange(len(xx)), xx)
    plt.axhline(y=0)
    wid = 1.0/18
    for i in range(17):
        plt.bar(x, corr17[:, i], width=wid, label=yeo17[i])
        for j in range(K):
            if p17[:, i][j]<0.001 and corr17[:, i][j]>0:
                plt.text(x[j], corr17[:, i][j], "*", horizontalalignment='center', verticalalignment= 'bottom')
            if p17[:, i][j]<0.001 and corr17[:, i][j]<0:
                plt.text(x[j], corr17[:, i][j], "*", horizontalalignment='center', verticalalignment= 'top')
        x = [a+wid for a in x]
    plt.xlabel('Brain States obtained with LEiDA for k='+str(K))
    plt.ylabel("Pearson's correlation with Yeo17")
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=8)
    plt.savefig('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/func_network/'+str(K)+'/yeo17_cluster.png')
    plt.clf()
  

def DT(K, g1, g2): 
    x = list(range(1, K+1, 1))
    xx = list(range(0, K+1, 1))
    plt.figure(figsize=(25, 5))
    plt.xticks(np.arange(len(xx)), xx)
    plt.axhline(y=0)
    wid = 1.0/3
    
    pvalue = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+g1+'_VS_'+g2+'/DT/'+str(K)+'/Dwell_Time_Test.txt')
    y1 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/'+g1+'/'+str(K)+'/DT.txt')
    y2 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/'+g2+'/'+str(K)+'/DT.txt')
    
    y1 = list(y1)
    y2 = list(y2)
    plt.bar(x, y1, width=wid, label=g1)
    x1 = [a+wid for a in x]
    plt.bar(x1, y2, width=wid, label=g2)
    for i in range(K):
        if pvalue[i]<0.05:
            plt.text((x[i]+x1[i])/2, max(y1[i], y2[i]), "*", horizontalalignment='center', verticalalignment= 'bottom')
    
    plt.xlabel('Brain States obtained with LEiDA for k='+str(K))
    plt.ylabel("Dwell Time(s)")
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=8)
    plt.savefig('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/DT/'+str(K)+'/'+g1+'_VS_'+g2+'_DT.png')
    plt.clf()

    
def FO(K, g1, g2):
    x = list(range(1, K+1, 1))
    xx = list(range(0, K+1, 1))
    plt.figure(figsize=(25, 5))
    plt.xticks(np.arange(len(xx)), xx)
    plt.axhline(y=0)
    wid = 1.0/3
    
    pvalue = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+g1+'_VS_'+g2+'/FO/'+str(K)+'/Fractional_Occupancy_Test.txt')
    y1 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/'+g1+'/'+str(K)+'/FO.txt')
    y2 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/'+g2+'/'+str(K)+'/FO.txt')
    
    y1 = list(y1)
    y2 = list(y2)

    plt.bar(x, y1, width=wid, label=g1)
    x1 = [a+wid for a in x]
    plt.bar(x1, y2, width=wid, label=g2)
    for i in range(K):
        if pvalue[i]<0.05:
            plt.text((x[i]+x1[i])/2, max(y1[i], y2[i]), "*", horizontalalignment='center', verticalalignment= 'bottom')
    
    plt.xlabel('Brain States obtained with LEiDA for k='+str(K))
    plt.ylabel("Fractional Occupancy")
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=8)
    plt.savefig('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/FO/'+str(K)+'/'+g1+'_VS_'+g2+'_FO.png')
    plt.clf()



def Markov(K, g1, g2):
    pvalue = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step4_permutaion_test/'+g1+'_VS_'+g2+'/Markov/'+str(K)+'/Transition_Probability_Test.txt')
    m1 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/'+g1+'/'+str(K)+'/Markov_Matrix.txt')
    m2 = np.loadtxt('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step3_index/'+g2+'/'+str(K)+'/Markov_Matrix.txt')

    annot1 = []
    annot2 = []
    for i in range(K):
        annot11 = []
        annot22 = []
        for j in range(K):
            if pvalue[i][j]<0.05:
                annot11.append(str(np.round(m1[i][j], 3))+'*')
                annot22.append(str(np.round(m2[i][j], 3))+'*')
            else:
                annot11.append(np.round(m1[i][j], 3))
                annot22.append(np.round(m2[i][j], 3))
        annot1.append(annot11)
        annot2.append(annot22)
        
    m1 = list(m1)
    m2 = list(m2)
    x_axis_labels = list(range(1, K+1, 1))
    y_axis_labels = list(range(1, K+1, 1))
    if K<7:
      plt.figure(figsize=(10, 10))
    else:
      plt.figure(figsize=(20, 20))
    sns.heatmap(m1, annot=annot1, vmax=1, vmin=0, fmt='', square=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.savefig('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/Markov/'+str(K)+'/'+g1+'_VS_'+g2+'_Markov_'+g1+'_.png')
    plt.clf() 
    sns.heatmap(m2, annot=annot2, vmax=1, vmin=0, fmt='', square=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    plt.savefig('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/Markov/'+str(K)+'/'+g1+'_VS_'+g2+'_Markov_'+g2+'_.png')
    plt.clf()



if __name__ == '__main__':
    for K in range(2, 21):
        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/func_network/'+str(K))
        Func_Network(K)

        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/DT/'+str(K))
        DT(K, 'MDD1','MDD2')
        DT(K, 'MDD1', 'HC')
        DT(K, 'MDD2', 'HC')

        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/FO/'+str(K))
        FO(K, 'MDD1','MDD2')
        FO(K, 'MDD1', 'HC')
        FO(K, 'MDD2', 'HC')

        os.makedirs('/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step5_visualize/Markov/'+str(K))
        Markov(K, 'MDD1', 'MDD2')
        Markov(K, 'MDD1', 'HC')
        Markov(K, 'MDD2', 'HC')
