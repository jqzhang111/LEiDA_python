# LEiDA(Cabral 2017. Sci Rep.)-PART1: get V1 across all subjects
# Author: zhangjiaqi(Smile.Z), CASIA, Brainnetome
import numpy as np 
from scipy.signal import hilbert, detrend
from scipy.spatial.distance import cosine
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

# get phase coherence matrix for all time points
# number of timepoints: 230

def dFC(bold):
    bold = bold.T 
    theta = np.zeros((bold.shape[0], bold.shape[1]))
    dFC = np.zeros((bold.shape[0], bold.shape[0], bold.shape[1]))
    for i in range(bold.shape[0]):
        bold[i] = detrend(bold[i])
        bold[i] = bold[i]-np.mean(bold[i])
        analytic_signal = hilbert(bold[i])
        instantaneous_phase = np.angle(analytic_signal)
        theta[i, :] = instantaneous_phase
    for j in range(bold.shape[1]):
        for m in range(bold.shape[0]):
            for n in range(bold.shape[0]):
                dFC[m][n][j] = math.cos(theta[m][j]-theta[n][j])
    return dFC


# get Leading Eigenvector
# dFC: N * N * T

def V1(dFC):
    V = np.zeros((dFC.shape[2], dFC.shape[0]))
    for i in range(dFC.shape[2]):
        eigval, eigvec = np.linalg.eig(dFC[:, :, i])
        indexes = np.argsort(-abs(eigval))[0:1]
        V[i, :] = eigvec[:, indexes].T
    return V


if __name__ == '__main__':
    path_mdd = '/share/home/zhangjiaqi/2022Project/HOPF/00_FMRI/xinxiang/MDD/'
    path_hc = '/share/home/zhangjiaqi/2022Project/HOPF/00_FMRI/xinxiang/HC/'
    mdd_file = os.listdir(path_mdd)
    hc_file = os.listdir(path_hc)
    
    for sub in mdd_file:
        print(sub[:7]+' starting...')
        bold = np.loadtxt(path_mdd+sub)
        dfc = dFC(bold)
        v1 = V1(dfc)
        # store dFC and V1 for per subject
        path1 = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/dFC/MDD/'+sub[:7]
        os.makedirs(path1)
        for j in range(dfc.shape[2]):
            np.savetxt(path1+'/'+str(j)+'.txt', dfc[:, :, j], delimiter=' ')
        path2 = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/MDD/'
        np.savetxt(path2+sub[:7]+'.txt', v1, delimiter=' ')
        print(sub[:7]+' finished.')
    
    for sub in hc_file:
        print(sub[:10]+' starting...')
        bold = np.loadtxt(path_hc+sub)
        dfc = dFC(bold)
        v1 = V1(dfc)
        # store dFC and V1 for per subject
        path1 = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/dFC/HC/'+sub[:10]
        os.makedirs(path1)
        for j in range(dfc.shape[2]):
            np.savetxt(path1+'/'+str(j)+'.txt', dfc[:, :, j], delimiter=' ')
        path2 = '/share/home/zhangjiaqi/2022Project/HOPF/02_LEiDA_Empircal/step1_get_dFC_V1/V1/HC/'
        np.savetxt(path2+sub[:10]+'.txt', v1, delimiter=' ')
        print(sub[:10]+' finished.')
