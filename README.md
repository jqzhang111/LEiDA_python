# LEiDA_python

LEiDA(Cabral 2017. Sci Rep.) python version 

Any questions please feel free to contact me: zhangjiaqi2021@ia.ac.cn, smile.zhang123@gmail.com

* Please modify your file path before you use it.


01_get_dFC_V1.py: calculate phase coherence matrix and leading eigenvectors across all subjects and time points. 

02_get_emp_kmeans.py: use k-means and 'validclust'(https://validclust.readthedocs.io/en/latest/) to decide the best k. Run k from 2-20 and get the corresponding centroids for each brain states.

03_index.py: calculate index for each brain state including fractional occupancy, dwell time, markov chain trainsition matrix, community for each cluster(k from 2 to 20), correlation with yeo7/yeo17(using 'dice_correlation.py' in https://github.com/neurodata/neuroparc/tree/master/scripts)

04_permutation_test.py: assess differences on measures using a permutation-based t test(5000 permutations and alpha = 0.05 for a standard threshold)

05_visualize.py: visualization for 03, 04 result.

06_individual_v1.py: run LEiDA on each subjects and get personal brain states.

06_fit.py: use personal brain states to fit static FC and compare with the original brain states(across all subjects and timepoints).
