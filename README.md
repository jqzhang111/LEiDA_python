# LEiDA_python
LEiDA(Cabral 2017. Sci Rep.) python version

01_get_dFC_V1.py: calculate phase coherence matrix and leading eigenvectors across all subjects and time points. Please modify your time series file path before you use it.

02_get_best_k.py: use k-means and 'validclust'(https://validclust.readthedocs.io/en/latest/) to decide the best k. Please modify your time series file path and range of k before you use it.

02_get_emp_kmeans.py: use k from 02_get_best_k.py to cluster empircal data and visualize it. Also, Please modify your time series file path and range of k before you use it.
