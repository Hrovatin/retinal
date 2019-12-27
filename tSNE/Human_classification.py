
# In[1]:
data_path='/home/khrovatin/retinal/data/counts/'
data_path_h='/home/khrovatin/retinal/data/human/'

import sys
sys.stdout = open(data_path_h+'Human_classification_out.txt', 'a')
sys.stderr = open(data_path_h+'Human_classification_err.txt', 'w')
print('********* NEW RUN *******')

import h5py
import random
import scipy.io
from collections import Counter
from random import shuffle
from numpy import arange
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import numpy as np
import glob

import importlib
import tsne_functions
importlib.reload(tsne_functions)
from tsne_functions import *



# ### Load data

col_data=pd.read_table(data_path+'passedQC_cellData.tsv',index_col=0)

with h5py.File(data_path+'scaledata.h5','r') as file:
    #As h5 file was saved in R it is imported as transposed
    data_int=pd.DataFrame(file.get('integrated/matrix')[:file.get('integrated/matrix').shape[0],:])
    data_int.columns=[name.decode() for name in file.get('integrated/rownames')]
    data_int.index=[name.decode() for name in file.get('integrated/colnames')[:file.get('integrated/colnames').shape[0]]]

with h5py.File(data_path+'normalisedlog.h5','r') as file:    
    data_norm=pd.DataFrame(file.get('ref/matrix')[:file.get('ref/matrix').shape[0],:])
    data_norm.columns=[name.decode() for name in file.get('ref/rownames')]
    data_norm.index=[name.decode() for name in file.get('ref/colnames')[:file.get('ref/colnames').shape[0]]]

retained_genes=loadPickle(data_path_h+'retained_genes.pkl')

#Filter macaque data
print('Shapes for data_int and data_norm before and after cell and gene filtering')
print(data_int.shape,data_norm.shape)
data_int=data_int[retained_genes]
data_norm=data_norm[retained_genes]
print(data_int.shape, data_norm.shape )


# Scale data 
data_norm=pd.DataFrame(scale(data_norm),index=data_norm.index,columns=data_norm.columns)
print('Shapes of scaled normalised data:',data_norm.shape)
# In[19]:

# Make classifiers on whole datasets

tsne_all=make_tsne(data_int)
savePickle(data_path_h+'tsne_all.pkl',(pd.DataFrame(np.array(tsne_all),index=data_int.index)))

knn_tsne_all = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne_all,col_data.loc[data_int.index,'cell_type'])
savePickle(data_path_h+'knn_tsne_all.pkl',knn_tsne_all)
#knn_tsne_all=loadPickle(data_path_h+'knn_tsne_all.pkl')

knn_expr_all = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    data_int, col_data.loc[data_int.index, 'cell_type'])
savePickle(data_path_h+'knn_expr_all.pkl',knn_expr_all)
#knn_expr_all = loadPickle(data_path_h+'knn_expr_all.pkl')

pca_all = PCA(n_components=30, random_state=0).fit(data_norm)
savePickle(data_path_h+'pca_all.pkl',pca_all)
knn_pca_all = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    pca_all.transform(data_norm), 
    col_data.loc[data_norm.index, 'cell_type'])
savePickle(data_path_h+'knn_pca_all.pkl',knn_pca_all)
#pca_all=loadPickle(data_path_h+'pca_all.pkl')
#knn_pca_all=loadPickle(data_path_h+'knn_pca_all.pkl')

c=0.01
softmax_all=LogisticRegression(**{'penalty':'l1','C':c,'random_state':0,'solver':'saga','n_jobs':30}).fit(
    data_norm, col_data.loc[data_norm.index,'cell_type'])
savePickle(data_path_h+'softmax_c'+str(c)+'_all.pkl',softmax_all)
#softmax_all=loadPickle(data_path_h+'softmax_c'+str(c)+'_all.pkl')

   
# HUMAN
file=h5py.File(data_path_h+'normalised_human.h5','r')
for path in glob.glob(
        '/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/*matrix'):
    # Load data
    sample_name = path.split('/')[-1]
    data_path_sample=data_path_h+sample_name+'_'
    print('****', sample_name, '****')
    data_h = pd.DataFrame(file.get(sample_name + '/matrix')[:file.get(sample_name + '/matrix').shape[0], :])
    data_h.columns = [name.decode() for name in file.get(sample_name + '/rownames')]
    data_h.index = [name.decode() for name in
                        file.get(sample_name + '/colnames')[:file.get(sample_name + '/colnames').shape[0]]]
    print('Shape of  normalised data before and after gene filtering and scaling')
    print(data_h.shape)
    data_h = data_h[retained_genes]
    data_h = pd.DataFrame(scale(data_h), index=data_h.index, columns=data_h.columns)
    print(data_h.shape)
    
    tsne_all_h = tsne_add(tsne1=tsne_all, data2=data_h)
    savePickle(data_path_sample + 'tsne_all_h.pkl', (pd.DataFrame(np.array(tsne_all_h), index=data_h.index)))
    
    knn_tsne_h = knn_tsne_all.predict(tsne_all_h)
    knn_expr_h = knn_expr_all.predict(data_h)
    knn_pca_h = knn_pca_all.predict(pca_all.transform(data_h))
    softmax_h = softmax_all.predict(data_h)
    
    classified = pd.DataFrame([knn_tsne_h, knn_expr_h, knn_pca_h, softmax_h],
                              index=['knn_tsne', 'knn_expr', 'knn_pca', 'softmax'], columns=data_h.index).T
    main_class = []
    n_main_class = []
    for row in classified.iterrows():
        row = row[1]
        class_counts = row.value_counts()
        count_main_class = class_counts.max()
        n_main_class.append(count_main_class)
        if count_main_class < row.shape[0] and class_counts.min() == count_main_class:
            main_class.append('NA')
        else:
            main_class.append(class_counts.idxmax())
    classified['main_class'] = main_class
    classified['N_main_class'] = n_main_class
    classified.to_csv(data_path_sample + 'classified.tsv', sep='\t')

    
file.close()
    
    
    
# OLD data loading
    #data_h=scipy.io.mmread(path+'/matrix.mtx.gz')
    #data_h=pd.DataFrame.sparse.from_spmatrix(data_h).T
    #data_h.columns=pd.read_table(path+'/features.tsv.gz',header=None)[1]
    #data_h.index=pd.read_table(path+'/barcodes.tsv.gz',header=None)[0]
    #data_h.index=[idx.split('-')[0] for idx in data_h.index]
    #print('N cells before filtering',data_h.shape[0])
    #data_h=data_h[(data_h.gt(0).sum(axis=1)>500).values]
    #print('N cells after filtering',data_h.shape[0])
    #data_h=normalise_counts(data_h)

