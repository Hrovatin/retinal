#!/usr/bin/env python
# coding: utf-8

# In[1]:
data_path='/home/khrovatin/retinal/data/counts/'
data_path_h='/home/khrovatin/retinal/data/human/'

import sys
sys.stdout = open(data_path_h+'Human_classification_test_out.txt', 'w')
sys.stderr = open(data_path_h+'Human_classification_test_err.txt', 'w')
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

import importlib
import tsne_functions
importlib.reload(tsne_functions)
from tsne_functions import *


# In[2]:


#import pylab
import matplotlib as mpl
#import IPython
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 100
#IPython.display.set_matplotlib_formats('png','pdf', quality=100)
#pylab.rcParams['figure.figsize'] = (5.0, 4.0)


# ### Load data


col_data=pd.read_table(data_path+'passedQC_cellData.tsv',index_col=0)
col_by_region=col_data.groupby('region')
fov_cells=col_by_region.get_group('fov').index
per_cells=col_by_region.get_group('per').index


with h5py.File(data_path+'scaledata.h5','r') as file:
    #As h5 file was saved in R it is imported as transposed
    data_int=pd.DataFrame(file.get('integrated/matrix')[:file.get('integrated/matrix').shape[0],:])
    data_int.columns=[name.decode() for name in file.get('integrated/rownames')]
    data_int.index=[name.decode() for name in file.get('integrated/colnames')[:file.get('integrated/colnames').shape[0]]]

with h5py.File(data_path+'normalisedlog.h5','r') as file:    
    data_norm=pd.DataFrame(file.get('ref/matrix')[:file.get('ref/matrix').shape[0],:])
    data_norm.columns=[name.decode() for name in file.get('ref/rownames')]
    data_norm.index=[name.decode() for name in file.get('ref/colnames')[:file.get('ref/colnames').shape[0]]]
    
#counts=scipy.io.mmread(data_path+'passedQC.mtx')
#counts=pd.DataFrame.sparse.from_spmatrix(counts).T
#counts.columns=pd.read_table(data_path+'passedQC_genes.tsv')['rownames']
#counts.index=col_data.index

#Perform  ln transform and CPM calculation
#data_norm=normalise_counts(counts)


#GENE FILTERING
# Duplicated genes
genes_human=pd.read_table('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/features.tsv.gz',header=None)[1]

genes_duplicated_human=[k for k,v in Counter(genes_human).items() if v > 1]
print('Duplicated gene symbols among all human genes', len(genes_duplicated_human))

variable_genes=set(data_int.columns)
retained_genes=variable_genes&set(genes_human)
retained_genes=retained_genes-set(genes_duplicated_human)
print('N retained genes:',len(retained_genes))
retained_genes=[gene for gene in retained_genes]
retained_genes.sort()
savePickle(data_path_h+'retained_genes.pkl',retained_genes)

#Filter macaque data
print('Shapes for data_int and data_norm before and after cell and gene filtering')
print(data_int.shape,data_norm.shape)
data_int=data_int[retained_genes]
data_norm=data_norm[retained_genes]
print(data_int.shape,data_norm.shape)


# Known markers
markers=pd.read_csv('/home/khrovatin/retinal/data/markers.csv')

genes_marker=set(markers['Marker'])
print('Retained genes:',len(retained_genes),'; marker genes:',len(genes_marker),'; intersection:',
      len(genes_marker & set(retained_genes)))

colour_dicts={'region':{'per':'#478216','fov':'#944368'},
             'cell_type':{'AC':'#bf7195','BC':'#b59a00','EpiImmune':'#75b05b',
                          'HC':'#a1651b','PR':'#7a0606','RGC':'#1c5191','added':'#4d4d4d'}}
# Scale data and split it
data_norm_fov=pd.DataFrame(scale(data_norm.loc[fov_cells,:]),index=fov_cells,columns=data_norm.columns)
data_norm_per=pd.DataFrame(scale(data_norm.loc[per_cells,:]),index=per_cells,columns=data_norm.columns)
data_norm=pd.DataFrame(scale(data_norm),index=data_norm.index,columns=data_norm.columns)
print('Shapes of scaled normalised data for all, fov and per:',data_norm.shape,data_norm_fov.shape,data_norm_per.shape)
# In[19]:

#tsne_fov=make_tsne(data_int.loc[fov_cells,:])
#savePickle(data_path_h+'tsne_fov.pkl',(pd.DataFrame(np.array(tsne_fov),index=fov_cells)))

tsne_per=make_tsne(data_int.loc[per_cells,:])
savePickle(data_path_h+'tsne_per.pkl',(pd.DataFrame(np.array(tsne_per),index=per_cells)))


# tSNE: Add per of fov or fov on per
#tsne_fov_aPer=tsne_add(tsne1=tsne_fov,data2=data_norm_per)
#savePickle(data_path_h+'tsne_fov_aPer.pkl',(pd.DataFrame(np.array(tsne_fov_aPer),index=per_cells)))

tsne_per_aFov=tsne_add(tsne1=tsne_per,data2=data_norm_fov)
savePickle(data_path_h+'tsne_per_aFov.pkl',(pd.DataFrame(np.array(tsne_per_aFov),index=fov_cells)))

# Classification KNN (test one region, train another)
# On tSNE
#print('Classifier: KNN scaled(log(CPM)) tSNE, train fov, test per')
#knn_tsne_fov = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne_fov,col_data.loc[fov_cells,'cell_type'])
#predicted=evaluate_classifier(classifier=knn_tsne_fov,data=pd.DataFrame(tsne_fov_aPer,index=per_cells),col_data=col_data,label='cell_type')
#savePickle(data_path_h+'knn_tsne_fov_aPer.pkl',(knn_tsne_fov,predicted))

print('Classifier: KNN scaled(log(CPM)) tSNE, train per, test fov')
knn_tsne_per = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne_per,col_data.loc[per_cells,'cell_type'])
predicted=evaluate_classifier(classifier=knn_tsne_per,data=pd.DataFrame(tsne_per_aFov,index=fov_cells),col_data=col_data,label='cell_type')
savePickle(data_path_h+'knn_tsne_per_aFov.pkl',(knn_tsne_per,predicted))

# On scaled log CPM normalised data
# Columns are not matched explicitely as both per and fov come from data_norm
print('Classifier: KNN scaled log(CPM), train fov, test per')
knn_expr_fov = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    data_int.loc[fov_cells,:], col_data.loc[fov_cells, 'cell_type'])
predicted=evaluate_classifier(classifier=knn_expr_fov,
                    data=data_norm_per,
                    col_data=col_data, label='cell_type')
savePickle(data_path_h+'knn_expr_fov_aPer.pkl',(knn_expr_fov,predicted))

print('Classifier: KNN scaled log(CPM), train per, test fov')
knn_expr_per = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    data_int.loc[per_cells,:], col_data.loc[per_cells, 'cell_type'])
predicted=evaluate_classifier(classifier=knn_expr_per,
                    data=data_norm_fov,
                    col_data=col_data, label='cell_type')
savePickle(data_path_h+'knn_expr_per_aFov.pkl',(knn_expr_per,predicted))

# On scaled (performed by PCA function) log CPM normalised data
pca_fov = PCA(n_components=30, random_state=0).fit(data_norm_fov)
savePickle(data_path_h+'pca_fov.pkl',pca_fov)

pca_per = PCA(n_components=30, random_state=0).fit(data_norm_per)
savePickle(data_path_h+'pca_per.pkl',pca_per)

print('Classifier: KNN scaled log(CPM) PCA, train fov, test per')
knn_pca_fov = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    pca_fov.transform(data_norm_fov), 
    col_data.loc[fov_cells, 'cell_type'])
predicted=evaluate_classifier(classifier=knn_pca_fov,
                    data=pd.DataFrame(pca_fov.transform(data_norm_per),index=per_cells),
                    col_data=col_data, label='cell_type')
savePickle(data_path_h+'knn_pca_fov_aPer.pkl',(knn_pca_fov,predicted))

print('Classifier: KNN scaled log(CPM) PCA, train per, test fov')
knn_pca_per = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    pca_per.transform(data_norm_per), 
    col_data.loc[per_cells, 'cell_type'])
predicted=evaluate_classifier(classifier=knn_pca_per,
                    data=pd.DataFrame(pca_per.transform(data_norm_fov),index=fov_cells),
                    col_data=col_data, label='cell_type')
savePickle(data_path_h+'knn_pca_per_aFov.pkl',(knn_pca_per,predicted))

# SOFTMAX
for c in [0.001,0.01,0.1,1]:
    print('Classifier: softmax c =',c,' scaled log(CPM), train fov, test per')
    m=make_log_regression(data1=data_norm_fov, data2=data_norm_per,
                          col_data=col_data,label='cell_type',
                        logreg={'penalty':'l1','C':c,'random_state':0,'solver':'saga','n_jobs':40})
    savePickle(data_path_h+'softmax_c'+str(c)+'_fov_aPer.pkl',m)   
    

for c in [0.001,0.01,0.1,1]:
    print('Classifier: softmax c =',c,'scaled log(CPM), train per, test fov')
    m=make_log_regression(data1=data_norm_per, data2=data_norm_fov,
                          col_data=col_data,label='cell_type',
                        logreg={'penalty':'l1','C':c,'random_state':0,'solver':'saga','n_jobs':40})
    savePickle(data_path_h+'softmax_c'+str(c)+'_per_aFov.pkl',m) 
    
 














































