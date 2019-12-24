#!/usr/bin/env python
# coding: utf-8

# In[1]:
data_path='/home/khrovatin/retinal/data/counts/'
data_path_h='/home/khrovatin/retinal/data/human/'

import sys
sys.stdout = open(data_path_h+'Human_classification_out.txt', 'a')
sys.stderr = open(data_path_h+'Human_classification_err.txt', 'a')
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



# In[4]:


col_data=pd.read_table(data_path+'passedQC_cellData.tsv',index_col=0)
col_by_region=col_data.groupby('region')
fov_cells=col_by_region.get_group('fov').index
per_cells=col_by_region.get_group('per').index


# In[5]:


with h5py.File(data_path+'scaledata.h5','r') as file:
    #As h5 file was saved in R it is imported as transposed
    data_int=pd.DataFrame(file.get('integrated/matrix')[:file.get('integrated/matrix').shape[0],:])
    data_int.columns=[name.decode() for name in file.get('integrated/rownames')]
    data_int.index=[name.decode() for name in file.get('integrated/colnames')[:file.get('integrated/colnames').shape[0]]]


# In[6]:


with h5py.File(data_path+'normalisedlog.h5','r') as file:    
    data_norm=pd.DataFrame(file.get('ref/matrix')[:file.get('ref/matrix').shape[0],:])
    data_norm.columns=[name.decode() for name in file.get('ref/rownames')]
    data_norm.index=[name.decode() for name in file.get('ref/colnames')[:file.get('ref/colnames').shape[0]]]
    
    data_norm_h=pd.DataFrame(file.get('00012_NeuNM/matrix')[:file.get('00012_NeuNM/matrix').shape[0],:])
    data_norm_h.columns=[name.decode() for name in file.get('00012_NeuNM/rownames')]
    data_norm_h.index=[name.decode() for name in file.get('00012_NeuNM/colnames')[:file.get('00012_NeuNM/colnames').shape[0]]]


# In[7]:24322

# Not used!
# Scale for logistic regression
#data_norm=pd.DataFrame(scale(data_norm),index=data_norm.index,columns=data_norm.columns)
#data_norm_h=pd.DataFrame(scale(data_norm_h),index=data_norm_h.index,columns=data_norm_h.columns)


# In[8]:24322


counts_h=scipy.io.mmread('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/matrix.mtx.gz')
counts_h=pd.DataFrame.sparse.from_spmatrix(counts_h).T
counts_h.columns=pd.read_table('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/features.tsv.gz',header=None)[1]
counts_h.index=pd.read_table('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/barcodes.tsv.gz',header=None)[0]
counts_h.index=[idx.split('-')[0] for idx in counts_h.index]


counts=scipy.io.mmread(data_path+'passedQC.mtx')
counts=pd.DataFrame.sparse.from_spmatrix(counts).T
counts.columns=pd.read_table(data_path+'passedQC_genes.tsv')['rownames']
counts.index=col_data.index


# ### Duplicated gene symbols in human

# Some gene symbols are duplicated in the human data (see count below). 

# In[9]:


genes_duplicated=[k for k,v in Counter(list(counts_h.columns)).items() if v > 1]
print('Duplicated gene symbols among all human genes', len(genes_duplicated))


# Remove low quality cells and unvariable or duplicated genes
print('Shapes for data_norm_h and data_norm before and after cell and gene filtering')
print(data_norm_h.shape,data_norm.shape)

for gene in genes_duplicated:
    if gene in data_norm_h.columns:
          data_norm_h=data_norm_h.drop(gene,axis=1)
    if gene in data_norm.columns:
          data_norm=data_norm.drop(gene,axis=1)

print(data_norm_h.shape,data_norm.shape)


# In[14]:

print('Shapes for counts_h,data_int and counts before and after cell and gene filtering')
print(counts_h.shape,data_int.shape,counts.shape)
counts_h=counts_h.loc[data_norm_h.index,data_norm.columns]
data_int=data_int[data_norm.columns]
counts=counts[data_norm.columns]
print(counts_h.shape,data_int.shape,counts.shape)


# In[15]:


markers=pd.read_csv('/home/khrovatin/retinal/data/markers.csv')


# In[16]:


genes_retained=set(data_int.columns.values.copy())
genes_marker=set(markers['Marker'])
print('Retained genes:',len(genes_retained),'; marker genes:',len(genes_marker),'; intersection:',
      len(genes_marker & genes_retained))




colour_dicts={'region':{'per':'#478216','fov':'#944368'},
             'cell_type':{'AC':'#bf7195','BC':'#b59a00','EpiImmune':'#75b05b',
                          'HC':'#a1651b','PR':'#7a0606','RGC':'#1c5191','added':'#4d4d4d'}}


# In[19]:

tsne_int=make_tsne(data_int)
tsne_fov=make_tsne(counts.loc[fov_cells,:])
tsne_per=make_tsne(counts.loc[per_cells,:])

# Plots of individual macaque datasets
plot_tsne(tsnes=[tsne_int], classes=[dict(zip(col_data.index,col_data['region']))],
          names=[data_int.index], legend=True,colour_dict=colour_dicts['region'])
plt.savefig(data_path_h+'tSNE_integrated_region.pdf')

plot_tsne(tsnes=[tsne_int], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[data_int.index], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_integrated_cellType.pdf')

plot_tsne(tsnes=[tsne_fov], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[fov_cells], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_fov_cellType.pdf')

plot_tsne(tsnes=[tsne_per], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[per_cells], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_per_cellType.pdf')

plot_tsne(tsnes=[tsne_int], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[data_int.index], legend=False)
plt.savefig(data_path_h+'tSNE_integrated_cellTypesFine.pdf')

plot_tsne(tsnes=[tsne_fov], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[fov_cells], legend=False)
plt.savefig(data_path_h+'tSNE_fov_cellTypesFine.pdf')

plot_tsne(tsnes=[tsne_per], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[per_cells], legend=False)
plt.savefig(data_path_h+'tSNE_per_cellTypesFine.pdf')


# tSNE: Add per of fov or fov on per
tsne_fov_aPer=tsne_add(tsne1=tsne_fov,data1=counts.loc[fov_cells,:],data2=counts.loc[per_cells,:])
tsne_per_aFov=tsne_add(tsne1=tsne_per,data1=counts.loc[per_cells,:],data2=counts.loc[fov_cells,:])

plot_tsne(tsnes=[tsne_fov,tsne_fov_aPer], 
          classes=[dict(zip(col_data.index,col_data['cell_type'])),
                   dict(zip(per_cells,['added']*len(per_cells)))],
          names=[fov_cells,per_cells], 
          legend=True,colour_dict=colour_dicts['cell_type'],
          plotting_params = [{'s'=0.1,'alpha'=0.1},{'s'=1,'alpha'=0.2}])
plt.savefig(data_path_h+'tSNE_fov_addedPer.pdf')

plot_tsne(tsnes=[tsne_per,tsne_per_aFov], 
          classes=[dict(zip(col_data.index,col_data['cell_type'])),
                   dict(zip(fov_cells,['added']*len(fov_cells)))],
          names=[per_cells,fov_cells], 
          legend=True,colour_dict=colour_dicts['cell_type'],
         plotting_params = [{'s'=0.1,'alpha'=0.1},{'s'=1,'alpha'=0.2}])
plt.savefig(data_path_h+'tSNE_per_addedFov.pdf')

# Classification KNN (test one region, train another)
# On tSNE
print('Classifier: KNN counts tSNE, train fov, test per')
knn_tsne_fov = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne_fov,col_data.loc[fov_cells,'cell_type'])
evaluate_classifier(classifier=knn_tsne_fov,data=pd.DataFrame(tsne_fov_aPer,index=per_cells),col_data=col_data,label='cell_type')

print('Classifier: KNN counts tSNE, train per, test fov')
knn_tsne_per = KNeighborsClassifier(weights='distance',n_jobs=4).fit(tsne_per,col_data.loc[per_cells,'cell_type'])
evaluate_classifier(classifier=knn_tsne_per,data=pd.DataFrame(tsne_per_aFov,index=fov_cells),col_data=col_data,label='cell_type')

# On mean-scaled log CPM normalised data
# Columns are not matched explicitely as both per and fov come from data_norm
print('Classifier: KNN mean-scaled log(CPM), train fov, test per')
knn_lcpm_fov = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    scale(data_norm.loc[fov_cells,:],with_std=False), col_data.loc[fov_cells, 'cell_type'])
evaluate_classifier(classifier=knn_lcpm_fov,
                    data=pd.DataFrame(scale(data_norm.loc[per_cells,:],with_std=False),index=per_cells),
                    col_data=col_data, label='cell_type')

print('Classifier: KNN mean-scaled log(CPM), train per, test fov')
knn_lcpm_per = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    scale(data_norm.loc[per_cells,:],with_std=False), col_data.loc[per_cells, 'cell_type'])
evaluate_classifier(classifier=knn_lcpm_per,
                    data=pd.DataFrame(scale(data_norm.loc[fov_cells,:],with_std=False),index=fov_cells),
                    col_data=col_data, label='cell_type')

# TODO
# On mean-scaled (performed by PCA function) log CPM normalised data
pca_fov = PCA(n_components=30, random_state=0).fit(data_norm.loc[fov_cells,:])
pca_per = PCA(n_components=30, random_state=0).fit(data_norm.loc[per_cells,:])

print('Classifier: KNN mean-scaled log(CPM) PCA, train fov, test per')
knn_pca_fov = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    pca_fov.transform(data_norm.loc[fov_cells,:]), 
    col_data.loc[fov_cells, 'cell_type'])
evaluate_classifier(classifier=knn_pca_fov,
                    data=pd.DataFrame(pca_fov.transform(data_norm.loc[per_cells,:]),index=per_cells),
                    col_data=col_data, label='cell_type')

print('Classifier: KNN mean-scaled log(CPM) PCA, train per, test fov')
knn_pca_fov = KNeighborsClassifier(metric='cosine', weights='distance', n_jobs=20).fit(
    pca_fov.transform(data_norm.loc[fov_cells,:]), 
    col_data.loc[fov_cells, 'cell_type'])
evaluate_classifier(classifier=knn_pca_fov,
                    data=pd.DataFrame(pca_fov.transform(data_norm.loc[per_cells,:]),index=per_cells),
                    col_data=col_data, label='cell_type')
























































