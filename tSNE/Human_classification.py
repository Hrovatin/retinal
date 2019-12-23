#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import random
import scipy.io
from collections import Counter
from random import shuffle
from numpy import arange
from sklearn.preprocessing import scale

import importlib
import tsne_functions
importlib.reload(tsne_functions)
from tsne_functions import *


# In[2]:


import pylab
import matplotlib as mpl
import IPython
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 100
IPython.display.set_matplotlib_formats('png','pdf', quality=100)
pylab.rcParams['figure.figsize'] = (5.0, 4.0)


import sys

orig_stdout = sys.stdout
f = open('Human_clustering_out.txt', 'a')
sys.stdout = f




# In[3]:


data_path='/home/khrovatin/retinal/data/counts/'


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


# In[7]:


# Scale for logistic regression
data_norm=pd.DataFrame(scale(data_norm),index=data_norm.index,columns=data_norm.columns)
data_norm_h=pd.DataFrame(scale(data_norm_h),index=data_norm_h.index,columns=data_norm_h.columns)


# In[8]:


counts_h=scipy.io.mmread('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/matrix.mtx.gz')
counts_h=pd.DataFrame.sparse.from_spmatrix(counts_h).T
counts_h.columns=pd.read_table('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/features.tsv.gz',header=None)[1]
counts_h.index=pd.read_table('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/barcodes.tsv.gz',header=None)[0]
counts_h.index=[idx.split('-')[0] for idx in counts_h.index]


# In[9]:


genes_duplicated=[k for k,v in Counter(list(counts_h.columns)).items() if v > 1]
print('N duplicated genes',len(genes_duplicated))


# In[10]:


#genes_duplicated


# In[11]:


genes=pd.read_table('/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/00012_NeuNM_filtered_feature_bc_matrix/features.tsv.gz',header=None)


# In[12]:


#for gene in genes_duplicated:
#    print(genes[genes[1]==gene])
    #print(counts_h.iloc[:,counts_h.columns==gene].sum())


# Some of these ENS IDs point to deprecated genes/poorly characterized proteins. As their homology in macaque is uncertain based solely on gene symbol they are removed.

# In[13]:


print('Starting sizes of data_norm_h and data_norm:',data_norm_h.shape,data_norm.shape)

for gene in genes_duplicated:
    if gene in data_norm_h.columns:
          data_norm_h=data_norm_h.drop(gene,axis=1)
    if gene in data_norm.columns:
          data_norm=data_norm.drop(gene,axis=1)

print('End sizes of data_norm_h and data_norm:',data_norm_h.shape,data_norm.shape)


# In[14]:


print('Starting sizes of counts_h and data_int:',counts_h.shape,data_int.shape)
counts_h=counts_h.loc[data_norm_h.index,data_norm.columns]
data_int=data_int[data_norm.columns]
print('End sizes of counts_h and data_int:',counts_h.shape,data_int.shape)


# In[18]:


markers=pd.read_csv('/home/khrovatin/retinal/data/markers.csv')


# In[38]:


genes_retained=set(data_int.columns.values.copy())
genes_marker=set(markers['Marker'])
print('Retained genes:',len(genes_retained),'; marker genes:',len(genes_marker),'; intersection:',
      len(genes_marker & genes_retained))


# # Split data for classifier evaluation

# In[15]:


idx_shuffled=data_int.index.values.copy()
shuffle(idx_shuffled)
split1=int(len(idx_shuffled)*0.8)
split2=int(len(idx_shuffled)*0.9)
train=idx_shuffled[:split1]
validation=idx_shuffled[split1:split2]
test=idx_shuffled[split2:]
print('N train, validation, test:',len(train),len(validation),len(test))


# # tSNE embedding 

# ## Fit tSNE and KNN

# In[16]:


colour_dicts={'region':{'per':'#478216','fov':'#944368'},
             'cell_type':{'AC':'#bf7195','BC':'#b59a00','EpiImmune':'#75b05b',
                          'HC':'#4d4d4d','PR':'#7a0606','RGC':'#1c5191'}}


# In[17]:


data_tsne_train=data_int.loc[train,:]
data_tsne_test=data_int.loc[test,:]


# tSNE evaluation on per+fov

# In[ ]:

print('***** Eval tsne on per+fov data')
tsne_data_int_eval=analyse_tsne(data1=data_tsne_train,data2=data_tsne_test,col_data=col_data,colour_dicts=colour_dicts)


# tSNE on per+fov for embedding of human

# In[ ]:


tsne_int=make_tsne(data_int)
#savePickle(data_path+'tSNE_integrated_sharedGenes.pkl',(tsne_int,data_int.index))

# tSNE of fov, evaluation done with per

# In[ ]:

print('***** Eval tsne on fov data with per as test')
tsne_int_fov=make_tsne(data_int.loc[fov_cells,:])


# In[ ]:

# In[ ]:


tsne_data_int_fov=analyse_tsne(data1=data_int.loc[fov_cells,:],data2=data_int.loc[per_cells,:],col_data=col_data,
                               tsne1=tsne_int_fov,colour_dicts=colour_dicts)


# tSNE of per, evaluation done with fov

# In[ ]:

print('***** Eval tsne on per data with fov as test')
tsne_int_per=make_tsne(data_int.loc[per_cells,:])


# In[ ]:



tsne_data_int_per=analyse_tsne(data1=data_int.loc[per_cells,:],data2=counts_h,col_data=col_data,
                               tsne1=tsne_int_per,colour_dicts=colour_dicts)


# ## Human embeding on tSNE

# All macaque data

# In[ ]:


tsne_data_human=embed_tsne_new(data1=data_int,data2=counts_h,col_data1=col_data,tsne1=tsne_int,
                                colour_dicts=colour_dicts)


# On fov

# In[ ]:


tsne_data_human_fov=embed_tsne_new(data1=data_int.loc[fov_cells,:],data2=counts_h,col_data1=col_data,
                                   tsne1=tsne_int_fov, colour_dicts=colour_dicts)


# On per

# In[ ]:


tsne_data_human_per=embed_tsne_new(data1=data_int.loc[per_cells,:],data2=data_int.loc[fov_cells,:],col_data1=col_data,
                                   tsne1=tsne_int_per,colour_dicts=colour_dicts)


# # Softmax regression

# ## Fit models

# In[ ]:


data_lr_train=data_norm.loc[train,:]
data_lr_validation=data_norm.loc[validation,:]
data_lr_test=data_norm.loc[train,:]


# In[ ]:

print('******** Softmax eval on per+fov')
for c in [0.1]+[arange(0.2,1.1,0.2)]:
    print('\nRegularisation parameter C:',round(c,3))
    m=make_log_regression(data1=data_lr_train,data2=data_lr_validation,col_data=col_data, label='cell_type',
                        logreg={'penalty':'l1','C':c,'random_state':0,'solver':'saga','n_jobs':30})


# In[ ]:


#softmax_params={'penalty':'l1','C':0.2,'random_state':0,'solver':'saga','n_jobs':30}


# In[ ]:


#softmax_all=LogisticRegression(**softmax_params).fit(data_norm, col_data.loc[data_norm.index,'cell_type'])


# In[ ]:


#softmax_fov=make_log_regression(data1=data_norm.loc[fov_cells,:],data2=data_norm.loc[per_cells,:],
                                     col_data=col_data,label='cell_type',logreg=softmax_params)


# In[ ]:


#softmax_per=make_log_regression(data1=data_norm.loc[per_cells,:],data2=data_norm.loc[fov_cells,:],
                                     col_data=col_data,label='cell_type',logreg=softmax_params)


# # Embed human

# In[ ]:


#softmax_predicted=predict(classifier=softmax_all,data=data_norm_h)


# In[ ]:



sys.stdout = orig_stdout
f.close()
