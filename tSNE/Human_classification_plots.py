data_path='/home/khrovatin/retinal/data/counts/'
data_path_h='/home/khrovatin/retinal/data/human/'

#import sys
#sys.stdout = open(data_path_h+'Human_classificatio_plot_out.txt', 'w')
#sys.stderr = open(data_path_h+'Human_classification_plot_err.txt', 'w')
#print('********* NEW RUN *******')

import glob
import pandas as pd

import importlib
import tsne_functions
importlib.reload(tsne_functions)
from tsne_functions import *


import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 10



colour_dicts={'region':{'per':'#478216','fov':'#944368'},
             'cell_type':{'reference':'#4d4d4d','AC':'#bf7195','BC':'#b59a00','EpiImmune':'#75b05b',
                          'HC':'#a1651b','PR':'#7a0606','RGC':'#1c5191','NA':'#9500ff',
                          'human':'#4d4d4d','test':'#4d4d4d','training':'#4d4d4d','fov':'#4d4d4d','per':'#4d4d4d'
                          },
             'classified':{'reference':'#4d4d4d',2:'#547ea8',3:'#5e8740',4:'#9da32f',
                          'unclassified':'#9500ff'}}
#Load data

# tSNE per region

# Plots of individual macaque datasets
tsne_all=loadPickle(data_path_h+'tsne_all.pkl')
tsne_fov=loadPickle(data_path_h+'tsne_fov.pkl')
tsne_fov_aPer=loadPickle(data_path_h+'tsne_fov_aPer.pkl')
tsne_per=loadPickle(data_path_h+'tsne_per.pkl')
tsne_per_aFov=loadPickle(data_path_h+'tsne_per_aFov.pkl')

tsne_all_nonint=loadPickle(data_path_h+'tsne_all_nonintegrated.pkl')
tsne_fov_nonint=loadPickle(data_path_h+'tsne_fov_nonintegrated.pkl')
tsne_per_nonint=loadPickle(data_path_h+'tsne_per_nonintegrated.pkl')

col_data=pd.read_table(data_path+'passedQC_cellData.tsv',index_col=0)

# Integrated tSNE
plot_tsne(tsnes=[tsne_all], classes=[dict(zip(col_data.index,col_data['region']))],
          names=[tsne_all.index], legend=True,colour_dict=colour_dicts['region'],
          title='Regions in integrated data set')
plt.savefig(data_path_h+'tSNE_all_region.pdf')

plot_tsne(tsnes=[tsne_all], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_all.index], legend=True,colour_dict=colour_dicts['cell_type'],
         title='Cell types in integrated data set')
plt.savefig(data_path_h+'tSNE_all_cellType.pdf')

plot_tsne(tsnes=[tsne_all], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[tsne_all.index], legend=False)
plt.savefig(data_path_h+'tSNE_all_cellTypesFine.pdf')

plot_tsne(tsnes=[tsne_fov], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_fov.index], legend=True,colour_dict=colour_dicts['cell_type'],
         title='Cell types in fovelar cells from integrated data set')
plt.savefig(data_path_h+'tSNE_fov_cellType.pdf')

plot_tsne(tsnes=[tsne_fov], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[tsne_fov.index], legend=False)
plt.savefig(data_path_h+'tSNE_fov_cellTypesFine.pdf')

plot_tsne(tsnes=[tsne_per], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_per.index], legend=True,colour_dict=colour_dicts['cell_type'],
         title='Cell types in periferal cells from integrated data set')
plt.savefig(data_path_h+'tSNE_per_cellType.pdf')

plot_tsne(tsnes=[tsne_per], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[tsne_per.index], legend=False)
plt.savefig(data_path_h+'tSNE_per_cellTypesFine.pdf')

# Nonintegrated
plot_tsne(tsnes=[tsne_all_nonint], classes=[dict(zip(col_data.index,col_data['region']))],
          names=[tsne_all_nonint.index], legend=True,colour_dict=colour_dicts['region'])
plt.savefig(data_path_h+'tSNE_all_region_nonintegrated.pdf')

plot_tsne(tsnes=[tsne_all_nonint], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_all_nonint.index], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_all_cellType_nonintegrated.pdf')

plot_tsne(tsnes=[tsne_all_nonint], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[tsne_all_nonint.index], legend=False)
plt.savefig(data_path_h+'tSNE_all_cellTypesFine_nonintegrated.pdf')

plot_tsne(tsnes=[tsne_fov_nonint], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_fov_nonint.index], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_fov_cellType_nonintegrated.pdf')

plot_tsne(tsnes=[tsne_fov_nonint], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[tsne_fov_nonint.index], legend=False)
plt.savefig(data_path_h+'tSNE_fov_cellTypesFine_nonintegrated.pdf')

plot_tsne(tsnes=[tsne_per_nonint], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_per_nonint.index], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_per_cellType_nonintegrated.pdf')

plot_tsne(tsnes=[tsne_per_nonint], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[tsne_per_nonint.index], legend=False)
plt.savefig(data_path_h+'tSNE_per_cellTypesFine_nonintegrated.pdf')


# tSNE added other region

plot_tsne(tsnes=[tsne_fov,tsne_fov_aPer], 
          classes=[dict(zip(tsne_fov.index,['fov']*tsne_fov.shape[0])),
              dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_fov.index,tsne_fov_aPer.index], 
          legend=True,colour_dict=colour_dicts['cell_type'],
          plotting_params = [{'s':0.1,'alpha':0.1},{'s':1,'alpha':0.2}],
         title='Embedding of per (non-integrated) on fov (integrated), coloured by per')
plt.savefig(data_path_h+'tSNE_fov_addedPer.pdf')

plot_tsne(tsnes=[tsne_per,tsne_per_aFov], 
          classes=[dict(zip(tsne_per.index,['per']*tsne_per.shape[0])),
              dict(zip(col_data.index,col_data['cell_type']))],
          names=[tsne_per.index,tsne_per_aFov.index], 
          legend=True,colour_dict=colour_dicts['cell_type'],
         plotting_params = [{'s':0.1,'alpha':0.1},{'s':1,'alpha':0.2}],
         title='Embedding of fov (non-integrated) on per (integrated), coloured by fov')
plt.savefig(data_path_h+'tSNE_per_addedFov.pdf')

# Plot for human samples

samples=[path.split('/')[-1] for path in
    glob.glob(
        '/share/LBI_share/public/retina/human snRNAseq data_10x_ChenLab_ZupanLab/humansn_10x_storage/*matrix')]
plt.rcParams["figure.titlesize"] = 'medium'
for sample_name in samples:
    data_path_sample=data_path_h+sample_name+'_'
    summary=pd.read_table(data_path_sample + 'classified.tsv',index_col=0)
    tsne_h=loadPickle(data_path_sample + 'tsne_all_h.pkl')
    
    plot_tsne(tsnes=[tsne_all,tsne_h], 
          classes=[dict(zip(col_data.index,col_data['cell_type'])),
                   dict(zip(tsne_h.index,['human']*tsne_h.shape[0]))],
          names=[tsne_all.index,tsne_h.index], 
          legend=True,colour_dict=colour_dicts['cell_type'],
         plotting_params = [{'s':0.1,'alpha':0.1},{'s':1,'alpha':0.2}],
             order_legend=colour_dicts['cell_type'].keys(),
             title='Embeding of human data on macaque integrated data, coloured by macaque cell type')
    plt.savefig(data_path_sample+'tSNE_all.pdf')
    
    plot_tsne(tsnes=[tsne_all,tsne_h], 
          classes=[dict(zip(col_data.index,['reference']*tsne_all.shape[0])),
                   dict(zip(summary.index,summary['main_class'].fillna('NA')))],
          names=[tsne_all.index,tsne_h.index], 
          legend=True,colour_dict=colour_dicts['cell_type'],
         plotting_params = [{'s':0.1,'alpha':0.1},{'s':1,'alpha':0.2}],title='Predicted class',
             order_legend=colour_dicts['cell_type'].keys())
    plt.savefig(data_path_sample+'tSNE_all_predicted.pdf')
    
    predicted_with = []
    for row in summary.iterrows():
        row = row[1]
        if row.isna()['main_class']:
            predicted_with.append('unclassified')
        else:
            predicted_with.append(row['N_main_class'])
    plot_tsne(tsnes=[tsne_all, tsne_h],
          classes=[dict(zip(col_data.index, ['reference'] * tsne_all.shape[0])),
                   dict(zip(summary.index,predicted_with))],
          names=[tsne_all.index, tsne_h.index],
          legend=True, colour_dict=colour_dicts['classified'],
          plotting_params=[{'s': 0.1, 'alpha': 0.1}, {'s': 1, 'alpha': 0.2}],title='N methods for main predicted class',
          order_legend=colour_dicts['classified'].keys())
    plt.savefig(data_path_sample+'tSNE_all_predicted_Nmain.pdf')
    
    plt.close('all')




    

