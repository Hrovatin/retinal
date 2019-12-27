data_path='/home/khrovatin/retinal/data/counts/'
data_path_h='/home/khrovatin/retinal/data/human/'

import sys
sys.stdout = open(data_path_h+'Human_classificatio_plot_out.txt', 'w')
sys.stderr = open(data_path_h+'Human_classification_plot_err.txt', 'w')
print('********* NEW RUN *******')

#import pylab
import matplotlib as mpl
#import IPython
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.dpi'] = 100
#IPython.display.set_matplotlib_formats('png','pdf', quality=100)
#pylab.rcParams['figure.figsize'] = (5.0, 4.0)



from tsne_functions import *

colour_dicts={'region':{'per':'#478216','fov':'#944368'},
             'cell_type':{'AC':'#bf7195','BC':'#b59a00','EpiImmune':'#75b05b',
                          'HC':'#a1651b','PR':'#7a0606','RGC':'#1c5191','added':'#4d4d4d'}}
#Load data

# tSNE per region

# Plots of individual macaque datasets
plot_tsne(tsnes=[tsne_int], classes=[dict(zip(col_data.index,col_data['region']))],
          names=[data_int.index], legend=True,colour_dict=colour_dicts['region'])
plt.savefig(data_path_h+'tSNE_integrated_region.pdf')

plot_tsne(tsnes=[tsne_int], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[data_int.index], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_integrated_cellType.pdf')

plot_tsne(tsnes=[tsne_int], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[data_int.index], legend=False)
plt.savefig(data_path_h+'tSNE_integrated_cellTypesFine.pdf')

#plot_tsne(tsnes=[tsne_fov], classes=[dict(zip(col_data.index,col_data['cell_type']))],
#          names=[fov_cells], legend=True,colour_dict=colour_dicts['cell_type'])
#plt.savefig(data_path_h+'tSNE_fov_cellType.pdf')

#plot_tsne(tsnes=[tsne_fov], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
#          names=[fov_cells], legend=False)
#plt.savefig(data_path_h+'tSNE_fov_cellTypesFine.pdf')

plot_tsne(tsnes=[tsne_per], classes=[dict(zip(col_data.index,col_data['cell_type']))],
          names=[per_cells], legend=True,colour_dict=colour_dicts['cell_type'])
plt.savefig(data_path_h+'tSNE_per_cellType.pdf')

plot_tsne(tsnes=[tsne_per], classes=[dict(zip(col_data.index,col_data['cell_types_fine']))],
          names=[per_cells], legend=False)
plt.savefig(data_path_h+'tSNE_per_cellTypesFine.pdf')

# tSNE added other region

#plot_tsne(tsnes=[tsne_fov,tsne_fov_aPer], 
#          classes=[dict(zip(col_data.index,col_data['cell_type'])),
#                   dict(zip(per_cells,['added']*len(per_cells)))],
#          names=[fov_cells,per_cells], 
#          legend=True,colour_dict=colour_dicts['cell_type'],
#          plotting_params = [{'s':0.1,'alpha':0.1},{'s':1,'alpha':0.2}])
#plt.savefig(data_path_h+'tSNE_fov_addedPer.pdf')

plot_tsne(tsnes=[tsne_per,tsne_per_aFov], 
          classes=[dict(zip(col_data.index,col_data['cell_type'])),
                   dict(zip(fov_cells,['added']*len(fov_cells)))],
          names=[per_cells,fov_cells], 
          legend=True,colour_dict=colour_dicts['cell_type'],
         plotting_params = [{'s':0.1,'alpha':0.1},{'s':1,'alpha':0.2}])
plt.savefig(data_path_h+'tSNE_per_addedFov.pdf')

