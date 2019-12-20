import sys

import pandas as pd

from Orange.data import MISSING_VALUES, Table, Domain
from orangecontrib.bioinformatics.utils import serverfiles
from orangecontrib.bioinformatics.widgets.utils.data import TAX_ID, GENE_ID_COLUMN, GENE_AS_ATTRIBUTE_NAME

sys.path.append(sys.path.abspath('../Foo'))

data_path='/home/karin/Documents/retinal/data/'

markers_retinal=pd.read_csv(data_path+'markers.csv')
#markers_retinal['Marker']=[marker.strip() for marker in markers_retinal['Marker']]

serverfiles_domain = 'marker_genes'
#found_sources = {}
#found_sources.update(serverfiles.ServerFiles().allinfo(serverfiles_domain))
markers_orange=pd.DataFrame()
for file_name in ['cellMarker_gene_markers.tab','panglao_gene_markers.tab']:
    serverfiles.update(serverfiles_domain, file_name)
    file_path = serverfiles.localpath_download(serverfiles_domain, file_name)
    data=Table(file_path)
    old_domain = data.domain
    new_domain = Domain(
        [],
        metas=[
            old_domain['Organism'],
            old_domain['Name'],
            old_domain['Entrez ID'],
            old_domain['Cell Type'],
            old_domain['Function'],
            old_domain['Reference'],
            old_domain['URL'],
        ],
    )
    data = data.transform(new_domain)
    data=pd.DataFrame(data.metas,columns=[var.name for var in data.domain.metas])
    data['Data Base']=[file_name.split('_')[0]]*data.shape[0]
    markers_orange=markers_orange.append(data)

markers_orange=markers_orange[markers_orange['Organism'] =='Human']

present=[]
absent=[]
for marker in markers_retinal['Marker'].unique():
    if marker in markers_orange['Name'].values:
        present.append(marker)
    else:
        absent.append(marker)
