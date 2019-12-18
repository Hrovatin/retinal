library(Seurat)
library(Matrix)

file<-'passedQC.mtx'
matrix<-readMM(file)
file_prefix<-gsub("\\.mtx","",file)
row_names<-read.table(paste(file_prefix,'_genes.tsv',sep=''),sep='\t',header=TRUE)
col_data<-read.table(paste(file_prefix,'_cellData.tsv',sep=''),sep='\t',header=TRUE,row.names=1) 
rownames(matrix)<-row_names$rownames
colnames(matrix)<-rownames(col_data)

# Make object
seurat.obj<-CreateSeuratObject(counts=matrix, meta.data = col_data)
seurat.obj<-SCTransform(seurat.obj, batch_var ='region' ,variable.features.n=NULL,
                          return.only.var.genes = TRUE,conserve.memory=TRUE)
  
#Plot
seurat.obj<-RunPCA(seurat.obj,verbose=FALSE,npcs = 100)

saveRDS(object=seurat.obj,'scaled.rds')

png('elbow.png')
ElbowPlot(seurat.obj)
dev.off()
seurat.obj<-RunUMAP(seurat.obj,dims=1:16,verbose=FALSE,min.dist=0.1)
#seurat.obj<-FindNeighbors(seurat.obj,dims=1:30,verbose=FALSE)
#seurat.obj<-FindClusters(seurat.obj,verbose=FALSE)
png('umap_region.png')
DimPlot(seurat.obj,group.by='region')
dev.off()
png('umap_cell_type.png')
DimPlot(seurat.obj,group.by='cell_type')
dev.off()

# Repeat with regress out MT
seurat.obj<-NULL
seurat.obj<-CreateSeuratObject(counts=matrix, meta.data = col_data)
seurat.obj<-SCTransform(seurat.obj, batch_var ='region' ,variable.features.n=NULL,
                        return.only.var.genes = TRUE,conserve.memory=TRUE, vars.to.regress = "percent.mt")

#Plot
seurat.obj<-RunPCA(seurat.obj,verbose=FALSE,npcs = 100)

saveRDS(object=seurat.obj,'scaled_regressedMT.rds')

png('elbow_regressedMT.png')
ElbowPlot(seurat.obj)
dev.off()
seurat.obj<-RunUMAP(seurat.obj,dims=1:16,verbose=FALSE,min.dist=0.1)
#seurat.obj<-FindNeighbors(seurat.obj,dims=1:30,verbose=FALSE)
#seurat.obj<-FindClusters(seurat.obj,verbose=FALSE)
png('umap_region_regressedMT.png')
DimPlot(seurat.obj,group.by='region')
dev.off()
png('umap_cell_type_regressedMT.png')
DimPlot(seurat.obj,group.by='cell_type')
dev.off()


