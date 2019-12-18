if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# BiocManager::install("SingleCellExperiment")
library(scater)
library(SingleCellExperiment)
# library(biomaRt)
# library(scran)
library(Matrix)

data_path='/home/karin/Documents/retinal/data/counts/'
mt_genes<-read.table('/home/karin/Documents/retinal/data/macaque_anno_MT.tsv',sep='\t',header=TRUE)$Gene
files=list.files(path = data_path, pattern = '*_expression_above500_sparse.mtx')
row_names=read.table(paste(data_path,'counts_RowNames.tsv',sep=''),sep='\t',header=TRUE) 
for (file in files){
  file_prefix<-gsub("_above500_sparse\\.mtx","",file)
  print(paste('*********************','Sample:',file_prefix,'************'))
  
  #********* Load data
  matrix<-readMM(paste(data_path,file,sep=''))
  col_names<-read.table(paste(data_path,file_prefix,'.txt_cells.csv',sep=''),sep=',',header=TRUE)
  rownames(matrix)<-row_names$rownames
  colnames(matrix)<-colnames(col_names)
  sce <- SingleCellExperiment(list(counts=matrix))
  
  #*********** QC
  is.mito <- rownames(sce) %in% mt_genes
  sce <- calculateQCMetrics(sce , feature_controls=list( Mt=is.mito))
  # Statistics
  par(mfrow=c(2,2), mar=c(8, 8, 1, 1))
  # print(hist(sce$total_counts/1e6, xlab="Library sizes (millions)", main=paste('Before QC',file_prefix),
  #            breaks=20, col="grey80", ylab="Number of cells"))
  # print(hist(sce$total_features_by_counts, xlab="Number of expressed genes", main=paste('Before QC',file_prefix),
  #            breaks=20, col="grey80", ylab="Number of cells"))
  # print(hist(sce$pct_counts_Mt, xlab="Mitochondrial proportion (%)",
  #            ylab="Number of cells", breaks=20, main=paste('Before QC',file_prefix), col="grey80"))
  par(mfrow=c(1,1))
  
  # Filter
  # Done in python: feature500.drop<-sce$total_features_by_counts<=500
  libsize.drop <- isOutlier(sce$total_counts, nmads=3, type="both", log=TRUE)
  feature.drop <- isOutlier(sce$total_features_by_counts, nmads=3, type="both", log=TRUE)
  mito.drop <- isOutlier(sce$pct_counts_Mt, nmads=4, type="higher")
  sce <- sce[,!(libsize.drop | feature.drop |mito.drop)]
  print(paste('*******To be removed',file_prefix))
  print(data.frame(ByLibSize=sum(libsize.drop), ByFeature=sum(feature.drop), ByMito=sum(mito.drop),Remaining=ncol(sce)),row.names = FALSE)
  
  # Statistics after
  par(mfrow=c(2,2), mar=c(8, 8, 1, 1))
  # print(hist(sce$total_counts/1e6, xlab="Library sizes (millions)", main=paste('After QC',file_prefix),
  #            breaks=20, col="grey80", ylab="Number of cells"))
  # print(hist(sce$total_features_by_counts, xlab="Number of expressed genes", main=paste('After QC',file_prefix),
  #            breaks=20, col="grey80", ylab="Number of cells"))
  # print(hist(sce$pct_counts_Mt, xlab="Mitochondrial proportion (%)",
  #            ylab="Number of cells", breaks=20, main=paste('After QC',file_prefix), col="grey80"))
  par(mfrow=c(1,1))
  # "Feature control" is mt
  # print(plotHighestExprs(sce, n=50) + theme(axis.text=element_text(size=6),
  #                                           axis.title=element_text(size=14,face="bold")))
  write.table(as.data.frame(colnames(sce)), file = paste(data_path,file_prefix,'_namesPassedQC.csv',sep=''),row.names=FALSE,col.names=FALSE,sep=',')
}





#*******Cell cycle
# Annotations saved in retinal/data/counts/Rdata
# ensembl = useMart("ensembl",dataset="mfascicularis_gene_ensembl")
# original_EnsID<-getBM(attributes=c('ensembl_gene_id','hgnc_symbol'), 
#                       filters = 'hgnc_symbol', 
#                       values = rownames(sce), 
#                       mart =ensembl)
# rowData(sce)$ensembl_gene_id <- original_EnsID[match(rownames(sce),original_EnsID$hgnc_symbol),'ensembl_gene_id']
# human_EnsID<-getBM(attributes=c('hsapiens_homolog_ensembl_gene','ensembl_gene_id'), 
#                    filters = 'ensembl_gene_id', 
#                    values = original_EnsID$ensembl_gene_id, 
#                    mart =ensembl)
# rowData(sce)$human_id <- human_EnsID[match(rowData(sce)$ensembl_gene_id,human_EnsID$ensembl_gene_id),'hsapiens_homolog_ensembl_gene']
# cicle.pairs <- readRDS(system.file("exdata", "human_cycle_markers.rds", package="scran"))
#
#Not all genes used for annotations are present + annotation is based on human genes
# G1<-unique(c(cicle.pairs$G1$first,cicle.pairs$G1$second))
# print(paste('N all G1:',length(G1), 'vs. N in data G1:',count(G1 %in% rowData(sce)$human_id)))
# "N all G1: 1263 vs. N in data G1: 838"
# G2M<-unique(c(cicle.pairs$G2M$first,cicle.pairs$G2M$second))
# print(paste('N all G2M:',length(G2M), 'vs. N in data G2M:',count(G2M %in% rowData(sce)$human_id)))
# "N all G2M: 1269 vs. N in data G2M: 848"
# S<-unique(c(cicle.pairs$S$first,cicle.pairs$S$second))
# print(paste('N all S:',length(S), 'vs. N in data S:',count(S %in% rowData(sce)$human_id)))
# "N all S: 1258 vs. N in data S: 845"
# 
# Cycle asignment and filtering
# cicle.assignments <- cyclone(sce, cicle.pairs, gene.names=rowData(sce)$human_id)
# plot(cicle.assignments$score$G1, cicle.assignments$score$G2M, xlab="G1 score", ylab="G2/M score", pch=16)
# sce <- sce[,assignments$phases=="G1"]
# 
