library(Tempora)
library(RCurl)

df <- read.csv('../data/proc_data/r_df.csv')

exprMat <- t(as.matrix(df[, -c(1,(ncol(df) - 2):ncol(df))]))

mdata <- df[, names(df)[c(ncol(df) - 2, ncol(df))]]
names(mdata) <- c('Timepoints', 'Clusters')
mdata['Clusters'] <- mdata['Clusters'] + 1

colnames(exprMat) <- rownames(mdata)
gene_list <- read.csv('gene_list.csv')
rownames(exprMat) <- gene_list$X

torder <- sort(unique(mdata$Timepoints))

clabel <- read.csv('cluster_info.csv')
clabel <- clabel$type

tempora_obj <- CreateTemporaObject(exprMat, meta.data=mdata, timepoint_order=torder, cluster_labels=clabel)

gmt_url = "http://download.baderlab.org/EM_Genesets/current_release/Mouse/symbol/"

#list all the files on the server
filenames <- getURL(gmt_url)
tc <- textConnection(filenames)
contents <- readLines(tc)
close(tc)

#get the gmt that has all the pathways and does not include terms inferred from electronic annotations(IEA)
#start with gmt file that has pathways only
# rx <- gregexpr("(?<=<a href=\")(.*.GOBP_AllPathways_no_GO_iea.*.)(.gmt)(?=\">)", contents, perl = TRUE)

# gmt_file <- unlist(regmatches(contents, rx))
# dest_gmt_file <- file.path(getwd(),gmt_file )
# download.file(
#     paste(gmt_url,gmt_file,sep=""),
#     destfile=dest_gmt_file
# )
gmt_file <- 'Mouse_GOBP_AllPathways_no_GO_iea.gmt'

tempora_obj <- CalculatePWProfiles(tempora_obj, gmt_path=gmt_file, method='gsva',
                                   min.sz=5, max.sz=200, parallel.sz=1)

tempora_obj <- BuildTrajectory(tempora_obj, n_pcs=6, difference_threshold=0.01)
tempora_obj <- PlotTrajectory(tempora_obj)
tempora_obj <- IdentifyVaryingPWs(tempora_obj, pval_threshold=0.1)
PlotVaryingPWs(tempora_obj)