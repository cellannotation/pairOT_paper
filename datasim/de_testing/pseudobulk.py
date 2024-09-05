import os
import shutil
from os.path import join
from pathlib import Path
from typing import Dict, Iterable

import anndata
import numpy as np
import pandas as pd
import pegasus as pg
import rpy2.robjects as ro
import scanpy as sc
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.sparse import csr_matrix

INSTALL_R_PACKAGES = """
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("limma")

install.packages("remotes")
library(remotes)
install_version("rhdf5", version = "2.46.1", repos = "https://bioconductor.org/packages/3.18/bioc")
install_version("Matrix", version = "1.6-0")
install_version("magrittr", version = "2.0.3")
install_version("data.table", version = "1.15.4")
install_version("glue", version = "1.7.0")
install_version("stringr", version = "1.5.1")
"""

R_DE_TEST_CODE = """
library(limma)
library(rhdf5)
library(Matrix)
library(magrittr)
library(data.table)
library(glue)
library(stringr)

# Read in the normalized counts
samps <- h5read(h5_path, "obs")
genes <- h5read(h5_path, "var")
mtx <- h5read(h5_path, "X")
norm_mtx <- sparseMatrix(
   i = as.numeric(mtx[[2]]),
   p = as.numeric(mtx[[3]]),
   x = as.numeric(mtx[[1]]),
   dims = c(length(genes[[1]]), length(samps[[1]])),
   index1 = FALSE
)
colnames(norm_mtx) <- samps[[1]]
rownames(norm_mtx) <- genes[[1]]
norm_mtx <- as.matrix(norm_mtx)
meta_data <- read.csv(meta_path, row.names = 1)
meta_data <- meta_data[colnames(norm_mtx),]
norm_mtx <- norm_mtx[, rownames(meta_data)]
stopifnot(colnames(norm_mtx) == rownames(meta_data))

ret <- data.frame(ID=character(), logfc=double(), P.Value=double())
for (clust in unique(meta_data$cluster)){
   # Make a temporary copy of metadata
   meta_temp <- meta_data
   # define "isclust"
   meta_temp$isclust <- ifelse(meta_temp$cluster == clust, "yes", "no")

   des1 <- with(meta_temp, model.matrix(~isclust))
   fit1 <- lmFit(object = norm_mtx, design = des1)
   fit1 <- eBayes(fit1)
   fit1$genes <- rownames(fit1$coefficients)

   # Get results for all genes
   res <- topTable(fit1, coef = "isclustyes", number = length(rownames(fit1$coefficients)))
   res$cluster <- clust
   ret <- rbind(ret, res[, c('ID', 'logFC', 'P.Value', 'adj.P.Val', 'cluster')])
}

# Now AVA
meta_data$x <- factor(meta_data$cluster)
des1 <- with(meta_data, model.matrix(~ 0 + x ))
ob <- as.matrix(norm_mtx)
fit1 <- lmFit(object = ob, design = des1)
fit1 <- eBayes(fit1)
cluster_pairs <- t(combn(levels(meta_data$x), 2))
cont <- makeContrasts(
   contrasts = lapply(seq(nrow(cluster_pairs)), 
   function(i) { glue("x{cluster_pairs[i,1]} - x{cluster_pairs[i,2]}")}), 
   levels = des1
)
colnames(cont) <- str_replace(colnames(cont), " - ", "vs")
fit2 <- contrasts.fit(fit1, cont)
fit2 <- eBayes(fit2)
de_ava <- rbindlist(lapply(colnames(cont), function(this_coef) {
 x <- topTable(fit2, coef = this_coef, number = nrow(fit1$coefficients))
 this_coef <- str_replace_all(this_coef, "x", "")
 this_coef <- str_replace(this_coef, "vs", " vs ")
 x$coef <- this_coef
 x$gene <- rownames(x)
 x
}))
"""


def _extract_ova_de_results(
    ova_vals, cluster_labels: Iterable[str], mapping: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    de_res_ova = {}
    for clust in cluster_labels:
        pseudo_markers = ova_vals[ova_vals["cluster"] == clust].set_index("ID")
        de_res_ova[mapping[clust]] = pseudo_markers[["logFC", "P.Value", "adj.P.Val"]]

    return de_res_ova


def _extract_ava_de_results(
    ava_vals, cluster_labels: Iterable[str], mapping: Dict[str, str]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    de_res_ava = {}
    for comp in np.unique(ava_vals["coef"]):
        ava_markers = ava_vals[ava_vals["coef"] == comp]
        de_res_ava[comp] = ava_markers[["logFC", "P.Value", "adj.P.Val"]]

    de_res_ava_ = {}
    for ct1 in cluster_labels:
        ct1_ = mapping[ct1]
        de_res_ava_[ct1_] = {}
        for ct2 in cluster_labels:
            ct2_ = mapping[ct2]
            if ct1 != ct2:
                if f"{ct1}:vs{ct2}" in de_res_ava:
                    res = de_res_ava[f"{ct1}:vs{ct2}"]
                elif f"{ct2}:vs{ct1}" in de_res_ava:
                    res = de_res_ava[f"{ct2}:vs{ct1}"].copy()
                    res["logFC"] = -res["logFC"]
                else:
                    raise RuntimeError(f"DE results for {ct1} vs {ct2} not found.")

                de_res_ava_[ct1_][ct2_] = res

    return de_res_ava_


def _get_auroc_scores(
    adata_raw: anndata.AnnData, cell_type_col: str
) -> Dict[str, pd.DataFrame]:
    pg.de_analysis(adata_raw, cluster=cell_type_col)
    pg_markers = pg.markers(adata_raw, alpha=np.inf)
    auroc_scores = {
        ct: pd.concat(
            [
                pg_markers[ct]["up"]["auroc"],
                pg_markers[ct]["down"]["auroc"],
            ]
        )
        for ct in adata_raw.obs[cell_type_col].unique()
    }

    return auroc_scores


def calc_pseudobulk_stats(
    adata: anndata.AnnData,
    cluster_label: str = "cell_type_author",
    sample_label: str = "sample_id",
):
    save_path = "pseudobulk/"
    save_name = "pseudobulk"
    # Replace cluster labels with int labels. Otherwise, R code below won't work
    mapping = {v: f"c_{i}" for i, v in enumerate(adata.obs[cluster_label].unique())}
    adata.obs[cluster_label] = (
        adata.obs[cluster_label].astype(str).replace(mapping).astype("category")
    )
    # Get relevant obs and var
    gene_sum_dict, cell_num_dict = {}, {}
    for samp in adata.obs[sample_label].unique():
        # Iterate across clusters
        for clust in adata.obs[cluster_label].unique():
            dat = adata[
                (adata.obs[sample_label] == samp) & (adata.obs[cluster_label] == clust)
            ]
            if len(dat) < 2:
                continue
            # Add info to my dictionaries
            key = f"{samp}_c{clust}"
            cell_num_dict[key] = {"n_cells": len(dat), "cluster": clust, "sample": samp}
            # Sum the counts
            count_sum = np.array(dat.X.sum(axis=0)).flatten()
            gene_sum_dict[key] = count_sum

    count_mtx = pd.DataFrame(gene_sum_dict, index=adata.var_names)
    meta_mtx = pd.DataFrame.from_dict(
        cell_num_dict, orient="index", columns=["n_cells", "cluster", "sample"]
    )
    # Normalize the matrix
    cols, index = count_mtx.index, count_mtx.columns
    norm_mtx = csr_matrix(count_mtx.T)
    scale = 100000 / norm_mtx.sum(axis=1).A1
    norm_mtx.data *= np.repeat(scale, np.diff(norm_mtx.indptr))
    norm_mtx.data = np.log1p(norm_mtx.data)
    # Save the matrix
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    meta_mtx.to_csv(os.path.join(save_path, f"{save_name}_pseudobulk_meta.csv"))
    # Need to save the norm count matrix as .h5, too slow to save it otherwise
    sc.AnnData(
        X=norm_mtx, obs=pd.DataFrame(index=index), var=pd.DataFrame(index=cols)
    ).write_h5ad(Path(join(save_path, f"{save_name}_pseudobulk_norm_counts.h5ad")))
    ro.globalenv["meta_path"] = join(save_path, f"{save_name}_pseudobulk_meta.csv")
    ro.globalenv["h5_path"] = join(
        save_path, f"{save_name}_pseudobulk_norm_counts.h5ad"
    )
    ro.globalenv["save_path"] = save_path
    ro.globalenv["save_name"] = save_name
    ro.r(R_DE_TEST_CODE)
    # Remove tmp directory again
    shutil.rmtree(save_path)
    # Get DE results from R code
    with localconverter(ro.default_converter + pandas2ri.converter):
        ova_vals = ro.conversion.rpy2py(ro.globalenv["ret"])
        ova_vals = ova_vals.reset_index().drop("index", axis=1)
    with localconverter(ro.default_converter + pandas2ri.converter):
        ava_vals = ro.conversion.rpy2py(ro.globalenv["de_ava"])
        ava_vals["coef"] = [name.replace(" vs ", ":vs") for name in ava_vals["coef"]]
        ava_vals.index = ava_vals["gene"]
    # Extract DE results
    cluster_labels = np.unique(adata.obs[cluster_label])
    inverse_mapping = {v: k for k, v in mapping.items()}
    de_res_ova = _extract_ova_de_results(ova_vals, cluster_labels, inverse_mapping)
    de_res_ava = _extract_ava_de_results(ava_vals, cluster_labels, inverse_mapping)
    # Convert back to original cluster labels
    adata.obs[cluster_label] = (
        adata.obs[cluster_label].astype(str).replace(inverse_mapping).astype("category")
    )
    # calculate AUROC scores with pegasus for OVA results
    auroc_scores = _get_auroc_scores(adata, cluster_label)
    for ct, de_df in de_res_ova.items():
        de_df["auroc"] = auroc_scores[ct]

    return de_res_ova, de_res_ava
