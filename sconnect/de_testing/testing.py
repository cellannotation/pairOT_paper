import os
import shutil
from os.path import join
from pathlib import Path
from typing import Dict, Iterable

import anndata
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scanpy as sc
import tqdm
from joblib import Parallel, delayed, parallel_backend
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.sparse import csr_matrix

from sconnect.de_testing.auroc import calc_auroc, csr_to_csc
from sconnect.utils import downsample_indices

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


def _eff_n_jobs(n_jobs: int) -> int:
    """If n_jobs < 0, set it as the number of physical cores _cpu_count"""
    if n_jobs > 0:
        return n_jobs

    import psutil

    _cpu_count = psutil.cpu_count(logical=False)
    if _cpu_count is None:
        _cpu_count = psutil.cpu_count()

    return _cpu_count


def _calc_auroc_parallel_helper(
    start_pos,
    end_pos,
    data,
    indices,
    indptr,
    n1arr,
    n2arr,
    cluster_cumsum,
    first_j,
    second_j,
):
    # Call the original calc_auroc function
    res = calc_auroc(
        start_pos,
        end_pos,
        data,
        indices,
        indptr,
        n1arr,
        n2arr,
        cluster_cumsum,
        first_j,
        second_j,
    )
    # Convert result to a NumPy array to ensure it is picklable
    return np.array(res)


def _calc_auroc(
    x: csr_matrix,
    cluster_labels: pd.Series,
    gene_names: pd.Series,
    n_jobs: int = 1,
) -> np.array:
    n_jobs = _eff_n_jobs(n_jobs)
    if not isinstance(cluster_labels.dtype, pd.CategoricalDtype):
        cluster_labels = cluster_labels.astype("category")
    cluster_labels = cluster_labels.values
    data, indices, indptr = csr_to_csc(
        x.data,
        x.indices,
        x.indptr,
        x.shape[0],
        x.shape[1],
        np.argsort(cluster_labels.codes),
    )
    cluster_cnts = cluster_labels.value_counts()
    n1arr = cluster_cnts.values
    n2arr = x.shape[0] - n1arr
    cluster_cumsum = cluster_cnts.cumsum().values

    first_j, second_j = -1, -1
    posvec = np.where(n1arr > 0)[0]
    if len(posvec) == 2:
        first_j = posvec[0]
        second_j = posvec[1]

    quotient = x.shape[1] // n_jobs
    residue = x.shape[1] % n_jobs
    intervals = []
    start_pos, end_pos = 0, 0
    for i in range(n_jobs):
        end_pos = start_pos + quotient + (i < residue)
        if end_pos == start_pos:
            break
        intervals.append((start_pos, end_pos))
        start_pos = end_pos

    with parallel_backend("loky", inner_max_num_threads=1):
        result_list = Parallel(n_jobs=len(intervals), temp_folder=None)(
            delayed(_calc_auroc_parallel_helper)(
                start_pos,
                end_pos,
                data,
                indices,
                indptr,
                n1arr,
                n2arr,
                cluster_cumsum,
                first_j,
                second_j,
            )
            for start_pos, end_pos in intervals
        )

    return pd.DataFrame(
        np.concatenate(result_list, axis=0),
        index=gene_names.values,
        columns=[f"{x}:auroc" for x in cluster_labels.categories],
    )


def _get_auroc_scores_ova(
    x: csr_matrix,
    cluster_labels: pd.Series,
    gene_names: pd.Series,
    n_samples_max: int = None,
    n_jobs: int = 1,
) -> Dict[str, pd.Series]:
    if n_samples_max is None:
        auroc_scores = _calc_auroc(x, cluster_labels, gene_names, n_jobs=n_jobs)
    else:
        idxs = downsample_indices(
            cluster_labels.to_numpy(), n_samples_max, random_state=0
        )
        auroc_scores = _calc_auroc(
            x[idxs, :], cluster_labels.iloc[idxs], gene_names, n_jobs=n_jobs
        )
    return {ct: auroc_scores[f"{ct}:auroc"] for ct in cluster_labels.unique()}


def _get_auroc_scores_ava(
    x: csr_matrix,
    cluster_labels: pd.Series,
    gene_names: pd.Series,
    n_samples_max: int = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    clusters = sorted(cluster_labels.unique())
    n_clusters = len(clusters)
    auroc_scores = {ct: {} for ct in clusters}
    with tqdm.tqdm(
        total=int(n_clusters * (n_clusters - 1) / 2),
        desc="Calculating AVA AUROC scores",
    ) as pbar:
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                ct1, ct2 = clusters[i], clusters[j]
                x_subset = x[cluster_labels.isin([ct1, ct2]), :]
                cluster_labels_subset = cluster_labels[
                    cluster_labels.isin([ct1, ct2])
                ].cat.remove_unused_categories()

                if n_samples_max is None:
                    scores_subset = _get_auroc_scores_ova(
                        x_subset,
                        cluster_labels_subset,
                        gene_names,
                    )
                else:
                    idxs = downsample_indices(
                        cluster_labels_subset.to_numpy(), n_samples_max, random_state=0
                    )
                    scores_subset = _get_auroc_scores_ova(
                        x_subset[idxs, :],
                        cluster_labels_subset.iloc[idxs],
                        gene_names,
                        n_jobs=-1,
                    )
                auroc_scores[ct1][ct2] = scores_subset[ct1]
                auroc_scores[ct2][ct1] = scores_subset[ct2]
                pbar.update()

    return auroc_scores


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


def calc_pseudobulk_stats(
    adata: anndata.AnnData,
    cluster_label: str = "cell_type_author",
    sample_label: str = "sample_id",
    n_samples_auroc: int = None,
):
    if not isinstance(adata.X, csr_matrix):
        adata.X = csr_matrix(adata.X)
    if not adata.obs[cluster_label].dtype.name == "category":
        adata.obs[cluster_label] = adata.obs[cluster_label].astype("category")

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
    # calculate AUROC scores for OVA results
    # noinspection PyTypeChecker
    auroc_scores_ova = _get_auroc_scores_ova(
        adata.X,
        adata.obs[cluster_label],
        adata.var.index.to_series(),
        n_samples_auroc,
    )
    for ct, de_df in de_res_ova.items():
        de_df["auroc"] = auroc_scores_ova[ct]
    # calculate AUROC scores for AVA results
    # noinspection PyTypeChecker
    auroc_scores_ava = _get_auroc_scores_ava(
        adata.X,
        adata.obs[cluster_label],
        adata.var.index.to_series(),
        n_samples_auroc,
    )
    for ct1, de_res in de_res_ava.items():
        for ct2, de_df in de_res.items():
            de_df["auroc"] = auroc_scores_ava[ct1][ct2]

    return de_res_ova, de_res_ava
