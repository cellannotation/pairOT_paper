from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from tqdm.notebook import tqdm

from datasim.de_testing.selection import (
    select_and_combine_de_results,
    sort_and_filter_de_genes_ova,
    sort_and_filter_de_genes_ava,
)
from datasim.utils import (
    _bures_wasserstein,
    _calc_cov,
    _calc_mean,
    _calc_scaled_jaccard,
)


def _check_dimensions(x1, x2, cell_type_labels_1, cell_type_labels_2):
    assert x1.shape[1] == x2.shape[1]
    assert x1.shape[0] == len(cell_type_labels_1)
    assert x2.shape[0] == len(cell_type_labels_2)


def _check_de_results(adata: anndata.AnnData, cell_type_column: str):
    ct_labels = adata.obs[cell_type_column].unique()
    # Check if all OVA DE results are present
    for ct in ct_labels:
        if ct not in adata.uns["de_res_ova"]:
            raise ValueError(f"OVA DE results for {ct} missing in adata object.")
    # Check if all AVA DE results are present
    for ct1 in ct_labels:
        for ct2 in ct_labels:
            if ct1 != ct2 and ct2 not in adata.uns["de_res_ava"][ct1]:
                raise ValueError(
                    f"AVA DE results for {ct1} vs {ct2} missing in adata object."
                )


def bures_wasserstein_label_distance(
    x_embed_1: np.ndarray,
    x_embed_2: np.ndarray,
    cell_type_labels_1: np.ndarray,
    cell_type_labels_2: np.ndarray,
) -> pd.DataFrame:
    """Compute the Bures-Wasserstein distance matrix between cell-type labels."""
    _check_dimensions(x_embed_1, x_embed_2, cell_type_labels_1, cell_type_labels_2)

    means1 = _calc_mean(x_embed_1, cell_type_labels_1)
    means2 = _calc_mean(x_embed_2, cell_type_labels_2)
    cov1 = _calc_cov(x_embed_1, cell_type_labels_1)
    cov2 = _calc_cov(x_embed_2, cell_type_labels_2)
    labels1 = list(means1.keys())
    labels2 = list(means2.keys())
    bures_wasserstein_dist_mtx = np.zeros((len(labels1), len(labels2)))
    with tqdm(total=len(labels1) * len(labels2)) as pbar:
        for i, label1 in enumerate(labels1):
            for j, label2 in enumerate(labels2):
                bures_wasserstein_dist_mtx[i, j] = _bures_wasserstein(
                    means1[label1], means2[label2], cov1[label1], cov2[label2]
                )
                pbar.update()

    return pd.DataFrame(data=bures_wasserstein_dist_mtx, index=labels1, columns=labels2)


def spearmanr_label_distance(
    x1: csr_matrix,
    x2: csr_matrix,
    cell_type_labels_1: np.ndarray,
    cell_type_labels_2: np.ndarray,
) -> pd.DataFrame:
    """Compute the SpearmanR distance matrix between cell-type labels."""
    _check_dimensions(x1, x2, cell_type_labels_1, cell_type_labels_2)

    means1 = _calc_mean(x1, cell_type_labels_1)
    means2 = _calc_mean(x2, cell_type_labels_2)
    labels1 = list(means1.keys())
    labels2 = list(means2.keys())
    # noinspection PyTypeChecker
    spearman_corr = spearmanr(
        np.vstack([means1[k] for k in labels1]),
        np.vstack([means2[k] for k in labels2]),
        axis=1,
    ).statistic[: len(labels1), len(labels1) :]

    return pd.DataFrame(data=spearman_corr, index=labels1, columns=labels2)


def de_gene_overlap_label_distance(
    adata1: anndata.AnnData,
    adata2: anndata.AnnData,
    cell_type_column: str,
    n_genes_adata1_ova: int = 10,
    n_genes_adata2_ova: int = 20,
    n_genes_adata1_ava: int = 3,
    n_genes_adata2_ava: int = 3,
    overlap_threshold_ava: float = 0.1,
    overlap_n_genes_ava: int = 10,
    adj_p_val_threshold: float = 0.05,
    auroc_threshold: float = 0.6,
    gene_filtering: Literal[
        "standard",
        "strict-pegasus-immune",
        "strict-villani-immune",
        "strict-villani-bonemarrow",
        None,
    ] = "standard",
) -> pd.DataFrame:
    """Compute cell-type label distance matrix based on overlap of differentially expressed genes between clusters."""
    _check_de_results(adata1, cell_type_column)
    _check_de_results(adata2, cell_type_column)

    de_genes_adata1 = select_and_combine_de_results(
        sort_and_filter_de_genes_ova(
            adata1.uns["de_res_ova"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            gene_filtering=gene_filtering,
        ),
        sort_and_filter_de_genes_ava(
            adata1.uns["de_res_ava"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            gene_filtering=gene_filtering,
        ),
        n_genes_ova=n_genes_adata1_ova,
        n_genes_ava=n_genes_adata1_ava,
        overlap_threshold=overlap_threshold_ava,
        overlap_n_genes=overlap_n_genes_ava,
    )
    de_genes_adata2 = select_and_combine_de_results(
        sort_and_filter_de_genes_ova(
            adata2.uns["de_res_ova"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            gene_filtering=gene_filtering,
        ),
        sort_and_filter_de_genes_ava(
            adata2.uns["de_res_ava"],
            aucroc_threshold=auroc_threshold,
            adj_pval_threshold=adj_p_val_threshold,
            gene_filtering=gene_filtering,
        ),
        n_genes_ova=n_genes_adata2_ova,
        n_genes_ava=n_genes_adata2_ava,
        overlap_threshold=overlap_threshold_ava,
        overlap_n_genes=overlap_n_genes_ava,
    )
    jaccard = _calc_scaled_jaccard(
        {ct: set(de_genes_adata1[ct].index) for ct in de_genes_adata1.keys()},
        {ct: set(de_genes_adata2[ct].index) for ct in de_genes_adata2.keys()},
    )

    return 1.0 - jaccard
