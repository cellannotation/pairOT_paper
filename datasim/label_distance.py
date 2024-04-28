from typing import Dict, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.tools._rank_genes_groups import _Method
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import sqeuclidean
from scipy.stats import spearmanr
from tqdm.notebook import tqdm


def _bures_wasserstein(
    mean1: np.ndarray, mean2: np.ndarray, cov1: np.ndarray, cov2: np.ndarray
) -> float:
    """Compute the Bures-Wasserstein cost between two normal distributions N(mean, cov)."""
    sqrt_cov1 = sqrtm(cov1)
    bures = np.trace(
        cov1 + cov2 - 2 * sqrtm(np.matmul(np.matmul(sqrt_cov1, cov2), sqrt_cov1))
    )
    return sqeuclidean(mean1, mean2) + np.real(bures)


def _calc_cov(
    x_embed: np.ndarray, cell_type_labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute per cell-type covariance matrices."""
    return {
        ct: np.cov(x_embed[cell_type_labels == ct], rowvar=False)
        for ct in np.unique(cell_type_labels)
    }


def _calc_mean(
    x_embed: Union[np.ndarray, csr_matrix], cell_type_labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute per cell-type mean vectors."""
    return {
        ct: np.array(
            x_embed[cell_type_labels == ct].mean(axis=0)
        )  # wrap with np.array in case input matrix is sparse
        for ct in np.unique(cell_type_labels)
    }


def bures_wasserstein_label_distance(
    x_embed_1: np.ndarray,
    x_embed_2: np.ndarray,
    cell_type_labels_1: np.ndarray,
    cell_type_labels_2: np.ndarray,
) -> pd.DataFrame:
    """Compute the Bures-Wasserstein distance matrix between cell-type labels."""
    assert x_embed_1.shape[1] == x_embed_2.shape[1]
    assert x_embed_1.shape[0] == len(cell_type_labels_1)
    assert x_embed_2.shape[0] == len(cell_type_labels_2)

    print("Calcualting mean vectors...")
    means1 = _calc_mean(x_embed_1, cell_type_labels_1)
    means2 = _calc_mean(x_embed_2, cell_type_labels_2)
    print("Calcualting covariance matrices...")
    cov1 = _calc_cov(x_embed_1, cell_type_labels_1)
    cov2 = _calc_cov(x_embed_2, cell_type_labels_2)
    print("Computing label distances...")
    labels1 = list(means1.keys())
    labels2 = list(means2.keys())
    bures_wasserstein_dist_mtx = np.zeros((len(labels1), len(labels2)))
    with tqdm(total=len(labels1) * len(labels2)) as pbar:
        for i, label1 in enumerate(labels1):
            for j, label2 in enumerate(labels2):
                bures_wasserstein_dist_mtx[i, j] = _bures_wasserstein(
                    means1[label1], means2[label2], cov1[label1], cov2[label2]
                )
                pbar.update(1)

    return pd.DataFrame(data=bures_wasserstein_dist_mtx, index=labels1, columns=labels2)


def spearmanr_label_distance(
    x1: csr_matrix,
    x2: csr_matrix,
    cell_type_labels_1: np.ndarray,
    cell_type_labels_2: np.ndarray,
) -> pd.DataFrame:
    """Compute the SpearmanR distance matrix between cell-type labels."""
    assert x1.shape[1] == x2.shape[1]
    assert x1.shape[0] == len(cell_type_labels_1)
    assert x2.shape[0] == len(cell_type_labels_2)

    print("Calculating mean vectors...")
    means1 = _calc_mean(x1, cell_type_labels_1)
    means2 = _calc_mean(x2, cell_type_labels_2)
    labels1 = list(means1.keys())
    labels2 = list(means2.keys())
    print("Computing label distances...")
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
    n_genes: int = 50,
    method: _Method = "t-test_overestim_var",
) -> pd.DataFrame:
    """Compute cell-type label distance matrix based on overlap of differentially expressed genes between clusters."""
    assert cell_type_column in adata1.obs.columns
    assert cell_type_column in adata2.obs.columns

    sc.tl.rank_genes_groups(
        adata1, groupby=cell_type_column, n_genes=n_genes, method=method
    )
    sc.tl.rank_genes_groups(adata2, groupby=cell_type_column, n_genes=n_genes)
    de_genes_adata2 = {
        ct: adata2.uns["rank_genes_groups"]["names"][ct].tolist()
        for ct in adata2.obs[cell_type_column].unique()
    }
    # use Jaccard distance to calculate measure overlap
    de_gene_overlap_distance = 1.0 - (
        sc.tl.marker_gene_overlap(adata1, de_genes_adata2, method="jaccard")
    )

    return de_gene_overlap_distance
