from typing import Dict, Union

import anndata
import numpy as np
import pandas as pd
from scanpy.tools._rank_genes_groups import _Method
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import sqeuclidean
from scipy.stats import spearmanr
from tqdm.notebook import tqdm

from datasim.utils import get_differentially_expressed_genes


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


def _calc_jaccard(markers1: Dict[str, set], markers2: Dict[str, set]) -> np.ndarray:
    """
    Calculate jaccard index between the values of two dictionaries.
    """
    jacc_results = np.zeros((len(markers1), len(markers2)))

    for j, marker_group in enumerate(markers1):
        tmp = [
            len(markers2[i].intersection(markers1[marker_group]))
            / len(markers2[i].union(markers1[marker_group]))
            for i in markers2.keys()
        ]
        jacc_results[j, :] = tmp

    return jacc_results


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

    means1 = _calc_mean(x1, cell_type_labels_1)
    means2 = _calc_mean(x2, cell_type_labels_2)
    labels1 = list(means1.keys())
    labels2 = list(means2.keys())
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
    n_genes: int = 15,
    method: _Method = "wilcoxon",
    p_value_threshold: float = 0.01,
) -> pd.DataFrame:
    """Compute cell-type label distance matrix based on overlap of differentially expressed genes between clusters."""
    assert cell_type_column in adata1.obs.columns
    assert cell_type_column in adata2.obs.columns

    de_res_adata1 = get_differentially_expressed_genes(
        adata1,
        cell_type_col=cell_type_column,
        n_genes=n_genes,
        method=method,
        p_value_threshold=p_value_threshold,
    )
    de_res_adata2 = get_differentially_expressed_genes(
        adata2,
        cell_type_col=cell_type_column,
        n_genes=n_genes,
        method=method,
        p_value_threshold=p_value_threshold,
    )
    de_genes_adata1 = {
        ct: set(de_res_adata1[ct]["gene"]) for ct in de_res_adata1.keys()
    }
    de_genes_adata2 = {
        ct: set(de_res_adata2[ct]["gene"]) for ct in de_res_adata2.keys()
    }
    de_gene_overlap = pd.DataFrame(
        _calc_jaccard(de_genes_adata2, de_genes_adata1),
        index=list(de_genes_adata2.keys()),
        columns=list(de_genes_adata1.keys()),
    )
    # use Jaccard distance to measure overlap (optimal transport needs a distance)
    return 1.0 - de_gene_overlap
