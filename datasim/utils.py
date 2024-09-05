import warnings
from typing import Dict, List, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.linalg import sqrtm
from scipy.sparse import csr_matrix
from scipy.spatial.distance import sqeuclidean


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


def _calc_scaled_jaccard(
    markers1: Dict[str, set], markers2: Dict[str, set]
) -> pd.DataFrame:
    """
    Calculate the soft overlap index between the values of two dictionaries.
    """
    soft_overlap = pd.DataFrame(
        np.zeros((len(markers1), len(markers2))),
        index=list(markers1.keys()),
        columns=list(markers2.keys()),
    )
    for c1, marker_genes1 in markers1.items():
        for c2, marker_genes2 in markers2.items():
            try:
                # scale factor to account for different number of marker genes
                scale_factor = len(marker_genes2) / len(marker_genes1)
                intersection = len(marker_genes1.intersection(marker_genes2))
                union = len(marker_genes1.union(marker_genes2))
                soft_overlap.loc[c1, c2] = (intersection / union) * scale_factor
            except ZeroDivisionError:
                soft_overlap.loc[c1, c2] = 0.0
                warnings.warn(
                    f"No marker genes provided for {c1} and {c2}. Setting overlap to 0."
                )

    return soft_overlap


def get_expressed_genes_intersection(
    adata1: anndata.AnnData, adata2: anndata.AnnData, min_counts: float = 0.0
) -> List[str]:
    """
    Return the intersection of the expressed genes between adata1 and adata2.

    Parameters
    ----------
    adata1 : anndata.AnnData
        Query data.
    adata2 : anndata.AnnData
        Reference data.
    min_counts : float
        Minimum number of counts for a gene to be expressed.

    Returns
    -------
    List[str]
        List of genes that are expressed in both in adata1 and adata2.
    """
    genes1 = set(
        adata1.var.index.to_numpy()[
            np.array(adata1.X.sum(axis=0)).flatten() > min_counts
        ]
    )
    genes2 = set(
        adata2.var.index.to_numpy()[
            np.array(adata2.X.sum(axis=0)).flatten() > min_counts
        ]
    )
    return list(genes1 & genes2)


def get_shared_highly_variable_genes(
    adata1: anndata.AnnData, adata2: anndata.AnnData, n_top_genes: int
):
    adata_concat = anndata.concat(
        [adata1, adata2], label="dataset", keys=["query", "ref"]
    )
    sc.pp.normalize_total(adata_concat, target_sum=1e4)
    sc.pp.log1p(adata_concat)
    highly_variable = sc.pp.highly_variable_genes(
        adata_concat,
        n_top_genes=n_top_genes,
        batch_key="dataset",
        inplace=False,
    )
    size_hvg_intersection = highly_variable["highly_variable_intersection"].sum()
    if size_hvg_intersection < n_top_genes:
        warnings.warn(
            f"Only {size_hvg_intersection} of {n_top_genes} genes are highly variable in both datasets."
        )

    return highly_variable["highly_variable"].to_numpy()
