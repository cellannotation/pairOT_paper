from typing import Dict, List

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.tools._rank_genes_groups import _Method


def get_expressed_genes_intersection(
    adata1: anndata.AnnData, adata2: anndata.AnnData, min_counts: float = 10.0
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
            np.array(adata1.X.sum(axis=0)).flatten() >= min_counts
        ]
    )
    genes2 = set(
        adata2.var.index.to_numpy()[
            np.array(adata2.X.sum(axis=0)).flatten() >= min_counts
        ]
    )
    return list(genes1 & genes2)


def get_differentially_expressed_genes(
    adata: anndata.AnnData,
    cell_type_col: str,
    n_genes: int = 15,
    method: _Method = "wilcoxon",
    p_value_threshold: float = 0.01,
) -> Dict[str, pd.DataFrame]:
    """
    Compute top `n_genes` differentially expressed genes.

    Sorting of top `n_genes` is done as following:
        1. Select all genes with a p-value <= p_value_threshold
        2. Sort genes according to their log-fold-change (descending order)
        3. Select top `n_genes` genes
    """
    sc.tl.rank_genes_groups(adata, groupby=cell_type_col, method=method)
    cell_types = adata.obs[cell_type_col].unique()

    de_genes = {}
    for cell_type in cell_types:
        de_results = pd.DataFrame(
            {
                "gene": adata.uns["rank_genes_groups"]["names"][cell_type],
                "logfoldchg": adata.uns["rank_genes_groups"]["logfoldchanges"][
                    cell_type
                ],
                "pval_adj": adata.uns["rank_genes_groups"]["pvals_adj"][cell_type],
                "score": adata.uns["rank_genes_groups"]["scores"][cell_type],
            }
        )
        de_genes[cell_type] = (
            de_results.query(f"pval_adj <= {p_value_threshold}")
            .sort_values("logfoldchg", ascending=False)
            .head(n_genes)
            .reset_index(drop=True)
        )

    return de_genes
