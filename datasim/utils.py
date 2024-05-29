from typing import Dict

import anndata
import pandas as pd
import scanpy as sc
from scanpy.tools._rank_genes_groups import _Method


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
