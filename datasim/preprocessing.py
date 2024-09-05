from typing import Tuple

import anndata

from datasim.utils import (
    get_expressed_genes_intersection,
    get_shared_highly_variable_genes,
)


def preprocess_adatas(
    adata1: anndata.AnnData,
    adata2: anndata.AnnData,
    n_top_genes: int = 750,
    cell_type_column: str = "cell_type_author",
    sample_column: str = "sample_id",
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Do the following preprocessing steps:
        1. Subset gene space to genes that are expressed in both datasets.
        2. Calculate differentially-expressed (DE) genes for each cluster.
        3. Subset to highly variable genes which are used to calculate the Spearman correlation between two cells.

    Parameters
    ----------
    adata1: anndata.AnnData
        Query data.
    adata2: anndata.AnnData
        Reference data.
    n_top_genes: int = 750
        Number of highly variable genes to use to calculate the Spearman correlation between two cells.
    cell_type_column: str = "cell_type_author"
        Name of the column in `adata.obs` that contains the cell type labels.
    sample_column: str = "sample_id"
        Name of the column in `adata.obs` that contains the sequencing sample ids/labels.

    Returns
    -------
    Tuple[anndata.AnnData, anndata.AnnData]
    """
    from datasim.de_testing.pseudobulk import calc_pseudobulk_stats

    adata1.obs["cell_type_author"] = adata1.obs[cell_type_column]
    adata1.obs["sample_id"] = adata1.obs[sample_column]
    adata2.obs["cell_type_author"] = adata2.obs[cell_type_column]
    adata2.obs["sample_id"] = adata2.obs[sample_column]
    # subset gene space to genes that are expressed in both datasets
    intersection_genes = get_expressed_genes_intersection(adata1, adata2, min_counts=10)
    adata1 = adata1[:, intersection_genes].copy()
    adata2 = adata2[:, intersection_genes].copy()
    # calculate DE genes
    adata1.uns["de_res_ova"], adata1.uns["de_res_ava"] = calc_pseudobulk_stats(adata1)
    adata2.uns["de_res_ova"], adata2.uns["de_res_ava"] = calc_pseudobulk_stats(adata2)
    # subset to highly variable genes for Spearman correlation
    highly_variable = get_shared_highly_variable_genes(adata1, adata2, n_top_genes)
    adata1 = adata1[:, highly_variable].copy()
    adata2 = adata2[:, highly_variable].copy()

    return adata1, adata2
