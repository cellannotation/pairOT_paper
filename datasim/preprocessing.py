from os.path import dirname, join
from typing import Tuple

import anndata
import pandas as pd
import scanpy as sc

from datasim.utils import (
    get_expressed_genes_intersection,
    get_shared_highly_variable_genes,
)


def preprocess_adatas(
    adata1: anndata.AnnData,
    adata2: anndata.AnnData,
    n_top_genes: int = 750,
    cell_type_column_adata1: str = "cell_type_author",
    cell_type_column_adata2: str = "cell_type_author",
    sample_column_adata1: str = "sample_id",
    sample_column_adata2: str = "sample_id",
    filter_genes: bool = False,
    n_samples_auroc: int = None,
    n_samples_hvg_selection: int = None,
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
    cell_type_column_adata1: str = "cell_type_author"
        Name of the column in `adata.obs` that contains the cell type labels for adata1.
    cell_type_column_adata2: str = "cell_type_author"
        Name of the column in `adata.obs` that contains the cell type labels for adata2.
    sample_column_adata1: str = "sample_id"
        Name of the column in `adata.obs` that contains the sequencing sample ids/labels for adata1.
    sample_column_adata2: str = "sample_id"
        Name of the column in `adata.obs` that contains the sequencing sample ids/labels for adata1.
    filter_genes: bool = False
        Whether to remove uninformative genes. If true mitochondrial, ribosomal, IncRNA, TCR and BCR genes are removed.
    n_samples_auroc: int = None
        Maximum number of samples to use for AUROC calculation. If None, all samples are used. This can drastically
        reduce computation time for large datasets.
    n_samples_hvg_selection: int = None
        Number of samples to use for highly variable gene selection. If None, all samples are used. This can drastically
        reduce the memory usage for large datasets.

    Returns
    -------
    Tuple[anndata.AnnData, anndata.AnnData]
    """
    from datasim.de_testing.testing import calc_pseudobulk_stats

    adata1.X = adata1.X.astype("float32")
    adata2.X = adata2.X.astype("float32")

    adata1.obs["cell_type_author"] = adata1.obs[cell_type_column_adata1]
    adata1.obs["sample_id"] = adata1.obs[sample_column_adata1]
    adata2.obs["cell_type_author"] = adata2.obs[cell_type_column_adata2]
    adata2.obs["sample_id"] = adata2.obs[sample_column_adata2]
    print(f"adata1: {adata1.shape}")
    print(f"adata2: {adata2.shape}")
    # subset gene space only to filtered genes
    if filter_genes:
        print("Applying uninformative gene filtering...")
        genes_to_filter = pd.read_csv(
            join(dirname(__file__), "de_testing/resources/filtered-genes.csv")
        )["feature_name"].tolist()
        official_gene_names = pd.read_csv(
            join(dirname(__file__), "de_testing/resources/official-genes.csv")
        )["feature_name"].tolist()
        adata1 = adata1[:, adata1.var.index.isin(official_gene_names)]
        adata1 = adata1[:, ~adata1.var.index.isin(genes_to_filter)]
        adata2 = adata2[:, adata2.var.index.isin(official_gene_names)]
        adata2 = adata2[:, ~adata2.var.index.isin(genes_to_filter)]
        adata1 = adata1.copy()
        adata2 = adata2.copy()
        print(f"adata1: {adata1.shape}")
        print(f"adata2: {adata2.shape}")
    # subset gene space to genes that are expressed in both datasets
    print("Sub-setting gene space to genes that are expressed in both datasets...")
    intersection_genes = get_expressed_genes_intersection(adata1, adata2, min_counts=10)
    adata1 = adata1[:, intersection_genes].copy()
    adata2 = adata2[:, intersection_genes].copy()
    print(f"adata1: {adata1.shape}")
    print(f"adata2: {adata2.shape}")
    # calculate DE genes
    print("Calculating differentially-expressed genes...")
    adata1.uns["de_res_ova"], adata1.uns["de_res_ava"] = calc_pseudobulk_stats(
        adata1, n_samples_auroc=n_samples_auroc
    )
    adata2.uns["de_res_ova"], adata2.uns["de_res_ava"] = calc_pseudobulk_stats(
        adata2, n_samples_auroc=n_samples_auroc
    )
    # subset to highly variable genes for Spearman correlation
    print("Sub-setting to highly variable genes...")
    if n_samples_hvg_selection is None:
        highly_variable = get_shared_highly_variable_genes(adata1, adata2, n_top_genes)
    else:
        highly_variable = get_shared_highly_variable_genes(
            sc.pp.subsample(adata1, n_obs=n_samples_hvg_selection, copy=True),
            sc.pp.subsample(adata2, n_obs=n_samples_hvg_selection, copy=True),
            n_top_genes,
        )
    adata1 = adata1[:, highly_variable].copy()
    adata2 = adata2[:, highly_variable].copy()
    print(f"adata1: {adata1.shape}")
    print(f"adata2: {adata2.shape}")

    return adata1, adata2
