from os.path import dirname, join
from typing import Dict, Literal, Union

import pandas as pd

from datasim.utils import _calc_scaled_jaccard

GENE_FILTERING = {
    "standard": join(dirname(__file__), "resources/filtered-genes.csv"),
    "strict-pegasus-immune": join(
        dirname(__file__), "resources/filtered-genes-strict-pegasus-immune.csv"
    ),
    "strict-villani-immune": join(
        dirname(__file__), "resources/filtered-genes-strict-villani-immune.csv"
    ),
    "strict-villani-bonemarrow": join(
        dirname(__file__), "resources/filtered-genes-strict-villani-bonemarrow.csv"
    ),
}


def _filter_de_genes(
    de_res: Dict[str, pd.DataFrame],
    gene_filtering: Literal[
        "standard",
        "strict-pegasus-immune",
        "strict-villani-immune",
        "strict-villani-bonemarrow",
        None,
    ] = "standard",
):
    de_res_return = {}
    # subset DE results to relevant genes
    gene_names_filtered = pd.read_csv(GENE_FILTERING[gene_filtering])["feature_name"]
    for ct, de_df in de_res.items():
        gene_names = [elem for elem in gene_names_filtered if elem in de_df.index]
        de_res_return[ct] = de_df[de_df.index.isin(gene_names)].copy()

    return de_res_return


def sort_and_filter_de_genes_ova(
    de_res: Dict[str, pd.DataFrame],
    aucroc_threshold: float = 0.6,
    adj_pval_threshold: float = 0.05,
    gene_filtering: Literal[
        "standard",
        "strict-pegasus-immune",
        "strict-villani-immune",
        "strict-villani-bonemarrow",
        None,
    ] = "standard",
) -> Dict[str, pd.DataFrame]:
    """Select top-`n_genes` differentially expressed genes from the DE results."""
    # Subset DE results to relevant genes
    if gene_filtering is not None:
        de_res = _filter_de_genes(de_res, gene_filtering)
    # Sort and filter DE genes
    top_de_genes = {}
    for ct, de_df in de_res.items():
        top_de_genes[ct] = (
            de_df.sort_values("logFC", ascending=False)
            .query(f"`logFC` >= 1.")
            .query(f"`adj.P.Val` <= {adj_pval_threshold}")
            .query(f"`auroc` >= {aucroc_threshold}")
            .copy()
        )

    return top_de_genes


def sort_and_filter_de_genes_ava(
    de_res: Dict[str, Dict[str, pd.DataFrame]],
    aucroc_threshold: float = 0.6,
    adj_pval_threshold: float = 0.05,
    gene_filtering: Literal[
        "standard",
        "strict-pegasus-immune",
        "strict-villani-immune",
        "strict-villani-bonemarrow",
        None,
    ] = "standard",
):
    clusters = list(sorted(de_res.keys()))
    top_de_genes_ava = {ct: {} for ct in clusters}
    for ct1 in clusters:
        res = de_res[ct1]
        # Subset DE results to relevant genes
        if gene_filtering is not None:
            res = _filter_de_genes(res, gene_filtering)
        # Sort and filter DE genes
        for ct2 in clusters:
            if ct1 != ct2:
                top_de_genes_ava[ct1][ct2] = (
                    res[ct2]
                    .sort_values("logFC", ascending=False)
                    .query(f"`logFC` >= 1.")
                    .query(f"`adj.P.Val` <= {adj_pval_threshold}")
                    .query(f"`auroc` >= {aucroc_threshold}")
                    .copy()
                )

    return top_de_genes_ava


def select_and_combine_de_results(
    de_res_ova: Dict[str, pd.DataFrame],
    de_res_ava: Dict[str, Dict[str, pd.DataFrame]],
    n_genes_ova: Union[int, None] = 10,
    n_genes_ava: Union[int, None] = 3,
    overlap_threshold: float = 0.1,
    overlap_n_genes: int = 10,
    remove_duplicated_genes: bool = True,
) -> Dict[str, pd.DataFrame]:
    # Find cluster to refine
    de_overlap = _calc_scaled_jaccard(
        {ct: set(res.head(overlap_n_genes).index) for ct, res in de_res_ova.items()},
        {ct: set(res.head(overlap_n_genes).index) for ct, res in de_res_ova.items()},
    )
    clusters_to_refine = {}
    for col in de_overlap.columns:
        o = de_overlap.loc[col, :]
        clusters_to_refine[col] = o[o >= overlap_threshold].index.tolist()
        if col in clusters_to_refine[col]:
            clusters_to_refine[col].remove(col)
    # Combine DE results
    combined_de_results = {}
    for ct, de_df_ova in de_res_ova.items():
        res = [
            de_df_ova.head(n_genes_ova)[["logFC", "adj.P.Val", "auroc"]].assign(
                reference="all"
            )
        ]
        for ct_refined in clusters_to_refine[ct]:
            res.append(
                de_res_ava[ct][ct_refined]
                .head(n_genes_ava)[["logFC", "adj.P.Val", "auroc"]]
                .assign(reference=ct_refined)
            )
        res = pd.concat(res).sort_values("logFC", ascending=False)

        if remove_duplicated_genes:
            reference = res["reference"].groupby(res.index).apply(lambda x: list(x))
            res = res[~res.index.duplicated(keep="first")]
            res["reference"] = reference
        combined_de_results[ct] = res.copy()

    return combined_de_results
